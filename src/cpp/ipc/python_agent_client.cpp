#include "python_agent_client.h"
#include "../utils/helpers.h"
#include <boost/asio/read_until.hpp>
#include <boost/asio/write.hpp>

namespace atomic {
namespace ipc {

/**
 * @brief Constructs a PythonAgentConnection for a given IO context and Unix domain socket path.
 *
 * @param io_context The Boost.Asio io_context used to run asynchronous I/O for this connection.
 * @param socket_path Filesystem path to the Unix domain socket this connection will use.
 */
PythonAgentConnection::PythonAgentConnection(
    boost::asio::io_context& io_context,
    const std::string& socket_path
) : io_context_(io_context), socket_(io_context), socket_path_(socket_path) {}

/**
 * @brief Destroy the connection object, ensuring the IPC socket and associated resources are closed.
 *
 * Ensures any active connection is terminated and per-connection resources are released before
 * the object is destroyed.
 */
PythonAgentConnection::~PythonAgentConnection() {
    disconnect();
}

/**
 * @brief Attempts to connect the socket to the configured Unix domain socket path and start the read loop.
 *
 * Attempts to establish a connection to the Python agent endpoint and, on success, marks the connection as
 * established and starts the asynchronous read loop.
 *
 * @param timeout_ms Connection attempt timeout in milliseconds (upper bound for the operation).
 * @return true if the socket was connected and the asynchronous read loop was started, false otherwise.
 */
bool PythonAgentConnection::connect(int timeout_ms) {
    try {
        stream_protocol::endpoint ep(socket_path_);
        
        boost::system::error_code ec;
        socket_.connect(ep, ec);
        
        if (ec) {
            LOG_ERROR("Failed to connect to Python agent: ", ec.message());
            return false;
        }
        
        connected_ = true;
        async_read();
        LOG_INFO("Connected to Python agent at ", socket_path_);
        return true;
        
    } catch (const std::exception& e) {
        LOG_ERROR("Exception connecting to Python agent: ", e.what());
        return false;
    }
}

/**
 * @brief Close the IPC socket and mark this connection as disconnected.
 *
 * If the connection is currently open, closes the underlying socket and updates
 * internal state so the connection is no longer considered connected. Safe to
 * call when already disconnected (no action taken).
 */
void PythonAgentConnection::disconnect() {
    if (connected_) {
        boost::system::error_code ec;
        socket_.close(ec);
        connected_ = false;
        LOG_DEBUG("Disconnected from Python agent");
    }
}

/**
 * Sends a request to the Python agent and waits for its corresponding response.
 *
 * The request is serialized and written to the socket; the method blocks until a matching
 * response is received or the timeout elapses. The response is matched by the request's
 * `request_id`.
 *
 * @param request The request to send. Its `request_id` is used to correlate the response.
 * @param timeout_ms Maximum time to wait for a response, in milliseconds.
 * @return LLMResponse The response matched to `request.request_id`.
 *
 * @throws std::runtime_error If the connection is not established or if the request times out.
 * @throws std::exception Rethrows exceptions from socket write or I/O operations.
 */
LLMResponse PythonAgentConnection::send_request(const LLMRequest& request, int timeout_ms) {
    if (!connected_) {
        throw std::runtime_error("Not connected to Python agent");
    }
    
    auto pending = std::make_shared<PendingRequest>();
    pending->request_id = request.request_id;
    pending->streaming = false;
    
    {
        std::lock_guard<std::mutex> lock(mutex_);
        pending_requests_[request.request_id] = pending;
    }
    
    Message msg;
    msg.id = utils::generate_uuid();
    msg.type = MessageType::REQUEST;
    msg.timestamp = utils::timestamp_ms();
    msg.payload = request.to_json();
    
    std::string data = msg.serialize() + "\n";
    
    try {
        boost::asio::write(socket_, boost::asio::buffer(data));
        
        std::unique_lock<std::mutex> lock(mutex_);
        bool completed = cv_.wait_for(
            lock,
            std::chrono::milliseconds(timeout_ms),
            [&pending]() { return pending->completed; }
        );
        
        if (!completed) {
            pending_requests_.erase(request.request_id);
            throw std::runtime_error("Request timeout");
        }
        
        pending_requests_.erase(request.request_id);
        return pending->response;
        
    } catch (const std::exception& e) {
        std::lock_guard<std::mutex> lock(mutex_);
        pending_requests_.erase(request.request_id);
        throw;
    }
}

/**
 * @brief Sends a streaming LLM request to the Python agent and registers callbacks for received chunks and errors.
 *
 * Registers a streaming pending request keyed by the request's `request_id`, serializes the request into a message,
 * and writes it to the connected Unix-domain socket. If the connection is not established or the write fails,
 * the `on_error` callback is invoked with an error message and the pending request is removed.
 *
 * @param request The LLM request to send; its `request_id` is used to correlate subsequent stream chunks.
 * @param on_chunk Callback invoked for each incoming stream chunk associated with this request.
 * @param on_error Callback invoked with an error message if sending the request fails or the connection is not available.
 * @param timeout_ms Timeout for the request in milliseconds (used to coordinate request lifecycle and timeouts).
 */
void PythonAgentConnection::send_streaming_request(
    const LLMRequest& request,
    StreamCallback on_chunk,
    ErrorCallback on_error,
    int timeout_ms
) {
    if (!connected_) {
        on_error("Not connected to Python agent");
        return;
    }
    
    auto pending = std::make_shared<PendingRequest>();
    pending->request_id = request.request_id;
    pending->streaming = true;
    pending->stream_callback = on_chunk;
    pending->error_callback = on_error;
    
    {
        std::lock_guard<std::mutex> lock(mutex_);
        pending_requests_[request.request_id] = pending;
    }
    
    Message msg;
    msg.id = utils::generate_uuid();
    msg.type = MessageType::REQUEST;
    msg.timestamp = utils::timestamp_ms();
    msg.payload = request.to_json();
    
    std::string data = msg.serialize() + "\n";
    
    try {
        boost::asio::write(socket_, boost::asio::buffer(data));
    } catch (const std::exception& e) {
        std::lock_guard<std::mutex> lock(mutex_);
        pending_requests_.erase(request.request_id);
        on_error(e.what());
    }
}

/**
 * @brief Sends a health-check message to the Python agent to verify the connection.
 *
 * If the connection is not established, the function returns false immediately.
 *
 * @return `true` if the health-check message was sent successfully, `false` otherwise.
 */
bool PythonAgentConnection::health_check() {
    if (!connected_) return false;
    
    try {
        Message msg;
        msg.id = utils::generate_uuid();
        msg.type = MessageType::HEALTH_CHECK;
        msg.timestamp = utils::timestamp_ms();
        
        std::string data = msg.serialize() + "\n";
        boost::asio::write(socket_, boost::asio::buffer(data));
        
        return true;
    } catch (const std::exception& e) {
        LOG_WARNING("Health check failed: ", e.what());
        return false;
    }
}

/**
 * @brief Starts an asynchronous read loop that receives newline-delimited messages and dispatches them.
 *
 * Initiates an asynchronous read until a newline is encountered on the connection socket. For each
 * complete line received, the line is passed to handle_message() and the read loop is re-issued to
 * continue receiving subsequent messages. On read error the connection is marked as not connected
 * and an error is logged.
 */
void PythonAgentConnection::async_read() {
    boost::asio::async_read_until(
        socket_,
        read_buffer_,
        '\n',
        [this](const boost::system::error_code& ec, std::size_t bytes_transferred) {
            if (!ec) {
                std::istream is(&read_buffer_);
                std::string line;
                std::getline(is, line);
                
                handle_message(line);
                async_read();
            } else {
                LOG_ERROR("Read error: ", ec.message());
                connected_ = false;
            }
        }
    );
}

/**
 * @brief Process a single serialized IPC message and dispatch its contents to in-flight requests.
 *
 * Parses a newline-terminated JSON message, then:
 * - For RESPONSE messages: associates the deserialized response with the matching pending request and marks it completed.
 * - For STREAM_CHUNK messages: invokes the pending request's stream callback with the chunk and, if the chunk is final, marks completion and removes the pending entry.
 * - For ERROR messages: invokes the pending request's error callback with the error text, marks completion, and removes the pending entry.
 *
 * Exceptions during parsing or dispatch are logged and swallowed.
 *
 * @param data Serialized message payload (one line, JSON) received from the socket.
 */
void PythonAgentConnection::handle_message(const std::string& data) {
    try {
        Message msg = Message::deserialize(data);
        
        if (msg.type == MessageType::RESPONSE) {
            auto resp = LLMResponse::from_json(msg.payload);
            
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = pending_requests_.find(resp.request_id);
            if (it != pending_requests_.end()) {
                auto pending = it->second;
                pending->response = resp;
                pending->completed = true;
                cv_.notify_all();
            }
            
        } else if (msg.type == MessageType::STREAM_CHUNK) {
            auto chunk = StreamChunk::from_json(msg.payload);
            
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = pending_requests_.find(chunk.request_id);
            if (it != pending_requests_.end() && it->second->streaming) {
                auto pending = it->second;
                if (pending->stream_callback) {
                    pending->stream_callback(chunk);
                }
                
                if (chunk.is_final) {
                    pending->completed = true;
                    pending_requests_.erase(it);
                }
            }
            
        } else if (msg.type == MessageType::ERROR) {
            std::string request_id = msg.payload.at("request_id").as_string().c_str();
            std::string error = msg.payload.at("error").as_string().c_str();
            
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = pending_requests_.find(request_id);
            if (it != pending_requests_.end()) {
                auto pending = it->second;
                if (pending->error_callback) {
                    pending->error_callback(error);
                }
                pending->completed = true;
                pending_requests_.erase(it);
            }
        }
        
    } catch (const std::exception& e) {
        LOG_ERROR("Error handling message: ", e.what());
    }
}

/**
     * @brief Construct a PythonAgentClient configured for IPC to the Python agent.
     *
     * Initializes the client with the provided IPC configuration and prepares the
     * internal io_context work guard so the IO context remains alive until the
     * client is explicitly stopped.
     *
     * @param config IPC configuration (connection pool size, timeouts, socket path, etc.)
     */
PythonAgentClient::PythonAgentClient(const utils::IPCConfig& config)
    : config_(config), work_(std::make_unique<boost::asio::io_context::work>(io_context_)) {}

/**
 * @brief Cleans up the client and ensures all background work is stopped.
 *
 * Stops background threads, disconnects pooled connections, and releases IO resources
 * held by the client before destruction.
 */
PythonAgentClient::~PythonAgentClient() {
    stop();
}

/**
 * @brief Starts the Python agent client, initializing the connection pool and launching IO and health-check threads.
 *
 * Attempts to create up to config_.connection_pool_size connections to the configured socket path,
 * adds successfully connected instances to the pool, starts worker threads to run the io_context_,
 * and starts the health-check loop in a dedicated thread.
 *
 * @return true if the client is running after the call (already running or successfully started),
 *         false if startup failed because no connections could be created.
 */
bool PythonAgentClient::start() {
    if (running_) return true;
    
    LOG_INFO("Starting Python agent client with ", config_.connection_pool_size, " connections");
    
    for (int i = 0; i < config_.connection_pool_size; ++i) {
        auto conn = std::make_shared<PythonAgentConnection>(io_context_, config_.socket_path);
        
        if (conn->connect(5000)) {
            connection_pool_.push_back(conn);
            available_connections_.push(conn);
        } else {
            LOG_WARNING("Failed to create connection ", i);
        }
    }
    
    if (connection_pool_.empty()) {
        LOG_ERROR("Failed to create any connections to Python agent");
        return false;
    }
    
    for (int i = 0; i < 2; ++i) {
        worker_threads_.emplace_back([this]() {
            io_context_.run();
        });
    }
    
    running_ = true;
    
    health_check_thread_ = std::thread([this]() {
        health_check_loop();
    });
    
    LOG_INFO("Python agent client started successfully");
    return true;
}

/**
 * @brief Stops the Python agent client and tears down its resources.
 *
 * Stops the client's background operation, waits for and joins the health-check and worker threads,
 * stops the IO context, disconnects and clears all pooled connections, and releases internal work.
 */
void PythonAgentClient::stop() {
    if (!running_) return;
    
    running_ = false;
    
    if (health_check_thread_.joinable()) {
        health_check_thread_.join();
    }
    
    work_.reset();
    io_context_.stop();
    
    for (auto& thread : worker_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    
    {
        std::lock_guard<std::mutex> lock(pool_mutex_);
        for (auto& conn : connection_pool_) {
            conn->disconnect();
        }
        connection_pool_.clear();
        while (!available_connections_.empty()) {
            available_connections_.pop();
        }
    }
    
    LOG_INFO("Python agent client stopped");
}

/**
 * Send an LLMRequest using an available pooled PythonAgentConnection and return the agent's response.
 *
 * @param request The request payload to send to the Python agent.
 * @return LLMResponse The response received for the provided request.
 *
 * @throws std::runtime_error If no connections are available from the pool.
 * @throws Any exception propagated from the underlying connection's send_request.
 */
LLMResponse PythonAgentClient::send_request(const LLMRequest& request) {
    auto conn = get_connection();
    if (!conn) {
        throw std::runtime_error("No available connections");
    }
    
    try {
        auto response = conn->send_request(request, config_.request_timeout_ms);
        return_connection(conn);
        return response;
    } catch (...) {
        return_connection(conn);
        throw;
    }
}

/**
 * @brief Sends a streaming LLM request using an available connection from the pool.
 *
 * Obtains a connection from the pool and forwards the streaming request to it.
 * If no connection is available, invokes the error callback with a descriptive message.
 * Ensures the connection is returned to the pool after the final stream chunk is received
 * or if an error occurs.
 *
 * @param request The LLM request payload to send.
 * @param on_chunk Callback invoked for each received stream chunk; receives the chunk data.
 * @param on_error Callback invoked when an error occurs; receives an error message.
 */
void PythonAgentClient::send_streaming_request(
    const LLMRequest& request,
    StreamCallback on_chunk,
    ErrorCallback on_error
) {
    auto conn = get_connection();
    if (!conn) {
        on_error("No available connections");
        return;
    }
    
    auto wrapped_chunk = [on_chunk, conn, this](const StreamChunk& chunk) {
        on_chunk(chunk);
        if (chunk.is_final) {
            return_connection(conn);
        }
    };
    
    auto wrapped_error = [on_error, conn, this](const std::string& error) {
        on_error(error);
        return_connection(conn);
    };
    
    conn->send_streaming_request(request, wrapped_chunk, wrapped_error, config_.request_timeout_ms);
}

/**
 * @brief Determine whether at least one connection in the pool is healthy.
 *
 * This method acquires pool_mutex_ to safely inspect the connection pool and
 * queries each connection's health status.
 *
 * @return `true` if at least one pooled connection reports healthy, `false` otherwise.
 */
bool PythonAgentClient::health_check() {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    
    if (connection_pool_.empty()) return false;
    
    int healthy_count = 0;
    for (const auto& conn : connection_pool_) {
        if (conn->health_check()) {
            healthy_count++;
        }
    }
    
    return healthy_count > 0;
}

/**
 * @brief Acquire an available connection from the pool, waiting up to five seconds.
 *
 * Blocks until a connection becomes available or the 5-second wait expires.
 *
 * @return std::shared_ptr<PythonAgentConnection> Shared pointer to an available connection on success; `nullptr` if no connection became available within five seconds or the pool is empty.
 */
std::shared_ptr<PythonAgentConnection> PythonAgentClient::get_connection() {
    std::unique_lock<std::mutex> lock(pool_mutex_);
    
    bool available = pool_cv_.wait_for(
        lock,
        std::chrono::seconds(5),
        [this]() { return !available_connections_.empty(); }
    );
    
    if (!available || available_connections_.empty()) {
        return nullptr;
    }
    
    auto conn = available_connections_.front();
    available_connections_.pop();
    return conn;
}

/**
 * @brief Returns a connection to the client's available pool, making it available for reuse.
 *
 * Marks the provided connection as available and wakes one waiter that is blocked waiting for a connection.
 *
 * @param conn Shared pointer to the connection to return to the pool.
 */
void PythonAgentClient::return_connection(std::shared_ptr<PythonAgentConnection> conn) {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    available_connections_.push(conn);
    pool_cv_.notify_one();
}

/**
 * @brief Periodically performs health checks for the Python agent while the client is running.
 *
 * Sleeps for the configured health check interval and invokes health_check() in a loop
 * until the client's running flag is cleared.
 */
void PythonAgentClient::health_check_loop() {
    while (running_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(config_.health_check_interval_ms));
        
        if (health_check()) {
            LOG_DEBUG("Python agent health check passed");
        } else {
            LOG_WARNING("Python agent health check failed");
        }
    }
}

} // namespace ipc
} // namespace atomic