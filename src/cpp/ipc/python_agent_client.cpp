#include "python_agent_client.h"
#include "../utils/helpers.h"
#include <boost/asio/read_until.hpp>
#include <boost/asio/write.hpp>

namespace atomic {
namespace ipc {

// PythonAgentConnection implementation
PythonAgentConnection::PythonAgentConnection(
    boost::asio::io_context& io_context,
    const std::string& socket_path
) : io_context_(io_context), socket_(io_context), socket_path_(socket_path) {}

PythonAgentConnection::~PythonAgentConnection() {
    disconnect();
}

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

void PythonAgentConnection::disconnect() {
    if (connected_) {
        boost::system::error_code ec;
        socket_.close(ec);
        connected_ = false;
        LOG_DEBUG("Disconnected from Python agent");
    }
}

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

// PythonAgentClient implementation
PythonAgentClient::PythonAgentClient(const utils::IPCConfig& config)
    : config_(config), work_(std::make_unique<boost::asio::io_context::work>(io_context_)) {}

PythonAgentClient::~PythonAgentClient() {
    stop();
}

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

void PythonAgentClient::return_connection(std::shared_ptr<PythonAgentConnection> conn) {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    available_connections_.push(conn);
    pool_cv_.notify_one();
}

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
