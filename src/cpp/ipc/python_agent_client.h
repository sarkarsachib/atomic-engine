#pragma once

#include "message_protocol.h"
#include "../utils/logger.h"
#include "../utils/config.h"
#include <boost/asio.hpp>
#include <boost/asio/local/stream_protocol.hpp>
#include <memory>
#include <queue>
#include <functional>
#include <mutex>
#include <condition_variable>

/**
 * Construct a PythonAgentConnection bound to the given io_context and UNIX domain socket path.
 * @param io_context Boost.Asio io_context used for asynchronous I/O.
 * @param socket_path Filesystem path to the local stream (UNIX domain) socket.
 */
/**
 * Destroy the connection object and release resources.
 */

/**
 * Establish a connection to the Python agent over the configured socket.
 * @param timeout_ms Maximum time to wait for the connection, in milliseconds.
 * @returns `true` if the connection was established, `false` otherwise.
 */

/**
 * Check whether the connection to the Python agent is currently established.
 * @returns `true` if connected, `false` otherwise.
 */

/**
 * Close the active connection and release associated resources.
 */

/**
 * Send a single request and wait for its response.
 * @param request Request payload to send to the Python agent.
 * @param timeout_ms Maximum time to wait for a response, in milliseconds.
 * @returns The LLMResponse produced by the Python agent for the given request.
 */

/**
 * Send a streaming request to the Python agent and receive streamed chunks via callbacks.
 * @param request Request payload to send to the Python agent.
 * @param on_chunk Callback invoked for each streamed chunk received.
 * @param on_error Callback invoked with an error message if the streaming request fails.
 * @param timeout_ms Maximum time to wait for the overall streaming operation, in milliseconds.
 */

/**
 * Perform a lightweight health check of the connection to the Python agent.
 * @returns `true` if the agent responded to the health check, `false` otherwise.
 */

/**
 * Initiate or continue asynchronous read operations from the socket.
 */

/**
 * Process a raw incoming message payload and dispatch it to the appropriate pending request handler.
 */

/**
 * Construct a PythonAgentClient configured with the given IPC settings.
 * @param config IPC configuration used to create and manage the connection pool.
 */
/**
 * Destroy the client and release resources, ensuring background threads are stopped.
 */

/**
 * Start the client, initialize the connection pool and worker threads.
 * @returns `true` if the client started successfully, `false` otherwise.
 */

/**
 * Stop the client, signal shutdown, and join background threads.
 */

/**
 * Send a single request using a pooled connection and return its response.
 * @param request Request payload to send to the Python agent.
 * @returns The LLMResponse produced by the Python agent for the given request.
 */

/**
 * Send a streaming request using a pooled connection and receive streamed chunks via callbacks.
 * @param request Request payload to send to the Python agent.
 * @param on_chunk Callback invoked for each streamed chunk received.
 * @param on_error Callback invoked with an error message if the streaming request fails.
 */

/**
 * Perform an overall health check across the client's connection pool.
 * @returns `true` if the client and its connections are healthy, `false` otherwise.
 */

/**
 * Acquire a connection from the pool, blocking until one becomes available.
 * @returns A shared pointer to an available PythonAgentConnection.
 */

/**
 * Return a previously acquired connection back to the pool for reuse.
 * @param conn Connection to return to the pool.
 */

/**
 * Background loop that periodically performs health checks and manages pool health.
 */
namespace atomic {
namespace ipc {

using boost::asio::local::stream_protocol;
using StreamCallback = std::function<void(const StreamChunk&)>;
using ErrorCallback = std::function<void(const std::string&)>;

class PythonAgentConnection {
public:
    PythonAgentConnection(boost::asio::io_context& io_context, const std::string& socket_path);
    ~PythonAgentConnection();
    
    bool connect(int timeout_ms = 5000);
    bool is_connected() const { return connected_; }
    void disconnect();
    
    LLMResponse send_request(const LLMRequest& request, int timeout_ms = 60000);
    void send_streaming_request(
        const LLMRequest& request,
        StreamCallback on_chunk,
        ErrorCallback on_error,
        int timeout_ms = 60000
    );
    
    bool health_check();
    
private:
    void async_read();
    void handle_message(const std::string& data);
    
    boost::asio::io_context& io_context_;
    stream_protocol::socket socket_;
    std::string socket_path_;
    bool connected_ = false;
    
    std::mutex mutex_;
    std::condition_variable cv_;
    
    struct PendingRequest {
        std::string request_id;
        bool streaming = false;
        StreamCallback stream_callback;
        ErrorCallback error_callback;
        LLMResponse response;
        bool completed = false;
    };
    
    std::map<std::string, std::shared_ptr<PendingRequest>> pending_requests_;
    boost::asio::streambuf read_buffer_;
};

class PythonAgentClient {
public:
    explicit PythonAgentClient(const utils::IPCConfig& config);
    ~PythonAgentClient();
    
    bool start();
    void stop();
    
    LLMResponse send_request(const LLMRequest& request);
    void send_streaming_request(
        const LLMRequest& request,
        StreamCallback on_chunk,
        ErrorCallback on_error
    );
    
    bool health_check();
    
private:
    std::shared_ptr<PythonAgentConnection> get_connection();
    void return_connection(std::shared_ptr<PythonAgentConnection> conn);
    void health_check_loop();
    
    utils::IPCConfig config_;
    boost::asio::io_context io_context_;
    std::unique_ptr<boost::asio::io_context::work> work_;
    std::vector<std::thread> worker_threads_;
    
    std::vector<std::shared_ptr<PythonAgentConnection>> connection_pool_;
    std::queue<std::shared_ptr<PythonAgentConnection>> available_connections_;
    std::mutex pool_mutex_;
    std::condition_variable pool_cv_;
    
    std::atomic<bool> running_{false};
    std::thread health_check_thread_;
};

} // namespace ipc
} // namespace atomic