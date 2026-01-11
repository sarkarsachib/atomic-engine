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
