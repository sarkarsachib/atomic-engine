#pragma once

#include "request_handler.h"
#include "../utils/config.h"
#include "../utils/logger.h"
#include <boost/asio.hpp>
#include <boost/beast.hpp>
#include <memory>
#include <thread>
#include <vector>
#include <functional>

namespace beast = boost::beast;
namespace http = beast::http;
namespace net = boost::asio;
using tcp = net::ip::tcp;

namespace atomic {
namespace server {

using WebSocketHandler = std::function<void(const std::string&, std::function<void(const std::string&)>)>;

class HttpServer {
public:
    explicit HttpServer(const utils::ServerConfig& config);
    ~HttpServer();
    
    /**
     * Set the router used to dispatch incoming HTTP requests.
     *
     * @param router Shared pointer to a Router that will handle request routing; passing `nullptr` clears the current router.
     */
    void set_router(std::shared_ptr<Router> router) {
        router_ = router;
    }
    
    /**
     * Sets the function invoked to handle WebSocket messages for accepted connections.
     *
     * @param handler Function that will be called for each incoming WebSocket message. The handler receives
     *                the received message as a `const std::string&` and a callback `std::function<void(std::string)>`
     *                to send a response or outbound message.
     */
    void set_websocket_handler(WebSocketHandler handler) {
        ws_handler_ = handler;
    }
    
    bool start();
    void stop();
    
    bool is_running() const { return running_; }
    
private:
    void accept_loop();
    void handle_connection(tcp::socket socket);
    void handle_http_request(
        http::request<http::string_body>& req,
        http::response<http::string_body>& res
    );
    void handle_websocket(tcp::socket socket, http::request<http::string_body> req);
    
    void add_cors_headers(http::response<http::string_body>& res);
    
    utils::ServerConfig config_;
    std::shared_ptr<Router> router_;
    WebSocketHandler ws_handler_;
    
    net::io_context io_context_;
    std::unique_ptr<tcp::acceptor> acceptor_;
    std::vector<std::thread> worker_threads_;
    std::atomic<bool> running_{false};
};

} // namespace server
} // namespace atomic