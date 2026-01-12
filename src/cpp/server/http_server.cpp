#include "http_server.h"
#include "../utils/helpers.h"
#include <boost/beast/websocket.hpp>

namespace atomic {
namespace server {

namespace websocket = beast::websocket;

/**
 * @brief Initialize the HTTP server instance with the provided configuration.
 *
 * @param config Server configuration (host, http_port, thread_count, CORS and other options)
 */
HttpServer::HttpServer(const utils::ServerConfig& config)
    : config_(config) {
    LOG_INFO("HTTP server initialized on ", config_.host, ":", config_.http_port);
}

/**
 * @brief Ensures the HTTP server is stopped and associated resources are released.
 *
 * Destructor for HttpServer that makes sure the server is no longer running and cleans up resources used for listening and connection handling.
 */
HttpServer::~HttpServer() {
    stop();
}

/**
 * @brief Initialize network listening and launch worker threads based on configuration.
 *
 * Sets up the TCP acceptor bound to the configured host and HTTP port, marks the server
 * as running, and spawns the configured number of worker threads that run the accept loop.
 *
 * @returns `true` if the server was successfully started and worker threads were launched, `false` otherwise (including when the server is already running or on startup errors).
 */
bool HttpServer::start() {
    if (running_) {
        LOG_WARNING("Server already running");
        return false;
    }
    
    try {
        auto address = net::ip::make_address(config_.host);
        tcp::endpoint endpoint(address, config_.http_port);
        
        acceptor_ = std::make_unique<tcp::acceptor>(io_context_, endpoint);
        acceptor_->set_option(net::socket_base::reuse_address(true));
        
        running_ = true;
        
        for (int i = 0; i < config_.thread_count; ++i) {
            worker_threads_.emplace_back([this]() {
                accept_loop();
            });
        }
        
        LOG_INFO("HTTP server started on ", config_.host, ":", config_.http_port);
        return true;
        
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to start server: ", e.what());
        return false;
    }
}

/**
 * @brief Stops the HTTP server and releases network and threading resources.
 *
 * If the server is not running, this is a no-op. Otherwise it closes the listening
 * acceptor, stops the I/O context, joins all worker threads, and clears the thread pool.
 */
void HttpServer::stop() {
    if (!running_) return;
    
    running_ = false;
    
    if (acceptor_) {
        boost::system::error_code ec;
        acceptor_->close(ec);
    }
    
    io_context_.stop();
    
    for (auto& thread : worker_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    
    worker_threads_.clear();
    LOG_INFO("HTTP server stopped");
}

/**
 * @brief Continuously accepts incoming TCP connections and dispatches each to a handler thread.
 *
 * This loop runs while the server is marked running, accepts sockets from the configured acceptor,
 * and spawns a detached thread to process each accepted connection via handle_connection().
 * The loop exits if the accept operation is aborted (e.g., acceptor closed) or when running_ is cleared.
 *
 * Errors from individual accept operations or exceptions thrown inside the loop are logged and do not
 * stop the overall accept loop except for operation-aborted conditions.
 */
void HttpServer::accept_loop() {
    while (running_) {
        try {
            tcp::socket socket(io_context_);
            
            boost::system::error_code ec;
            acceptor_->accept(socket, ec);
            
            if (ec) {
                if (ec == net::error::operation_aborted) {
                    break;
                }
                LOG_ERROR("Accept error: ", ec.message());
                continue;
            }
            
            std::thread([this, socket = std::move(socket)]() mutable {
                handle_connection(std::move(socket));
            }).detach();
            
        } catch (const std::exception& e) {
            LOG_ERROR("Accept loop error: ", e.what());
        }
    }
}

/**
 * @brief Handle a single accepted TCP connection by dispatching it to HTTP or WebSocket processing.
 *
 * Reads an incoming HTTP request from the provided socket; if the request is a WebSocket upgrade,
 * the connection is handed to the WebSocket handler, otherwise an HTTP response is produced,
 * optional CORS headers are applied when enabled, and the response is written back to the socket.
 *
 * @param socket TCP socket for the accepted connection. Ownership of the socket is taken by value
 *               and may be moved into internal handlers.
 */
void HttpServer::handle_connection(tcp::socket socket) {
    try {
        beast::flat_buffer buffer;
        http::request<http::string_body> req;
        
        http::read(socket, buffer, req);
        
        if (websocket::is_upgrade(req)) {
            handle_websocket(std::move(socket), std::move(req));
        } else {
            http::response<http::string_body> res;
            handle_http_request(req, res);
            
            if (config_.enable_cors) {
                add_cors_headers(res);
            }
            
            http::write(socket, res);
        }
        
    } catch (const std::exception& e) {
        LOG_ERROR("Connection handling error: ", e.what());
    }
}

/**
 * @brief Process an incoming HTTP request using the configured router and populate the HTTP response.
 *
 * If a router is configured, converts the incoming request into the server's HttpRequest representation,
 * invokes the router to obtain an HttpResponse, and maps the status, headers, and body back to the outgoing
 * Beast HTTP response. If no router is configured or an exception occurs while handling the request,
 * sets the response to HTTP 500 with a JSON error body.
 *
 * @param req The incoming HTTP request to handle.
 * @param res The HTTP response to populate with status, headers, and body.
 */
void HttpServer::handle_http_request(
    http::request<http::string_body>& req,
    http::response<http::string_body>& res
) {
    try {
        if (!router_) {
            res.result(http::status::internal_server_error);
            res.set(http::field::content_type, "application/json");
            res.body() = R"({"error": "No router configured"})";
            res.prepare_payload();
            return;
        }
        
        HttpRequest request;
        request.method = std::string(req.method_string());
        request.path = std::string(req.target());
        request.body = req.body();
        
        for (const auto& field : req) {
            request.headers[std::string(field.name_string())] = std::string(field.value());
        }
        
        HttpResponse response = router_->handle(request);
        
        res.result(static_cast<http::status>(response.status_code));
        
        for (const auto& [key, value] : response.headers) {
            res.set(key, value);
        }
        
        res.body() = response.body;
        res.prepare_payload();
        
    } catch (const std::exception& e) {
        LOG_ERROR("Request handling error: ", e.what());
        res.result(http::status::internal_server_error);
        res.set(http::field::content_type, "application/json");
        res.body() = R"({"error": "Internal server error"})";
        res.prepare_payload();
    }
}

/**
 * @brief Handle an accepted HTTP upgrade request and run a WebSocket session.
 *
 * Accepts the provided upgrade request to establish a WebSocket over the given socket,
 * then continuously reads incoming messages and dispatches them to the configured
 * WebSocket handler callback. Provides the handler a send callback that transmits
 * a string message back to the client.
 *
 * Errors encountered while sending or while processing the WebSocket session are
 * caught and do not propagate out of this function; the session ends on socket
 * closure or other WebSocket errors.
 *
 * @param socket Connected TCP socket for the WebSocket session (moved into the handler).
 * @param req The HTTP upgrade request used to accept the WebSocket connection.
 */
void HttpServer::handle_websocket(tcp::socket socket, http::request<http::string_body> req) {
    try {
        websocket::stream<tcp::socket> ws(std::move(socket));
        ws.accept(req);
        
        LOG_INFO("WebSocket connection established");
        
        auto send_message = [&ws](const std::string& message) {
            try {
                ws.write(net::buffer(message));
            } catch (const std::exception& e) {
                LOG_ERROR("WebSocket send error: ", e.what());
            }
        };
        
        while (running_) {
            beast::flat_buffer buffer;
            
            try {
                ws.read(buffer);
                
                std::string message = beast::buffers_to_string(buffer.data());
                
                if (ws_handler_) {
                    ws_handler_(message, send_message);
                }
                
            } catch (beast::system_error const& se) {
                if (se.code() != websocket::error::closed) {
                    LOG_ERROR("WebSocket error: ", se.what());
                }
                break;
            }
        }
        
        LOG_INFO("WebSocket connection closed");
        
    } catch (const std::exception& e) {
        LOG_ERROR("WebSocket handling error: ", e.what());
    }
}

/**
 * @brief Adds standard Cross-Origin Resource Sharing (CORS) headers to an HTTP response.
 *
 * Sets the following headers on the provided response:
 * - `Access-Control-Allow-Origin: *`
 * - `Access-Control-Allow-Methods: GET, POST, PUT, DELETE, OPTIONS`
 * - `Access-Control-Allow-Headers: Content-Type, Authorization`
 * - `Access-Control-Max-Age: 86400`
 *
 * @param res HTTP response to which the CORS headers will be added.
 */
void HttpServer::add_cors_headers(http::response<http::string_body>& res) {
    res.set(http::field::access_control_allow_origin, "*");
    res.set(http::field::access_control_allow_methods, "GET, POST, PUT, DELETE, OPTIONS");
    res.set(http::field::access_control_allow_headers, "Content-Type, Authorization");
    res.set(http::field::access_control_max_age, "86400");
}

} // namespace server
} // namespace atomic