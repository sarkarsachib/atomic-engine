#include "http_server.h"
#include "../utils/helpers.h"
#include <boost/beast/websocket.hpp>

namespace atomic {
namespace server {

namespace websocket = beast::websocket;

HttpServer::HttpServer(const utils::ServerConfig& config)
    : config_(config) {
    LOG_INFO("HTTP server initialized on ", config_.host, ":", config_.http_port);
}

HttpServer::~HttpServer() {
    stop();
}

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

void HttpServer::add_cors_headers(http::response<http::string_body>& res) {
    res.set(http::field::access_control_allow_origin, "*");
    res.set(http::field::access_control_allow_methods, "GET, POST, PUT, DELETE, OPTIONS");
    res.set(http::field::access_control_allow_headers, "Content-Type, Authorization");
    res.set(http::field::access_control_max_age, "86400");
}

} // namespace server
} // namespace atomic
