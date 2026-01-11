#pragma once

#include <string>
#include <functional>
#include <map>
#include <boost/json.hpp>

namespace atomic {
namespace server {

struct HttpRequest {
    std::string method;
    std::string path;
    std::map<std::string, std::string> headers;
    std::string body;
    std::map<std::string, std::string> query_params;
};

struct HttpResponse {
    int status_code = 200;
    std::map<std::string, std::string> headers;
    std::string body;
    
    void set_json(const boost::json::object& obj) {
        headers["Content-Type"] = "application/json";
        body = boost::json::serialize(obj);
    }
    
    void set_error(int code, const std::string& message) {
        status_code = code;
        boost::json::object error;
        error["error"] = message;
        error["status"] = code;
        set_json(error);
    }
};

using RequestHandler = std::function<HttpResponse(const HttpRequest&)>;

class Router {
public:
    void add_route(const std::string& method, const std::string& path, RequestHandler handler) {
        routes_[method + ":" + path] = handler;
    }
    
    HttpResponse handle(const HttpRequest& request) {
        std::string key = request.method + ":" + request.path;
        
        auto it = routes_.find(key);
        if (it != routes_.end()) {
            return it->second(request);
        }
        
        HttpResponse response;
        response.set_error(404, "Not Found");
        return response;
    }
    
private:
    std::map<std::string, RequestHandler> routes_;
};

} // namespace server
} // namespace atomic
