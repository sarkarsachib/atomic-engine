#pragma once

#include <string>
#include <functional>
#include <map>
#include <boost/json.hpp>

/**
 * Represents an HTTP request.
 */
 
/**
 * Represents an HTTP response with helpers to produce JSON payloads.
 */
 
/**
 * Set the response body to the serialized `obj` and ensure the
 * Content-Type header is set to "application/json".
 * @param obj JSON object to serialize into the response body.
 */
 
/**
 * Set the response to an error with the given HTTP status code and
 * a JSON body containing the fields `"error"` (message) and `"status"` (code).
 * @param code HTTP status code to set on the response.
 * @param message Human-readable error message to include in the JSON body.
 */
 
/**
 * Register a request handler for the given HTTP method and path.
 * The route is stored under a composite key formed from `method` and `path`.
 * @param method HTTP method (e.g., "GET", "POST") for the route.
 * @param path Request path (e.g., "/items") for the route.
 * @param handler Function to invoke when a request matching `method` and `path` is received.
 */
 
/**
 * Dispatch the provided request to a matching registered handler.
 * If no handler matches the request's method and path, returns a 404 Not Found
 * response encoded as a JSON error.
 * @param request The incoming HTTP request to route.
 * @returns HttpResponse produced by the matched handler, or a 404 JSON error response when no match exists.
 */
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