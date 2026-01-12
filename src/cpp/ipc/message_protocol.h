#pragma once

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <boost/json.hpp>

/**
     * Convert the Message into a JSON string suitable for IPC transmission.
     * @returns A JSON string containing the message fields: `id`, `type` (as integer), `timestamp`, and `payload`.
     */
    
    /**
     * Parse a JSON string into a Message.
     * @param json_str JSON string containing keys `id`, `type`, `timestamp`, and `payload`.
     * @returns A Message populated from the parsed JSON.
     */
    
    /**
     * Build a Boost.JSON object representing this LLMRequest.
     * @returns A `boost::json::object` with keys: `request_id`, `request_type` (as integer), `prompt`, `metadata` (object), `stream`, `max_tokens`, and `temperature`.
     */
    
    /**
     * Construct an LLMRequest from a Boost.JSON object.
     * @param obj JSON object expected to contain `request_id`, `request_type`, and `prompt`. Optional keys that will be applied if present: `metadata`, `stream`, `max_tokens`, `temperature`.
     * @returns An LLMRequest populated from `obj`.
     */
    namespace atomic {
namespace ipc {

enum class MessageType {
    REQUEST,
    RESPONSE,
    STREAM_CHUNK,
    ERROR,
    HEALTH_CHECK,
    HEALTH_RESPONSE
};

enum class RequestType {
    PARSE_IDEA,
    GENERATE_CODE,
    GENERATE_SPEC,
    GENERATE_DOCS,
    RESEARCH,
    CREATE_BRAND,
    BUSINESS_PLAN,
    LAUNCH_CONFIG
};

struct Message {
    std::string id;
    MessageType type;
    int64_t timestamp;
    boost::json::object payload;
    
    std::string serialize() const {
        boost::json::object obj;
        obj["id"] = id;
        obj["type"] = static_cast<int>(type);
        obj["timestamp"] = timestamp;
        obj["payload"] = payload;
        return boost::json::serialize(obj);
    }
    
    static Message deserialize(const std::string& json_str) {
        auto obj = boost::json::parse(json_str).as_object();
        Message msg;
        msg.id = obj["id"].as_string().c_str();
        msg.type = static_cast<MessageType>(obj["type"].as_int64());
        msg.timestamp = obj["timestamp"].as_int64();
        msg.payload = obj["payload"].as_object();
        return msg;
    }
};

struct LLMRequest {
    std::string request_id;
    RequestType request_type;
    std::string prompt;
    std::map<std::string, std::string> metadata;
    bool stream = true;
    int max_tokens = 4096;
    float temperature = 0.7f;
    
    boost::json::object to_json() const {
        boost::json::object obj;
        obj["request_id"] = request_id;
        obj["request_type"] = static_cast<int>(request_type);
        obj["prompt"] = prompt;
        
        boost::json::object meta;
        for (const auto& [key, value] : metadata) {
            meta[key] = value;
        }
        obj["metadata"] = meta;
        obj["stream"] = stream;
        obj["max_tokens"] = max_tokens;
        obj["temperature"] = temperature;
        
        return obj;
    }
    
    static LLMRequest from_json(const boost::json::object& obj) {
        LLMRequest req;
        req.request_id = obj.at("request_id").as_string().c_str();
        req.request_type = static_cast<RequestType>(obj.at("request_type").as_int64());
        req.prompt = obj.at("prompt").as_string().c_str();
        
        if (obj.contains("metadata")) {
            auto meta = obj.at("metadata").as_object();
            for (const auto& [key, value] : meta) {
                req.metadata[std::string(key)] = std::string(value.as_string());
            }
        }
        
        if (obj.contains("stream")) req.stream = obj.at("stream").as_bool();
        if (obj.contains("max_tokens")) req.max_tokens = obj.at("max_tokens").as_int64();
        if (obj.contains("temperature")) req.temperature = static_cast<float>(obj.at("temperature").as_double());
        
        return req;
    }
};

struct LLMResponse {
    std::string request_id;
    std::string content;
    std::string model;
    std::string provider;
    int input_tokens = 0;
    int output_tokens = 0;
    int64_t latency_ms = 0;
    bool is_final = true;
    std::string error;
    
    boost::json::object to_json() const {
        boost::json::object obj;
        obj["request_id"] = request_id;
        obj["content"] = content;
        obj["model"] = model;
        obj["provider"] = provider;
        obj["input_tokens"] = input_tokens;
        obj["output_tokens"] = output_tokens;
        obj["latency_ms"] = latency_ms;
        obj["is_final"] = is_final;
        if (!error.empty()) {
            obj["error"] = error;
        }
        return obj;
    }
    
    /**
     * Constructs an LLMResponse from a Boost.JSON object.
     *
     * @param obj JSON object expected to contain the keys "request_id" and "content".
     *            May optionally include "model", "provider", "input_tokens", "output_tokens",
     *            "latency_ms", "is_final", and "error"; those fields will be copied into
     *            the corresponding LLMResponse members if present.
     * @returns An LLMResponse populated from the provided JSON object. Required fields
     *          "request_id" and "content" are set; optional fields are set when present. 
     */
    static LLMResponse from_json(const boost::json::object& obj) {
        LLMResponse resp;
        resp.request_id = obj.at("request_id").as_string().c_str();
        resp.content = obj.at("content").as_string().c_str();
        
        if (obj.contains("model")) resp.model = obj.at("model").as_string().c_str();
        if (obj.contains("provider")) resp.provider = obj.at("provider").as_string().c_str();
        if (obj.contains("input_tokens")) resp.input_tokens = obj.at("input_tokens").as_int64();
        if (obj.contains("output_tokens")) resp.output_tokens = obj.at("output_tokens").as_int64();
        if (obj.contains("latency_ms")) resp.latency_ms = obj.at("latency_ms").as_int64();
        if (obj.contains("is_final")) resp.is_final = obj.at("is_final").as_bool();
        if (obj.contains("error")) resp.error = obj.at("error").as_string().c_str();
        
        return resp;
    }
};

struct StreamChunk {
    std::string request_id;
    std::string delta;
    std::string accumulated_content;
    int chunk_index = 0;
    bool is_final = false;
    std::string finish_reason;
    
    boost::json::object to_json() const {
        boost::json::object obj;
        obj["request_id"] = request_id;
        obj["delta"] = delta;
        obj["accumulated_content"] = accumulated_content;
        obj["chunk_index"] = chunk_index;
        obj["is_final"] = is_final;
        if (!finish_reason.empty()) {
            obj["finish_reason"] = finish_reason;
        }
        return obj;
    }
    
    /**
     * Parse a JSON object into a StreamChunk structure.
     *
     * Populates request_id, delta, accumulated_content, chunk_index, and is_final from the JSON object.
     * If the "finish_reason" key is present, sets finish_reason as well.
     *
     * @param obj JSON object representing a stream chunk.
     * @returns StreamChunk populated from obj.
     */
    static StreamChunk from_json(const boost::json::object& obj) {
        StreamChunk chunk;
        chunk.request_id = obj.at("request_id").as_string().c_str();
        chunk.delta = obj.at("delta").as_string().c_str();
        chunk.accumulated_content = obj.at("accumulated_content").as_string().c_str();
        chunk.chunk_index = obj.at("chunk_index").as_int64();
        chunk.is_final = obj.at("is_final").as_bool();
        if (obj.contains("finish_reason")) {
            chunk.finish_reason = obj.at("finish_reason").as_string().c_str();
        }
        return chunk;
    }
};

} // namespace ipc
} // namespace atomic