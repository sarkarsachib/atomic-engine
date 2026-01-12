#include "orchestrator.h"
#include "../utils/logger.h"
#include "../utils/helpers.h"
#include <boost/json.hpp>

namespace atomic {
namespace orchestrator {

Orchestrator::Orchestrator(const utils::Config& config)
    : config_(config) {
    
    LOG_INFO("Initializing Atomic Orchestrator");
    
    http_server_ = std::make_unique<server::HttpServer>(config_.server);
    router_ = std::make_shared<server::Router>();
    llm_client_ = std::make_unique<ipc::PythonAgentClient>(config_.ipc);
    sandbox_ = std::make_unique<sandbox::DockerController>(config_.sandbox);
    pipeline_ = std::make_unique<Pipeline>();
    request_queue_ = std::make_unique<queue::RequestQueue<GenerateRequest>>(
        config_.queue.max_queue_size,
        config_.queue.max_concurrent_requests
    );
    
    setup_routes();
    setup_pipeline_handlers();
    setup_websocket_handler();
}

Orchestrator::~Orchestrator() {
    stop();
}

bool Orchestrator::start() {
    if (running_) {
        LOG_WARNING("Orchestrator already running");
        return false;
    }
    
    LOG_INFO("Starting Atomic Orchestrator");
    
    if (!llm_client_->start()) {
        LOG_ERROR("Failed to start LLM client");
        return false;
    }
    
    if (!sandbox_->is_docker_available()) {
        LOG_WARNING("Docker not available - sandbox features will be limited");
    }
    
    http_server_->set_router(router_);
    if (!http_server_->start()) {
        LOG_ERROR("Failed to start HTTP server");
        return false;
    }
    
    request_processor_ = std::make_unique<queue::RequestProcessor<GenerateRequest>>(
        *request_queue_,
        [this](const queue::QueuedRequest<GenerateRequest>& req) {
            auto start_time = utils::timestamp_ms();
            try {
                auto response = generate(req.data);
                auto latency = utils::timestamp_ms() - start_time;
                MetricsCollector::instance().record_request(latency, response.success);
            } catch (const std::exception& e) {
                auto latency = utils::timestamp_ms() - start_time;
                MetricsCollector::instance().record_request(latency, false);
                LOG_ERROR("Request processing error: ", e.what());
            }
        },
        config_.queue.max_concurrent_requests
    );
    
    request_processor_->start();
    
    running_ = true;
    LOG_INFO("Atomic Orchestrator started successfully");
    
    return true;
}

void Orchestrator::stop() {
    if (!running_) return;
    
    LOG_INFO("Stopping Atomic Orchestrator");
    
    running_ = false;
    
    if (request_processor_) {
        request_processor_->stop();
    }
    
    if (http_server_) {
        http_server_->stop();
    }
    
    if (llm_client_) {
        llm_client_->stop();
    }
    
    LOG_INFO("Atomic Orchestrator stopped");
}

GenerateResponse Orchestrator::generate(const GenerateRequest& request) {
    GenerateResponse response;
    response.request_id = utils::generate_uuid();
    
    auto start_time = utils::timestamp_ms();
    MetricsCollector::instance().record_pipeline_start();
    
    try {
        PipelineContext ctx = pipeline_->execute(request.prompt, request.metadata);
        
        response.pipeline_id = ctx.pipeline_id;
        response.success = ctx.completed && !ctx.failed;
        response.error = ctx.error;
        
        if (ctx.stage_results.count(PipelineStage::GENERATING)) {
            response.content = ctx.stage_results[PipelineStage::GENERATING].content;
        }
        
        if (ctx.stage_results.count(PipelineStage::PACKAGING)) {
            response.artifacts = ctx.stage_results[PipelineStage::PACKAGING].artifacts;
        }
        
        response.processing_time_ms = utils::timestamp_ms() - start_time;
        
        MetricsCollector::instance().record_pipeline_complete(
            response.processing_time_ms,
            response.success
        );
        
    } catch (const std::exception& e) {
        response.success = false;
        response.error = e.what();
        response.processing_time_ms = utils::timestamp_ms() - start_time;
        
        MetricsCollector::instance().record_pipeline_complete(
            response.processing_time_ms,
            false
        );
        
        LOG_ERROR("Generate error: ", e.what());
    }
    
    return response;
}

void Orchestrator::generate_streaming(
    const GenerateRequest& request,
    std::function<void(const std::string&)> on_chunk,
    std::function<void(const std::string&)> on_error
) {
    try {
        ipc::LLMRequest llm_request;
        llm_request.request_id = utils::generate_uuid();
        llm_request.request_type = ipc::RequestType::GENERATE_CODE;
        llm_request.prompt = request.prompt;
        llm_request.stream = true;
        
        for (const auto& [key, value] : request.metadata) {
            llm_request.metadata[key] = value;
        }
        
        llm_client_->send_streaming_request(
            llm_request,
            [on_chunk](const ipc::StreamChunk& chunk) {
                boost::json::object obj;
                obj["type"] = "chunk";
                obj["delta"] = chunk.delta;
                obj["content"] = chunk.accumulated_content;
                obj["is_final"] = chunk.is_final;
                on_chunk(boost::json::serialize(obj));
            },
            on_error
        );
        
    } catch (const std::exception& e) {
        on_error(e.what());
    }
}

PipelineContext Orchestrator::get_pipeline_status(const std::string& pipeline_id) {
    return pipeline_->get_context(pipeline_id);
}

std::map<std::string, double> Orchestrator::get_metrics() {
    auto metrics = MetricsCollector::instance().get_metrics();
    return metrics.to_map();
}

void Orchestrator::setup_routes() {
    router_->add_route("POST", "/api/generate", [this](const server::HttpRequest& req) {
        return handle_generate(req);
    });
    
    router_->add_route("GET", "/api/status", [this](const server::HttpRequest& req) {
        return handle_status(req);
    });
    
    router_->add_route("GET", "/api/metrics", [this](const server::HttpRequest& req) {
        return handle_metrics(req);
    });
    
    router_->add_route("GET", "/health", [this](const server::HttpRequest& req) {
        return handle_health(req);
    });
}

void Orchestrator::setup_pipeline_handlers() {
    pipeline_->add_stage(PipelineStage::PARSING, [this](PipelineContext& ctx) {
        return parse_stage(ctx);
    });
    
    pipeline_->add_stage(PipelineStage::GENERATING, [this](PipelineContext& ctx) {
        return generate_stage(ctx);
    });
    
    pipeline_->add_stage(PipelineStage::PACKAGING, [this](PipelineContext& ctx) {
        return package_stage(ctx);
    });
    
    pipeline_->add_stage(PipelineStage::EXPORTING, [this](PipelineContext& ctx) {
        return export_stage(ctx);
    });
}

void Orchestrator::setup_websocket_handler() {
    http_server_->set_websocket_handler([this](
        const std::string& message,
        std::function<void(const std::string&)> send_response
    ) {
        handle_websocket_message(message, send_response);
    });
}

server::HttpResponse Orchestrator::handle_generate(const server::HttpRequest& request) {
    server::HttpResponse response;
    
    try {
        auto body = boost::json::parse(request.body).as_object();
        
        GenerateRequest gen_request;
        gen_request.prompt = body.at("prompt").as_string().c_str();
        
        if (body.contains("metadata")) {
            auto meta = body.at("metadata").as_object();
            for (const auto& [key, value] : meta) {
                gen_request.metadata[std::string(key)] = std::string(value.as_string());
            }
        }
        
        if (body.contains("stream") && body.at("stream").as_bool()) {
            response.set_error(400, "Use WebSocket for streaming requests");
            return response;
        }
        
        auto result = generate(gen_request);
        
        boost::json::object obj;
        obj["success"] = result.success;
        obj["request_id"] = result.request_id;
        obj["pipeline_id"] = result.pipeline_id;
        obj["content"] = result.content;
        obj["processing_time_ms"] = result.processing_time_ms;
        
        if (!result.error.empty()) {
            obj["error"] = result.error;
        }
        
        if (!result.artifacts.empty()) {
            boost::json::object artifacts;
            for (const auto& [key, value] : result.artifacts) {
                artifacts[key] = value;
            }
            obj["artifacts"] = artifacts;
        }
        
        response.set_json(obj);
        
    } catch (const std::exception& e) {
        response.set_error(500, e.what());
    }
    
    return response;
}

server::HttpResponse Orchestrator::handle_status(const server::HttpRequest& request) {
    server::HttpResponse response;
    
    try {
        auto active = pipeline_->get_active_pipelines();
        
        boost::json::object obj;
        obj["running"] = running_.load();
        obj["active_pipelines"] = static_cast<int64_t>(active.size());
        obj["queue_size"] = static_cast<int64_t>(request_queue_->size());
        
        response.set_json(obj);
        
    } catch (const std::exception& e) {
        response.set_error(500, e.what());
    }
    
    return response;
}

server::HttpResponse Orchestrator::handle_metrics(const server::HttpRequest& request) {
    server::HttpResponse response;
    
    try {
        auto metrics = get_metrics();
        
        boost::json::object obj;
        for (const auto& [key, value] : metrics) {
            obj[key] = value;
        }
        
        response.set_json(obj);
        
    } catch (const std::exception& e) {
        response.set_error(500, e.what());
    }
    
    return response;
}

server::HttpResponse Orchestrator::handle_health(const server::HttpRequest& request) {
    server::HttpResponse response;
    
    try {
        bool llm_healthy = llm_client_->health_check();
        bool docker_healthy = sandbox_->is_docker_available();
        
        boost::json::object obj;
        obj["status"] = "ok";
        obj["llm_agent"] = llm_healthy ? "healthy" : "unhealthy";
        obj["docker"] = docker_healthy ? "available" : "unavailable";
        obj["uptime_ms"] = MetricsCollector::instance().get_metrics().uptime_ms();
        
        response.set_json(obj);
        
    } catch (const std::exception& e) {
        response.set_error(500, e.what());
    }
    
    return response;
}

void Orchestrator::handle_websocket_message(
    const std::string& message,
    std::function<void(const std::string&)> send_response
) {
    try {
        auto obj = boost::json::parse(message).as_object();
        
        GenerateRequest request;
        request.prompt = obj.at("prompt").as_string().c_str();
        request.stream = true;
        
        if (obj.contains("metadata")) {
            auto meta = obj.at("metadata").as_object();
            for (const auto& [key, value] : meta) {
                request.metadata[std::string(key)] = std::string(value.as_string());
            }
        }
        
        generate_streaming(
            request,
            send_response,
            [send_response](const std::string& error) {
                boost::json::object err;
                err["type"] = "error";
                err["error"] = error;
                send_response(boost::json::serialize(err));
            }
        );
        
    } catch (const std::exception& e) {
        boost::json::object err;
        err["type"] = "error";
        err["error"] = e.what();
        send_response(boost::json::serialize(err));
    }
}

PipelineResult Orchestrator::parse_stage(PipelineContext& ctx) {
    PipelineResult result;
    result.stage = "parsing";
    
    try {
        ipc::LLMRequest request;
        request.request_id = utils::generate_uuid();
        request.request_type = ipc::RequestType::PARSE_IDEA;
        request.prompt = "Parse this idea: " + ctx.original_prompt;
        request.stream = false;
        
        auto response = llm_client_->send_request(request);
        
        result.success = response.error.empty();
        result.content = response.content;
        result.error = response.error;
        
        LOG_INFO("Parse stage completed for pipeline: ", ctx.pipeline_id);
        
    } catch (const std::exception& e) {
        result.success = false;
        result.error = e.what();
        LOG_ERROR("Parse stage failed: ", e.what());
    }
    
    return result;
}

PipelineResult Orchestrator::generate_stage(PipelineContext& ctx) {
    PipelineResult result;
    result.stage = "generating";
    
    try {
        ipc::LLMRequest request;
        request.request_id = utils::generate_uuid();
        request.request_type = ipc::RequestType::GENERATE_CODE;
        request.prompt = ctx.original_prompt;
        request.stream = false;
        
        auto response = llm_client_->send_request(request);
        
        result.success = response.error.empty();
        result.content = response.content;
        result.error = response.error;
        
        LOG_INFO("Generate stage completed for pipeline: ", ctx.pipeline_id);
        
    } catch (const std::exception& e) {
        result.success = false;
        result.error = e.what();
        LOG_ERROR("Generate stage failed: ", e.what());
    }
    
    return result;
}

PipelineResult Orchestrator::package_stage(PipelineContext& ctx) {
    PipelineResult result;
    result.stage = "packaging";
    
    try {
        result.artifacts["generated_code"] = ctx.stage_results[PipelineStage::GENERATING].content;
        result.artifacts["timestamp"] = utils::timestamp_iso();
        
        result.success = true;
        result.content = "Artifacts packaged successfully";
        
        LOG_INFO("Package stage completed for pipeline: ", ctx.pipeline_id);
        
    } catch (const std::exception& e) {
        result.success = false;
        result.error = e.what();
        LOG_ERROR("Package stage failed: ", e.what());
    }
    
    return result;
}

PipelineResult Orchestrator::export_stage(PipelineContext& ctx) {
    PipelineResult result;
    result.stage = "exporting";
    
    try {
        result.success = true;
        result.content = "Export completed";
        
        LOG_INFO("Export stage completed for pipeline: ", ctx.pipeline_id);
        
    } catch (const std::exception& e) {
        result.success = false;
        result.error = e.what();
        LOG_ERROR("Export stage failed: ", e.what());
    }
    
    return result;
}

} // namespace orchestrator
} // namespace atomic
