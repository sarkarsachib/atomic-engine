#include "orchestrator.h"
#include "../utils/logger.h"
#include "../utils/helpers.h"
#include <boost/json.hpp>

namespace atomic {
namespace orchestrator {

/**
 * @brief Constructs an Orchestrator and initializes its internal subsystems.
 *
 * Initializes internal components (HTTP server, router, LLM client, sandbox controller,
 * pipeline, and request queue) from the provided configuration and registers HTTP routes,
 * pipeline stage handlers, and the WebSocket handler.
 *
 * @param config Configuration used to initialize servers, IPC, sandbox, queue, and other subsystems.
 */
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

/**
 * @brief Stops the orchestrator and cleans up owned resources.
 *
 * Ensures processing is halted, network servers and clients are shut down,
 * and internal resources are released before the object is destroyed.
 */
Orchestrator::~Orchestrator() {
    stop();
}

/**
 * @brief Initializes and starts orchestrator subsystems and marks the orchestrator as running.
 *
 * Starts the LLM client and HTTP server, configures the router on the HTTP server, checks Docker
 * availability (logging a warning if unavailable), and creates and starts the request processor
 * that consumes queued GenerateRequest items. If the orchestrator is already running, the call is
 * a no-op.
 *
 * @return `true` if all components were started and the orchestrator was marked running,
 *         `false` if the orchestrator was already running or if any startup step failed.
 */
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

/**
 * @brief Stops the orchestrator and shuts down its managed components.
 *
 * If the orchestrator is running, sets internal running state to false and
 * stops the request processor, HTTP server, and LLM client. If the
 * orchestrator is not running, this call has no effect.
 */
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

/**
 * @brief Execute the pipeline for a generation request and produce a consolidated response.
 *
 * Runs the configured pipeline using the request's prompt and metadata, records pipeline metrics,
 * and returns a GenerateResponse containing identifiers, success status, content (from the GENERATING stage),
 * packaged artifacts (from the PACKAGING stage), any error message, and total processing time in milliseconds.
 *
 * @param request GenerateRequest containing the prompt and optional metadata for the pipeline.
 * @return GenerateResponse Struct with:
 *   - `request_id`: newly generated UUID for this request,
 *   - `pipeline_id`: pipeline execution identifier,
 *   - `success`: `true` if the pipeline completed without failure, `false` otherwise,
 *   - `content`: text produced by the GENERATING stage (if available),
 *   - `artifacts`: packaged artifacts from the PACKAGING stage (if available),
 *   - `error`: error message when a failure occurred,
 *   - `processing_time_ms`: total time spent processing this request in milliseconds.
 */
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

/**
 * @brief Streams code-generation output from the LLM and forwards framed JSON chunks and errors to the caller.
 *
 * Sends a streaming generate request to the LLM using the provided prompt and metadata, then invokes
 * the supplied handlers for each streamed chunk and for any errors that occur.
 *
 * @param request Generate request containing the prompt and optional metadata.
 * @param on_chunk Callback invoked for each streamed chunk. The callback receives a serialized JSON object with the keys:
 *                 - `type`: string, always `"chunk"`.
 *                 - `delta`: string, the newly received text fragment.
 *                 - `content`: string, the accumulated content so far.
 *                 - `is_final`: boolean, true for the last chunk.
 * @param on_error Callback invoked with an error message if streaming cannot be started or an exception occurs.
 */
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

/**
 * @brief Retrieve the pipeline context for a given pipeline ID.
 *
 * @param pipeline_id Identifier of the pipeline to retrieve.
 * @return PipelineContext The PipelineContext associated with the provided `pipeline_id`.
 */
PipelineContext Orchestrator::get_pipeline_status(const std::string& pipeline_id) {
    return pipeline_->get_context(pipeline_id);
}

/**
 * @brief Retrieve the current aggregated runtime metrics snapshot.
 *
 * @return std::map<std::string, double> A mapping from metric names to their numeric values representing the latest metrics collected by MetricsCollector.
 */
std::map<std::string, double> Orchestrator::get_metrics() {
    auto metrics = MetricsCollector::instance().get_metrics();
    return metrics.to_map();
}

/**
 * @brief Register HTTP endpoints used by the orchestrator.
 *
 * Configures the router with the API routes consumed by clients:
 * - POST /api/generate -> generate request handling
 * - GET  /api/status   -> service and pipeline status
 * - GET  /api/metrics  -> runtime metrics
 * - GET  /health       -> health checks for LLM and sandbox
 */
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

/**
 * @brief Registers the pipeline stages and their corresponding handlers.
 *
 * Associates each PipelineStage (PARSING, GENERATING, PACKAGING, EXPORTING)
 * with the orchestrator's stage handlers so the pipeline executes the correct
 * processing function for each stage.
 */
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

/**
 * @brief Registers a WebSocket message handler on the HTTP server.
 *
 * Configures the HTTP server to forward incoming WebSocket messages to
 * Orchestrator::handle_websocket_message, using the provided send_response
 * callback to deliver replies back to the client.
 */
void Orchestrator::setup_websocket_handler() {
    http_server_->set_websocket_handler([this](
        const std::string& message,
        std::function<void(const std::string&)> send_response
    ) {
        handle_websocket_message(message, send_response);
    });
}

/**
 * @brief Handle HTTP requests to generate code/content from a prompt.
 *
 * Expects the request body to be a JSON object with:
 * - "prompt" (string) — required.
 * - "metadata" (object) — optional key/value pairs added to the generation request.
 * - "stream" (bool) — optional; if true the request is rejected (streaming is supported via WebSocket).
 *
 * Processes the parsed input by constructing a GenerateRequest, invoking generate(...), and
 * returns a JSON response with the generation outcome.
 *
 * @param request The incoming HTTP request whose body must be the JSON described above.
 * @return server::HttpResponse A JSON response containing:
 * - "success" (bool)
 * - "request_id" (string)
 * - "pipeline_id" (string)
 * - "content" (string)
 * - "processing_time_ms" (number)
 * - optional "error" (string) when an error occurred
 * - optional "artifacts" (object) when artifacts were produced
 *
 * On malformed requests or internal exceptions the response status and error message are set
 * accordingly (400 for streaming-over-HTTP, 500 for internal errors).
 */
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

/**
 * @brief Build the orchestrator status response.
 *
 * Constructs an HTTP response whose JSON body contains:
 * - "running": whether the orchestrator is currently running
 * - "active_pipelines": count of active pipelines
 * - "queue_size": current size of the request queue
 *
 * @param request Incoming HTTP request (unused).
 * @return server::HttpResponse HTTP response with the status JSON; on exception the response is a 500 error with the exception message.
 */
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

/**
 * @brief Handle the /api/metrics endpoint and produce a JSON object of collected metrics.
 *
 * Processes the incoming HTTP request and returns a response whose body is a JSON object
 * mapping metric names to numeric values as reported by the orchestrator's metrics collector.
 *
 * @param request The incoming HTTP request for the metrics endpoint (not inspected by this handler).
 * @return server::HttpResponse A response containing a JSON object of metrics on success,
 *         or an HTTP 500 error response with the exception message on failure.
 */
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

/**
 * Perform health checks for core services and return a JSON HTTP response.
 *
 * Checks the LLM client and sandbox (Docker) availability and returns an HTTP
 * response whose JSON body contains the overall status and component states.
 *
 * @param request Incoming HTTP request (request body and parameters are ignored).
 * @return server::HttpResponse JSON body fields:
 *         - `status`: `"ok"`.
 *         - `llm_agent`: `"healthy"` if the LLM client reports healthy, `"unhealthy"` otherwise.
 *         - `docker`: `"available"` if Docker/sandbox is available, `"unavailable"` otherwise.
 *         - `uptime_ms`: uptime in milliseconds from the MetricsCollector.
 *         On error, the response is an HTTP 500 with the exception message as the error body.
 */
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

/**
 * @brief Handle an incoming WebSocket message that requests a streaming generation.
 *
 * Parses the incoming JSON message, constructs a GenerateRequest with `stream` set to true,
 * and invokes generate_streaming to forward chunked results back over the WebSocket.
 *
 * Expected input JSON fields:
 * - "prompt" (string): the prompt to generate from.
 * - "metadata" (object, optional): key/value pairs to attach to the request.
 *
 * The function sends back JSON messages via the provided callback:
 * - Chunk messages forwarded from the LLM as-is by generate_streaming.
 * - Error messages of the form `{ "type": "error", "error": "<message>" }` for parsing,
 *   generation, or streaming errors.
 *
 * @param message Raw JSON text received from the WebSocket.
 * @param send_response Callback used to send a JSON-formatted text message back to the client.
 */
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

/**
 * @brief Perform the pipeline's parsing stage by requesting an idea parse from the LLM.
 *
 * Sends a parse request using the pipeline's original prompt and returns a PipelineResult
 * describing the outcome.
 *
 * @param ctx Pipeline context whose `original_prompt` is used to construct the parse request and whose
 *            `pipeline_id` is used for logging.
 * @return PipelineResult Result for the parsing stage with `stage` set to "parsing",
 *         `success` set to `true` if the LLM returned no error, `content` populated with the
 *         LLM response content, and `error` populated on failure.
 */
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

/**
 * @brief Executes the generating stage by requesting code generation for the pipeline's original prompt.
 *
 * Sends a generation request using the pipeline's original prompt and returns the outcome of that stage.
 *
 * @param ctx Pipeline context containing the original prompt and pipeline identifiers; the result corresponds to this context.
 * @return PipelineResult Result for the "generating" stage with `success` indicating success, `content` containing generated output (if any), and `error` populated on failure.
 */
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

/**
 * @brief Packages generated artifacts for the current pipeline run.
 *
 * Uses the pipeline context to collect outputs from the generating stage and
 * produces packaged artifacts (including `generated_code` and a `timestamp`).
 *
 * @param ctx Pipeline context for the current run; used to read previous stage results.
 * @return PipelineResult Result for the packaging stage. On success `stage` is `"packaging"`,
 * `success` is `true`, `artifacts` contains `generated_code` and `timestamp`, and `content`
 * contains a success message. On failure `success` is `false` and `error` contains the reason.
 */
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

/**
 * Execute the exporting stage for the given pipeline context and produce its result.
 *
 * @param ctx PipelineContext representing the pipeline being processed (identifies pipeline and carries stage data).
 * @return PipelineResult Result for the exporting stage: `success` is `true` if export completed, `false` otherwise; `content` holds a human-readable status message; `error` contains an error message when `success` is `false`.
 */
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