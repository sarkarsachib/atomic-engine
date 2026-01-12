#pragma once

#include "pipeline.h"
#include "metrics.h"
#include "../server/http_server.h"
#include "../server/request_handler.h"
#include "../ipc/python_agent_client.h"
#include "../sandbox/docker_controller.h"
#include "../queue/request_queue.h"
#include "../utils/config.h"
#include <memory>
#include <atomic>

/**
 * Encapsulates input parameters for a generation task.
 *
 * prompt: User-facing prompt to drive the generation.
 * metadata: Arbitrary key/value metadata associated with the request.
 * stream: When true, enables streaming responses (default false).
 * priority: Priority of the request where higher values indicate higher priority (default 1).
 */

/**
 * Encapsulates the result of a generation task.
 *
 * request_id: Unique identifier for the request.
 * pipeline_id: Identifier of the pipeline that processed the request.
 * success: `true` if the generation completed successfully, `false` otherwise.
 * content: Generated textual output.
 * artifacts: Map of artifact names to their locations or identifiers.
 * error: Error message when `success` is `false`.
 * processing_time_ms: Total processing duration in milliseconds.
 */

/**
 * Orchestrator manages lifecycle, routing, and execution of generation pipelines.
 *
 * It provides synchronous and streaming generation APIs, endpoints for status/metrics,
 * and access to pipeline status and collected metrics.
 */

/**
 * Construct an Orchestrator with the provided configuration.
 *
 * @param config Configuration values used to initialize the orchestrator.
 */

/**
 * Clean up orchestrator resources.
 */

/**
 * Start the orchestrator and any internal services (HTTP server, workers, etc.).
 *
 * @returns `true` if the orchestrator started successfully, `false` otherwise.
 */

/**
 * Stop the orchestrator and gracefully shut down internal services.
 */

/**
 * Check whether the orchestrator is currently running.
 *
 * @returns `true` if running, `false` otherwise.
 */

/**
 * Execute a generation task synchronously and return its result.
 *
 * @param request GenerateRequest containing prompt, metadata, and execution options.
 * @returns GenerateResponse with identifiers, output, artifacts, success flag, and timing.
 */

/**
 * Execute a generation task in streaming mode, invoking callbacks for chunks and errors.
 *
 * @param request GenerateRequest containing prompt, metadata, and execution options.
 * @param on_chunk Callback invoked for each streamed chunk of generated content.
 * @param on_error Callback invoked with an error message if streaming or generation fails.
 */

/**
 * Retrieve the current status of a named pipeline.
 *
 * @param pipeline_id Identifier of the pipeline to query.
 * @returns PipelineContext describing the pipeline's current status and metadata.
 */

/**
 * Retrieve collected runtime metrics from the orchestrator.
 *
 * @returns Map from metric name to its numeric value.
 */
namespace atomic {
namespace orchestrator {

struct GenerateRequest {
    std::string prompt;
    std::map<std::string, std::string> metadata;
    bool stream = false;
    int priority = 1;
};

struct GenerateResponse {
    std::string request_id;
    std::string pipeline_id;
    bool success = false;
    std::string content;
    std::map<std::string, std::string> artifacts;
    std::string error;
    int64_t processing_time_ms = 0;
};

class Orchestrator {
public:
    explicit Orchestrator(const utils::Config& config);
    ~Orchestrator();
    
    bool start();
    void stop();
    
    bool is_running() const { return running_; }
    
    GenerateResponse generate(const GenerateRequest& request);
    
    void generate_streaming(
        const GenerateRequest& request,
        std::function<void(const std::string&)> on_chunk,
        std::function<void(const std::string&)> on_error
    );
    
    PipelineContext get_pipeline_status(const std::string& pipeline_id);
    
    std::map<std::string, double> get_metrics();
    
private:
    void setup_routes();
    void setup_pipeline_handlers();
    void setup_websocket_handler();
    
    server::HttpResponse handle_generate(const server::HttpRequest& request);
    server::HttpResponse handle_status(const server::HttpRequest& request);
    server::HttpResponse handle_metrics(const server::HttpRequest& request);
    server::HttpResponse handle_health(const server::HttpRequest& request);
    
    void handle_websocket_message(
        const std::string& message,
        std::function<void(const std::string&)> send_response
    );
    
    PipelineResult parse_stage(PipelineContext& ctx);
    PipelineResult generate_stage(PipelineContext& ctx);
    PipelineResult package_stage(PipelineContext& ctx);
    PipelineResult export_stage(PipelineContext& ctx);
    
    void process_queue();
    
    utils::Config config_;
    std::atomic<bool> running_{false};
    
    std::unique_ptr<server::HttpServer> http_server_;
    std::shared_ptr<server::Router> router_;
    std::unique_ptr<ipc::PythonAgentClient> llm_client_;
    std::unique_ptr<sandbox::DockerController> sandbox_;
    std::unique_ptr<Pipeline> pipeline_;
    std::unique_ptr<queue::RequestQueue<GenerateRequest>> request_queue_;
    std::unique_ptr<queue::RequestProcessor<GenerateRequest>> request_processor_;
};

} // namespace orchestrator
} // namespace atomic