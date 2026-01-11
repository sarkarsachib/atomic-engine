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
