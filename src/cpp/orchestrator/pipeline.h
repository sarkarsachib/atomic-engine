#pragma once

#include <string>
#include <vector>
#include <map>
#include <functional>
#include <chrono>
#include <memory>
#include "../ipc/message_protocol.h"
#include "../utils/helpers.h"

namespace atomic {
namespace orchestrator {

enum class PipelineStage {
    IDLE,
    PARSING,
    GENERATING,
    PACKAGING,
    EXPORTING,
    COMPLETED,
    FAILED
};

inline std::string stage_to_string(PipelineStage stage) {
    switch (stage) {
        case PipelineStage::IDLE: return "idle";
        case PipelineStage::PARSING: return "parsing";
        case PipelineStage::GENERATING: return "generating";
        case PipelineStage::PACKAGING: return "packaging";
        case PipelineStage::EXPORTING: return "exporting";
        case PipelineStage::COMPLETED: return "completed";
        case PipelineStage::FAILED: return "failed";
        default: return "unknown";
    }
}

struct PipelineResult {
    std::string stage;
    bool success = false;
    std::string content;
    std::map<std::string, std::string> artifacts;
    std::string error;
    int64_t duration_ms = 0;
};

struct PipelineContext {
    std::string pipeline_id;
    std::string request_id;
    PipelineStage current_stage = PipelineStage::IDLE;
    
    std::string original_prompt;
    std::map<std::string, std::string> metadata;
    
    std::map<PipelineStage, PipelineResult> stage_results;
    
    int64_t start_time = 0;
    int64_t end_time = 0;
    
    std::string error;
    bool completed = false;
    bool failed = false;
    
    int progress_percentage = 0;
    std::string status_message;
};

using StageHandler = std::function<PipelineResult(PipelineContext&)>;
using ProgressCallback = std::function<void(const PipelineContext&)>;

class Pipeline {
public:
    Pipeline();
    
    void add_stage(PipelineStage stage, StageHandler handler);
    void set_progress_callback(ProgressCallback callback);
    
    PipelineContext execute(const std::string& prompt, const std::map<std::string, std::string>& metadata);
    
    PipelineContext get_context(const std::string& pipeline_id) const;
    std::vector<PipelineContext> get_active_pipelines() const;
    
private:
    void transition_to_stage(PipelineContext& ctx, PipelineStage stage);
    void update_progress(PipelineContext& ctx);
    
    std::map<PipelineStage, StageHandler> stage_handlers_;
    ProgressCallback progress_callback_;
    
    mutable std::mutex mutex_;
    std::map<std::string, PipelineContext> active_pipelines_;
};

class PipelineStateMachine {
public:
    PipelineStateMachine();
    
    bool can_transition(PipelineStage from, PipelineStage to) const;
    PipelineStage next_stage(PipelineStage current) const;
    std::vector<PipelineStage> get_valid_transitions(PipelineStage from) const;
    
private:
    std::map<PipelineStage, std::vector<PipelineStage>> transitions_;
};

} // namespace orchestrator
} // namespace atomic
