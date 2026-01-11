#include "pipeline.h"
#include "../utils/logger.h"

namespace atomic {
namespace orchestrator {

Pipeline::Pipeline() {}

void Pipeline::add_stage(PipelineStage stage, StageHandler handler) {
    std::lock_guard<std::mutex> lock(mutex_);
    stage_handlers_[stage] = handler;
}

void Pipeline::set_progress_callback(ProgressCallback callback) {
    progress_callback_ = callback;
}

PipelineContext Pipeline::execute(
    const std::string& prompt,
    const std::map<std::string, std::string>& metadata
) {
    PipelineContext ctx;
    ctx.pipeline_id = utils::generate_uuid();
    ctx.request_id = utils::generate_uuid();
    ctx.original_prompt = prompt;
    ctx.metadata = metadata;
    ctx.start_time = utils::timestamp_ms();
    ctx.current_stage = PipelineStage::IDLE;
    
    {
        std::lock_guard<std::mutex> lock(mutex_);
        active_pipelines_[ctx.pipeline_id] = ctx;
    }
    
    LOG_INFO("Starting pipeline: ", ctx.pipeline_id);
    
    std::vector<PipelineStage> stages = {
        PipelineStage::PARSING,
        PipelineStage::GENERATING,
        PipelineStage::PACKAGING,
        PipelineStage::EXPORTING
    };
    
    try {
        for (auto stage : stages) {
            transition_to_stage(ctx, stage);
            update_progress(ctx);
            
            auto it = stage_handlers_.find(stage);
            if (it != stage_handlers_.end()) {
                auto start = utils::timestamp_ms();
                PipelineResult result = it->second(ctx);
                result.duration_ms = utils::timestamp_ms() - start;
                
                ctx.stage_results[stage] = result;
                
                if (!result.success) {
                    ctx.failed = true;
                    ctx.error = result.error;
                    ctx.current_stage = PipelineStage::FAILED;
                    LOG_ERROR("Pipeline stage failed: ", stage_to_string(stage), " - ", result.error);
                    break;
                }
                
                LOG_INFO("Pipeline stage completed: ", stage_to_string(stage), 
                        " (", result.duration_ms, "ms)");
            }
        }
        
        if (!ctx.failed) {
            ctx.current_stage = PipelineStage::COMPLETED;
            ctx.completed = true;
            LOG_INFO("Pipeline completed: ", ctx.pipeline_id);
        }
        
    } catch (const std::exception& e) {
        ctx.failed = true;
        ctx.error = e.what();
        ctx.current_stage = PipelineStage::FAILED;
        LOG_ERROR("Pipeline exception: ", e.what());
    }
    
    ctx.end_time = utils::timestamp_ms();
    update_progress(ctx);
    
    {
        std::lock_guard<std::mutex> lock(mutex_);
        active_pipelines_[ctx.pipeline_id] = ctx;
    }
    
    return ctx;
}

PipelineContext Pipeline::get_context(const std::string& pipeline_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = active_pipelines_.find(pipeline_id);
    if (it != active_pipelines_.end()) {
        return it->second;
    }
    return PipelineContext{};
}

std::vector<PipelineContext> Pipeline::get_active_pipelines() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<PipelineContext> result;
    for (const auto& [id, ctx] : active_pipelines_) {
        if (!ctx.completed && !ctx.failed) {
            result.push_back(ctx);
        }
    }
    return result;
}

void Pipeline::transition_to_stage(PipelineContext& ctx, PipelineStage stage) {
    ctx.current_stage = stage;
    ctx.status_message = "Processing " + stage_to_string(stage);
    LOG_DEBUG("Pipeline ", ctx.pipeline_id, " -> ", stage_to_string(stage));
}

void Pipeline::update_progress(PipelineContext& ctx) {
    int completed_stages = ctx.stage_results.size();
    int total_stages = 4;
    ctx.progress_percentage = (completed_stages * 100) / total_stages;
    
    if (progress_callback_) {
        progress_callback_(ctx);
    }
    
    {
        std::lock_guard<std::mutex> lock(mutex_);
        active_pipelines_[ctx.pipeline_id] = ctx;
    }
}

// PipelineStateMachine implementation
PipelineStateMachine::PipelineStateMachine() {
    transitions_[PipelineStage::IDLE] = {PipelineStage::PARSING};
    transitions_[PipelineStage::PARSING] = {PipelineStage::GENERATING, PipelineStage::FAILED};
    transitions_[PipelineStage::GENERATING] = {PipelineStage::PACKAGING, PipelineStage::FAILED};
    transitions_[PipelineStage::PACKAGING] = {PipelineStage::EXPORTING, PipelineStage::FAILED};
    transitions_[PipelineStage::EXPORTING] = {PipelineStage::COMPLETED, PipelineStage::FAILED};
    transitions_[PipelineStage::COMPLETED] = {};
    transitions_[PipelineStage::FAILED] = {};
}

bool PipelineStateMachine::can_transition(PipelineStage from, PipelineStage to) const {
    auto it = transitions_.find(from);
    if (it == transitions_.end()) return false;
    
    const auto& valid = it->second;
    return std::find(valid.begin(), valid.end(), to) != valid.end();
}

PipelineStage PipelineStateMachine::next_stage(PipelineStage current) const {
    auto it = transitions_.find(current);
    if (it != transitions_.end() && !it->second.empty()) {
        return it->second[0];
    }
    return current;
}

std::vector<PipelineStage> PipelineStateMachine::get_valid_transitions(PipelineStage from) const {
    auto it = transitions_.find(from);
    if (it != transitions_.end()) {
        return it->second;
    }
    return {};
}

} // namespace orchestrator
} // namespace atomic
