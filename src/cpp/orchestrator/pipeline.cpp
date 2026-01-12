#include "pipeline.h"
#include "../utils/logger.h"

namespace atomic {
namespace orchestrator {

/**
 * @brief Default-constructs a Pipeline with default internal state.
 *
 * Creates a Pipeline instance ready for stage registration and execution.
 */
Pipeline::Pipeline() {}

/**
 * @brief Register or replace the handler associated with a specific pipeline stage.
 *
 * Stores the provided handler for the given stage; if a handler already exists for that
 * stage it will be overwritten. The registered handler will be used when the pipeline
 * processes that stage.
 *
 * @param stage The pipeline stage to associate the handler with.
 * @param handler Callable invoked to process the specified stage.
 */
void Pipeline::add_stage(PipelineStage stage, StageHandler handler) {
    std::lock_guard<std::mutex> lock(mutex_);
    stage_handlers_[stage] = handler;
}

/**
 * @brief Registers a callback to be invoked whenever a pipeline's progress is updated.
 *
 * @param callback Function that will be called with the current PipelineContext on progress updates.
 *                 Providing an empty/null callback clears any previously registered callback.
 */
void Pipeline::set_progress_callback(ProgressCallback callback) {
    progress_callback_ = callback;
}

/**
 * @brief Executes a pipeline for the given prompt through the defined stages and returns the final context.
 *
 * Executes the pipeline stages (PARSING, GENERATING, PACKAGING, EXPORTING) in order, invoking any registered
 * stage handlers, recording per-stage results and durations, and updating progress. The pipeline context is
 * registered and updated in the active pipeline registry; if a stage handler reports failure or an exception
 * is thrown, the context is marked failed and contains the error information.
 *
 * @param prompt Input prompt used to seed the pipeline.
 * @param metadata Optional metadata key/value pairs attached to the pipeline request.
 * @return PipelineContext Final context for the pipeline, containing timestamps, current stage, stage results,
 *         completion/failure flags, and any error message.
 */
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

/**
 * @brief Retrieve the stored PipelineContext for a given pipeline ID.
 *
 * @returns The PipelineContext associated with the provided pipeline_id, or a default-constructed PipelineContext if no matching context exists.
 */
PipelineContext Pipeline::get_context(const std::string& pipeline_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = active_pipelines_.find(pipeline_id);
    if (it != active_pipelines_.end()) {
        return it->second;
    }
    return PipelineContext{};
}

/**
 * Collects a thread-safe snapshot of pipeline contexts that are currently active.
 *
 * Active pipelines are those whose contexts are neither completed nor failed.
 *
 * @return std::vector<PipelineContext> Vector of active PipelineContext objects.
 */
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

/**
 * @brief Set the pipeline context's current stage and update its status message.
 *
 * Updates ctx.current_stage to the given stage and sets ctx.status_message to
 * indicate which stage is being processed.
 *
 * @param ctx Pipeline context to update.
 * @param stage Target stage to transition the context into.
 */
void Pipeline::transition_to_stage(PipelineContext& ctx, PipelineStage stage) {
    ctx.current_stage = stage;
    ctx.status_message = "Processing " + stage_to_string(stage);
    LOG_DEBUG("Pipeline ", ctx.pipeline_id, " -> ", stage_to_string(stage));
}

/**
 * @brief Updates the pipeline context's progress percentage, invokes the progress callback if set, and persists the context to the active pipelines map.
 *
 * Computes progress as (number of completed stage results) / 4 * 100, assigns the value to ctx.progress_percentage, calls the configured progress callback with the updated context when present, and stores the context in the internal active_pipelines_ map under mutex protection.
 *
 * @param ctx Pipeline context to update and persist.
 */
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

/**
 * @brief Initialize the state machine's allowed stage transitions.
 *
 * Populates the internal transitions map with the valid next stages for each
 * PipelineStage:
 * - IDLE -> PARSING
 * - PARSING -> GENERATING, FAILED
 * - GENERATING -> PACKAGING, FAILED
 * - PACKAGING -> EXPORTING, FAILED
 * - EXPORTING -> COMPLETED, FAILED
 * - COMPLETED -> (no transitions)
 * - FAILED -> (no transitions)
 */
PipelineStateMachine::PipelineStateMachine() {
    transitions_[PipelineStage::IDLE] = {PipelineStage::PARSING};
    transitions_[PipelineStage::PARSING] = {PipelineStage::GENERATING, PipelineStage::FAILED};
    transitions_[PipelineStage::GENERATING] = {PipelineStage::PACKAGING, PipelineStage::FAILED};
    transitions_[PipelineStage::PACKAGING] = {PipelineStage::EXPORTING, PipelineStage::FAILED};
    transitions_[PipelineStage::EXPORTING] = {PipelineStage::COMPLETED, PipelineStage::FAILED};
    transitions_[PipelineStage::COMPLETED] = {};
    transitions_[PipelineStage::FAILED] = {};
}

/**
 * @brief Checks whether a transition from one pipeline stage to another is allowed.
 *
 * @param from The current pipeline stage.
 * @param to The candidate next pipeline stage.
 * @return `true` if `to` is a valid next stage for `from`, `false` otherwise.
 */
bool PipelineStateMachine::can_transition(PipelineStage from, PipelineStage to) const {
    auto it = transitions_.find(from);
    if (it == transitions_.end()) return false;
    
    const auto& valid = it->second;
    return std::find(valid.begin(), valid.end(), to) != valid.end();
}

/**
 * @brief Selects the first valid next stage for a given pipeline stage.
 *
 * If no valid next stages are defined for the provided stage, the function
 * returns the provided `current` stage unchanged.
 *
 * @param current The current pipeline stage to query.
 * @return PipelineStage The first valid next stage, or `current` if none exist.
 */
PipelineStage PipelineStateMachine::next_stage(PipelineStage current) const {
    auto it = transitions_.find(current);
    if (it != transitions_.end() && !it->second.empty()) {
        return it->second[0];
    }
    return current;
}

/**
 * @brief Retrieves the set of valid next stages for a given pipeline stage.
 *
 * @param from The current pipeline stage to query transitions for.
 * @return std::vector<PipelineStage> Vector of stages that are valid to transition to from `from`. Returns an empty vector if no transitions are defined.
 */
std::vector<PipelineStage> PipelineStateMachine::get_valid_transitions(PipelineStage from) const {
    auto it = transitions_.find(from);
    if (it != transitions_.end()) {
        return it->second;
    }
    return {};
}

} // namespace orchestrator
} // namespace atomic