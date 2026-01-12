#pragma once

#include <string>
#include <vector>
#include <map>
#include <functional>
#include <chrono>
#include <memory>
#include "../ipc/message_protocol.h"
#include "../utils/helpers.h"

/**
 * Convert a PipelineStage enum value to its lowercase string representation.
 *
 * @param stage The pipeline stage to convert.
 * @returns The lowercase name of the stage (e.g., "parsing"), or "unknown" for unrecognized values.
 */

/**
 * Result produced by a single pipeline stage.
 *
 * Contains the stage name, success flag, textual content produced by the stage,
 * named artifacts produced by the stage, an error message if any, and the stage duration in milliseconds.
 */

/**
 * Execution context for a pipeline run.
 *
 * Holds identifiers, the current stage, the original prompt and metadata,
 * per-stage results, start/end timestamps, terminal error/state flags, progress percentage, and a status message.
 */

/**
 * Register a handler for a specific pipeline stage.
 *
 * @param stage The stage to register the handler for.
 * @param handler Function invoked with the pipeline context to produce a PipelineResult for the stage.
 */

/**
 * Set an optional callback that will be invoked when pipeline progress updates occur.
 *
 * @param callback Function called with the current PipelineContext to report progress.
 */

/**
 * Execute a new pipeline run using the provided prompt and metadata.
 *
 * @param prompt The original prompt or input for the pipeline.
 * @param metadata Key/value metadata to attach to the pipeline run.
 * @returns A PipelineContext representing the completed or final state of the executed pipeline.
 */

/**
 * Retrieve a stored PipelineContext by pipeline identifier.
 *
 * @param pipeline_id Identifier of the pipeline to retrieve.
 * @returns The PipelineContext associated with the given id. If not found, returns a default-constructed PipelineContext.
 */

/**
 * Return a snapshot of currently active pipeline contexts.
 *
 * @returns A vector of PipelineContext objects representing pipelines that are currently active.
 */

/**
 * Transition a pipeline context to the given stage and update its state accordingly.
 *
 * @param ctx The PipelineContext to modify.
 * @param stage The stage to transition the context into.
 */

/**
 * Update progress-related fields on the provided PipelineContext and invoke the progress callback if one is set.
 *
 * @param ctx The PipelineContext to update.
 */

/**
 * Construct a new PipelineStateMachine describing valid pipeline stage transitions.
 */

/**
 * Determine whether a transition from one stage to another is allowed.
 *
 * @param from The current stage.
 * @param to The desired next stage.
 * @returns `true` if the transition is permitted, `false` otherwise.
 */

/**
 * Determine the next logical stage following the provided current stage.
 *
 * @param current The current pipeline stage.
 * @returns The next pipeline stage. If no next stage is defined, returns `current`.
 */

/**
 * Get the list of valid next stages from the given stage.
 *
 * @param from The stage to query transitions for.
 * @returns A vector of PipelineStage values representing allowed subsequent stages.
 */
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