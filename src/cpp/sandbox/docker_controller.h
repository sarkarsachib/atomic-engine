#pragma once

#include "resource_limits.h"
#include "../utils/config.h"
#include "../utils/logger.h"
#include <string>
#include <vector>
#include <memory>
#include <map>

/**
 * Metadata describing a Docker container managed by DockerController.
 *
 * Contains identifier, human-readable name, image, runtime status, creation timestamp,
 * and arbitrary labels associated with the container.
 */

/**
 * Controller for creating, managing, and inspecting Docker containers for sandboxed execution.
 *
 * Uses the provided SandboxConfig to configure Docker-related operations.
 */

/**
 * Initialize the DockerController with sandbox configuration.
 * @param config Configuration used to control Docker operations and environment.
 */

/**
 * Release resources held by the DockerController.
 */

/**
 * Check whether the Docker daemon is available and accessible.
 * @returns `true` if Docker can be invoked and responds, `false` otherwise.
 */

/**
 * Create a container from the specified image configured for sandboxed execution.
 *
 * The created container is configured with the given command, resource limits, environment
 * variables, and volume bindings but is not started automatically.
 * @param image Name of the container image to create.
 * @param command Command and arguments to set as the container's entrypoint.
 * @param limits ResourceLimits describing CPU, memory, and other resource constraints.
 * @param env Map of environment variables to set inside the container.
 * @param volumes Map of host-to-container volume bindings.
 * @returns The identifier of the created container on success, or an empty string on failure.
 */

/**
 * Start a previously created container.
 * @param container_id Identifier of the container to start.
 * @returns `true` if the container was started successfully, `false` otherwise.
 */

/**
 * Stop a running container, waiting up to the specified timeout for graceful termination.
 * @param container_id Identifier of the container to stop.
 * @param timeout_sec Seconds to wait for graceful shutdown before forcing termination.
 * @returns `true` if the container was stopped (gracefully or after timeout), `false` otherwise.
 */

/**
 * Remove a container from the host.
 * @param container_id Identifier of the container to remove.
 * @param force If `true`, forcefully remove the container (kills if running).
 * @returns `true` if the container was removed successfully, `false` otherwise.
 */

/**
 * Run a command inside a transient container using the specified image and limits, returning execution results.
 *
 * The command runs with the provided working directory and environment variables and is constrained
 * by the given ResourceLimits.
 * @param image Image to use for execution.
 * @param command Command and arguments to execute inside the container.
 * @param limits ResourceLimits to apply to the execution.
 * @param working_dir Working directory inside the container for the executed command.
 * @param env Environment variables to set for the execution.
 * @returns ExecutionResult containing exit status, stdout, stderr, and any execution metadata.
 */

/**
 * Retrieve logs for a container.
 * @param container_id Identifier of the container whose logs to retrieve.
 * @param stderr If `true`, include stderr output; if `false`, return only stdout.
 * @returns Combined logs as a single string.
 */

/**
 * Fetch metadata for a specific container.
 * @param container_id Identifier of the container to inspect.
 * @returns ContainerInfo populated with the container's metadata; fields may be empty if the container is not found.
 */

/**
 * List containers on the host.
 * @param all If `true`, include non-running (exited) containers; if `false`, include only running containers.
 * @returns Vector of ContainerInfo entries for matching containers.
 */

/**
 * Remove containers older than the specified age to reclaim resources.
 * @param max_age_seconds Maximum allowed container age in seconds; containers older than this will be cleaned up.
 */

/**
 * Execute a Docker CLI command constructed from the provided arguments and return its raw output.
 * @param args Components of the docker command to run (e.g., {"run", "--rm", "image", "cmd"}).
 * @returns Raw stdout/stderr output produced by the docker command.
 */

/**
 * Parse a container identifier from Docker command output.
 * @param output Command output potentially containing a container ID.
 * @param container_id Output parameter set to the parsed container identifier on success.
 * @returns `true` if a container ID was successfully parsed and assigned, `false` otherwise.
 */
namespace atomic {
namespace sandbox {

struct ContainerInfo {
    std::string id;
    std::string name;
    std::string image;
    std::string status;
    int64_t created_at = 0;
    std::map<std::string, std::string> labels;
};

class DockerController {
public:
    explicit DockerController(const utils::SandboxConfig& config);
    ~DockerController();
    
    bool is_docker_available();
    
    std::string create_container(
        const std::string& image,
        const std::vector<std::string>& command,
        const ResourceLimits& limits,
        const std::map<std::string, std::string>& env = {},
        const std::map<std::string, std::string>& volumes = {}
    );
    
    bool start_container(const std::string& container_id);
    
    bool stop_container(const std::string& container_id, int timeout_sec = 10);
    
    bool remove_container(const std::string& container_id, bool force = false);
    
    ExecutionResult execute_in_container(
        const std::string& image,
        const std::vector<std::string>& command,
        const ResourceLimits& limits,
        const std::string& working_dir = "/workspace",
        const std::map<std::string, std::string>& env = {}
    );
    
    std::string get_container_logs(const std::string& container_id, bool stderr = true);
    
    ContainerInfo get_container_info(const std::string& container_id);
    
    std::vector<ContainerInfo> list_containers(bool all = false);
    
    void cleanup_old_containers(int max_age_seconds = 3600);
    
private:
    std::string execute_docker_command(const std::vector<std::string>& args);
    
    bool parse_container_id(const std::string& output, std::string& container_id);
    
    utils::SandboxConfig config_;
};

} // namespace sandbox
} // namespace atomic