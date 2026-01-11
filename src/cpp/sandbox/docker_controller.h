#pragma once

#include "resource_limits.h"
#include "../utils/config.h"
#include "../utils/logger.h"
#include <string>
#include <vector>
#include <memory>
#include <map>

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
