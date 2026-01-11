#include "docker_controller.h"
#include "../utils/helpers.h"
#include <sstream>
#include <cstdio>
#include <array>
#include <chrono>
#include <algorithm>

namespace atomic {
namespace sandbox {

DockerController::DockerController(const utils::SandboxConfig& config)
    : config_(config) {
    LOG_INFO("Docker controller initialized with image: ", config_.base_image);
}

DockerController::~DockerController() {
    LOG_DEBUG("Docker controller destroyed");
}

bool DockerController::is_docker_available() {
    try {
        auto output = execute_docker_command({"version"});
        return !output.empty();
    } catch (const std::exception& e) {
        LOG_WARNING("Docker not available: ", e.what());
        return false;
    }
}

std::string DockerController::create_container(
    const std::string& image,
    const std::vector<std::string>& command,
    const ResourceLimits& limits,
    const std::map<std::string, std::string>& env,
    const std::map<std::string, std::string>& volumes
) {
    std::vector<std::string> args = {"create"};
    
    args.push_back("--memory=" + std::to_string(limits.memory_bytes));
    args.push_back("--cpus=" + std::to_string(limits.cpu_millicores / 1000.0));
    args.push_back("--pids-limit=" + std::to_string(limits.max_processes));
    args.push_back("--ulimit");
    args.push_back("nofile=" + std::to_string(limits.max_open_files));
    args.push_back("--network=none");
    args.push_back("--read-only");
    
    for (const auto& [key, value] : env) {
        args.push_back("-e");
        args.push_back(key + "=" + value);
    }
    
    for (const auto& [host_path, container_path] : volumes) {
        args.push_back("-v");
        args.push_back(host_path + ":" + container_path);
    }
    
    args.push_back(image);
    
    for (const auto& cmd : command) {
        args.push_back(cmd);
    }
    
    try {
        std::string output = execute_docker_command(args);
        std::string container_id;
        
        if (parse_container_id(output, container_id)) {
            LOG_INFO("Created container: ", container_id);
            return container_id;
        }
        
        throw std::runtime_error("Failed to parse container ID from output");
        
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to create container: ", e.what());
        throw;
    }
}

bool DockerController::start_container(const std::string& container_id) {
    try {
        execute_docker_command({"start", container_id});
        LOG_DEBUG("Started container: ", container_id);
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to start container: ", e.what());
        return false;
    }
}

bool DockerController::stop_container(const std::string& container_id, int timeout_sec) {
    try {
        execute_docker_command({"stop", "-t", std::to_string(timeout_sec), container_id});
        LOG_DEBUG("Stopped container: ", container_id);
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to stop container: ", e.what());
        return false;
    }
}

bool DockerController::remove_container(const std::string& container_id, bool force) {
    try {
        std::vector<std::string> args = {"rm"};
        if (force) {
            args.push_back("-f");
        }
        args.push_back(container_id);
        
        execute_docker_command(args);
        LOG_DEBUG("Removed container: ", container_id);
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to remove container: ", e.what());
        return false;
    }
}

ExecutionResult DockerController::execute_in_container(
    const std::string& image,
    const std::vector<std::string>& command,
    const ResourceLimits& limits,
    const std::string& working_dir,
    const std::map<std::string, std::string>& env
) {
    ExecutionResult result;
    auto start_time = std::chrono::steady_clock::now();
    
    try {
        std::vector<std::string> docker_command = {"run", "--rm"};
        
        docker_command.push_back("--memory=" + std::to_string(limits.memory_bytes));
        docker_command.push_back("--cpus=" + std::to_string(limits.cpu_millicores / 1000.0));
        docker_command.push_back("--pids-limit=" + std::to_string(limits.max_processes));
        docker_command.push_back("--network=none");
        
        if (!working_dir.empty()) {
            docker_command.push_back("-w");
            docker_command.push_back(working_dir);
        }
        
        for (const auto& [key, value] : env) {
            docker_command.push_back("-e");
            docker_command.push_back(key + "=" + value);
        }
        
        docker_command.push_back(image);
        
        for (const auto& cmd : command) {
            docker_command.push_back(cmd);
        }
        
        result.stdout_output = execute_docker_command(docker_command);
        result.success = true;
        result.exit_code = 0;
        
    } catch (const std::exception& e) {
        result.success = false;
        result.error = e.what();
        result.exit_code = 1;
        LOG_ERROR("Container execution failed: ", e.what());
    }
    
    auto end_time = std::chrono::steady_clock::now();
    result.execution_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time).count();
    
    return result;
}

std::string DockerController::get_container_logs(const std::string& container_id, bool stderr) {
    try {
        std::vector<std::string> args = {"logs"};
        if (!stderr) {
            args.push_back("--no-stderr");
        }
        args.push_back(container_id);
        
        return execute_docker_command(args);
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to get container logs: ", e.what());
        return "";
    }
}

ContainerInfo DockerController::get_container_info(const std::string& container_id) {
    ContainerInfo info;
    info.id = container_id;
    
    try {
        auto output = execute_docker_command({
            "inspect",
            "--format={{.Name}}|{{.Config.Image}}|{{.State.Status}}",
            container_id
        });
        
        auto parts = utils::split(utils::trim(output), '|');
        if (parts.size() >= 3) {
            info.name = parts[0];
            info.image = parts[1];
            info.status = parts[2];
        }
        
    } catch (const std::exception& e) {
        LOG_WARNING("Failed to get container info: ", e.what());
    }
    
    return info;
}

std::vector<ContainerInfo> DockerController::list_containers(bool all) {
    std::vector<ContainerInfo> containers;
    
    try {
        std::vector<std::string> args = {"ps", "--format={{.ID}}|{{.Names}}|{{.Image}}|{{.Status}}"};
        if (all) {
            args.push_back("-a");
        }
        
        auto output = execute_docker_command(args);
        auto lines = utils::split(output, '\n');
        
        for (const auto& line : lines) {
            if (line.empty()) continue;
            
            auto parts = utils::split(line, '|');
            if (parts.size() >= 4) {
                ContainerInfo info;
                info.id = parts[0];
                info.name = parts[1];
                info.image = parts[2];
                info.status = parts[3];
                containers.push_back(info);
            }
        }
        
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to list containers: ", e.what());
    }
    
    return containers;
}

void DockerController::cleanup_old_containers(int max_age_seconds) {
    try {
        auto containers = list_containers(true);
        
        for (const auto& container : containers) {
            if (utils::starts_with(container.status, "Exited") || 
                utils::starts_with(container.status, "Dead")) {
                LOG_INFO("Cleaning up old container: ", container.id);
                remove_container(container.id, true);
            }
        }
        
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to cleanup old containers: ", e.what());
    }
}

std::string DockerController::execute_docker_command(const std::vector<std::string>& args) {
    std::ostringstream command;
    command << "docker";
    
    for (const auto& arg : args) {
        command << " ";
        if (arg.find(' ') != std::string::npos) {
            command << "\"" << arg << "\"";
        } else {
            command << arg;
        }
    }
    
    LOG_DEBUG("Executing: ", command.str());
    
    std::array<char, 128> buffer;
    std::string result;
    
    FILE* pipe = popen(command.str().c_str(), "r");
    if (!pipe) {
        throw std::runtime_error("Failed to execute docker command");
    }
    
    while (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
        result += buffer.data();
    }
    
    int status = pclose(pipe);
    if (status != 0) {
        throw std::runtime_error("Docker command failed with status: " + std::to_string(status));
    }
    
    return result;
}

bool DockerController::parse_container_id(const std::string& output, std::string& container_id) {
    container_id = utils::trim(output);
    
    if (container_id.length() >= 12 && container_id.length() <= 64) {
        return true;
    }
    
    return false;
}

} // namespace sandbox
} // namespace atomic
