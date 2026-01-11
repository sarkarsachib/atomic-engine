#pragma once

#include <cstdint>
#include <string>

namespace atomic {
namespace sandbox {

struct ResourceLimits {
    int64_t memory_bytes = 512 * 1024 * 1024;  // 512MB
    int64_t cpu_millicores = 1000;              // 1 CPU
    int64_t disk_bytes = 1024 * 1024 * 1024;   // 1GB
    int timeout_seconds = 300;                  // 5 minutes
    int max_processes = 100;
    int max_open_files = 1024;
    
    std::string to_docker_args() const {
        std::string args;
        args += " --memory=" + std::to_string(memory_bytes);
        args += " --cpus=" + std::to_string(cpu_millicores / 1000.0);
        args += " --pids-limit=" + std::to_string(max_processes);
        args += " --ulimit nofile=" + std::to_string(max_open_files);
        return args;
    }
};

struct ExecutionResult {
    bool success = false;
    int exit_code = 0;
    std::string stdout_output;
    std::string stderr_output;
    int64_t execution_time_ms = 0;
    int64_t memory_used_bytes = 0;
    std::string error;
};

} // namespace sandbox
} // namespace atomic
