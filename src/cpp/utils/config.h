#pragma once

#include <string>
#include <cstdint>

namespace atomic {
namespace utils {

struct ServerConfig {
    std::string host = "0.0.0.0";
    uint16_t http_port = 8080;
    uint16_t ws_port = 8081;
    int thread_count = 4;
    int max_connections = 100;
    int request_timeout_ms = 30000;
    bool enable_cors = true;
};

struct IPCConfig {
    std::string socket_path = "/tmp/atomic_llm_agent.sock";
    int connection_pool_size = 4;
    int health_check_interval_ms = 5000;
    int request_timeout_ms = 60000;
    int reconnect_delay_ms = 1000;
    int max_reconnect_attempts = 5;
};

struct SandboxConfig {
    std::string docker_socket = "/var/run/docker.sock";
    std::string base_image = "atomic-sandbox:latest";
    int64_t memory_limit_mb = 512;
    int64_t cpu_limit_millicores = 1000;
    int64_t disk_limit_mb = 1024;
    int execution_timeout_sec = 300;
    std::string workspace_path = "/tmp/atomic_sandbox";
};

struct QueueConfig {
    int max_queue_size = 1000;
    int max_concurrent_requests = 10;
    int priority_levels = 3;
    bool enable_backpressure = true;
};

struct Config {
    ServerConfig server;
    IPCConfig ipc;
    SandboxConfig sandbox;
    QueueConfig queue;
    
    std::string artifacts_path = "/tmp/atomic_artifacts";
    std::string log_level = "INFO";
    bool enable_metrics = true;
};

class ConfigLoader {
public:
    static Config load_from_env();
    static Config load_from_file(const std::string& path);
    static Config load_default();
};

} // namespace utils
} // namespace atomic
