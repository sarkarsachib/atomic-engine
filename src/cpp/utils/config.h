#pragma once

#include <string>
#include <cstdint>

/**
 * @brief Server networking and runtime configuration.
 *
 * Holds server-related settings such as bind address, HTTP and WebSocket ports,
 * thread and connection limits, request timeout, and whether CORS is enabled.
 *
 * @var host IP address or hostname the server binds to (default "0.0.0.0").
 * @var http_port TCP port for HTTP traffic (default 8080).
 * @var ws_port TCP port for WebSocket traffic (default 8081).
 * @var thread_count Number of worker threads the server should use (default 4).
 * @var max_connections Maximum simultaneous connections the server accepts (default 100).
 * @var request_timeout_ms Request timeout in milliseconds (default 30000).
 * @var enable_cors Whether Cross-Origin Resource Sharing is enabled (default true).
 */

/**
 * @brief Inter-process communication (IPC) configuration.
 *
 * Contains settings for the IPC transport, connection pooling, health checks,
 * timeouts and reconnect behavior for agent communication.
 *
 * @var socket_path Filesystem path to the IPC socket (default "/tmp/atomic_llm_agent.sock").
 * @var connection_pool_size Number of pooled connections for IPC (default 4).
 * @var health_check_interval_ms Interval in milliseconds between IPC health checks (default 5000).
 * @var request_timeout_ms Timeout in milliseconds for IPC requests (default 60000).
 * @var reconnect_delay_ms Delay in milliseconds before retrying a failed IPC connection (default 1000).
 * @var max_reconnect_attempts Maximum number of reconnect attempts before giving up (default 5).
 */
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

/**
 * Load configuration values from environment variables.
 *
 * Environment-provided values override the built-in defaults; fields not set
 * in the environment remain at their default values.
 *
 * @returns Config populated from environment variables.
 */
/**
 * Load configuration from a file at the given path.
 *
 * The file's values override the built-in defaults; fields omitted in the
 * file remain at their default values.
 *
 * @param path Filesystem path to the configuration file.
 * @returns Config parsed from the specified file.
 */
/**
 * Produce a configuration populated with the built-in default values.
 *
 * @returns Config with all fields set to their default values.
 */
class ConfigLoader {
public:
    static Config load_from_env();
    static Config load_from_file(const std::string& path);
    static Config load_default();
};

} // namespace utils
} // namespace atomic