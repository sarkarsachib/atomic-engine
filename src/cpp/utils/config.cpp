#include "config.h"
#include "logger.h"
#include <cstdlib>
#include <fstream>

namespace atomic {
namespace utils {

Config ConfigLoader::load_from_env() {
    Config config = load_default();
    
    auto get_env = [](const char* name, const std::string& default_val) -> std::string {
        const char* val = std::getenv(name);
        return val ? std::string(val) : default_val;
    };
    
    auto get_env_int = [&](const char* name, int default_val) -> int {
        std::string val = get_env(name, "");
        return val.empty() ? default_val : std::stoi(val);
    };
    
    // Server config
    config.server.host = get_env("ATOMIC_HOST", config.server.host);
    config.server.http_port = get_env_int("ATOMIC_HTTP_PORT", config.server.http_port);
    config.server.ws_port = get_env_int("ATOMIC_WS_PORT", config.server.ws_port);
    config.server.thread_count = get_env_int("ATOMIC_THREADS", config.server.thread_count);
    
    // IPC config
    config.ipc.socket_path = get_env("ATOMIC_IPC_SOCKET", config.ipc.socket_path);
    config.ipc.connection_pool_size = get_env_int("ATOMIC_IPC_POOL_SIZE", config.ipc.connection_pool_size);
    
    // Sandbox config
    config.sandbox.docker_socket = get_env("DOCKER_HOST", config.sandbox.docker_socket);
    config.sandbox.base_image = get_env("ATOMIC_SANDBOX_IMAGE", config.sandbox.base_image);
    config.sandbox.memory_limit_mb = get_env_int("ATOMIC_MEMORY_LIMIT_MB", config.sandbox.memory_limit_mb);
    
    // Queue config
    config.queue.max_queue_size = get_env_int("ATOMIC_MAX_QUEUE_SIZE", config.queue.max_queue_size);
    config.queue.max_concurrent_requests = get_env_int("ATOMIC_MAX_CONCURRENT", config.queue.max_concurrent_requests);
    
    // General config
    config.artifacts_path = get_env("ATOMIC_ARTIFACTS_PATH", config.artifacts_path);
    config.log_level = get_env("ATOMIC_LOG_LEVEL", config.log_level);
    
    return config;
}

Config ConfigLoader::load_from_file(const std::string& path) {
    LOG_INFO("Loading config from file: ", path);
    // TODO: Implement JSON config file parsing
    return load_from_env();
}

Config ConfigLoader::load_default() {
    Config config;
    LOG_DEBUG("Using default configuration");
    return config;
}

} // namespace utils
} // namespace atomic
