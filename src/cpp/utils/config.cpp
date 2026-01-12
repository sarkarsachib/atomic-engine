#include "config.h"
#include "logger.h"
#include <cstdlib>
#include <fstream>

namespace atomic {
namespace utils {

/**
 * @brief Builds a Config by applying environment-variable overrides to the default configuration.
 *
 * Loads a default Config and overrides fields when corresponding environment variables are set.
 * Overrides include server settings (ATOMIC_HOST, ATOMIC_HTTP_PORT, ATOMIC_WS_PORT, ATOMIC_THREADS),
 * IPC settings (ATOMIC_IPC_SOCKET, ATOMIC_IPC_POOL_SIZE), sandbox settings (DOCKER_HOST,
 * ATOMIC_SANDBOX_IMAGE, ATOMIC_MEMORY_LIMIT_MB), queue settings (ATOMIC_MAX_QUEUE_SIZE,
 * ATOMIC_MAX_CONCURRENT), and general settings (ATOMIC_ARTIFACTS_PATH, ATOMIC_LOG_LEVEL).
 *
 * @return Config The resulting configuration populated from defaults with any environment-variable overrides applied.
 */
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

/**
 * @brief Load configuration from a file path.
 *
 * Attempts to load configuration values from the file at @p path. If file
 * parsing is not possible or not implemented, the function falls back to
 * constructing a configuration from environment variables and defaults.
 *
 * @param path Filesystem path to the configuration file.
 * @return Config Configuration populated from the file when available,
 *         otherwise populated from environment variables and defaults.
 */
Config ConfigLoader::load_from_file(const std::string& path) {
    LOG_INFO("Loading config from file: ", path);
    // TODO: Implement JSON config file parsing
    return load_from_env();
}

/**
 * @brief Create a configuration populated with the library's defaults.
 *
 * @return Config A Config object initialized with the default settings.
 */
Config ConfigLoader::load_default() {
    Config config;
    LOG_DEBUG("Using default configuration");
    return config;
}

} // namespace utils
} // namespace atomic