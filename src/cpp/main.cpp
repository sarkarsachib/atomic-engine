#include "orchestrator/orchestrator.h"
#include "utils/logger.h"
#include "utils/config.h"
#include <iostream>
#include <csignal>
#include <atomic>

std::atomic<bool> shutdown_requested{false};

/**
 * @brief Handles termination signals by requesting a graceful shutdown.
 *
 * When invoked with SIGINT or SIGTERM, logs an informational message and sets
 * the global shutdown_requested flag to true to initiate graceful shutdown.
 *
 * @param signal The received POSIX signal number (e.g., SIGINT, SIGTERM).
 */
void signal_handler(int signal) {
    if (signal == SIGINT || signal == SIGTERM) {
        LOG_INFO("Shutdown signal received");
        shutdown_requested = true;
    }
}

/**
 * @brief Initializes and runs the Atomic Orchestrator, setting up configuration, logging, signal handlers, and graceful shutdown.
 *
 * Loads configuration from the environment, configures logging level, constructs and starts the orchestrator, logs runtime endpoints and status, then waits for SIGINT/SIGTERM to stop the orchestrator and exit.
 *
 * @return int `0` on successful shutdown, `1` if the orchestrator fails to start or an unhandled exception occurs.
 */
int main(int argc, char* argv[]) {
    using namespace atomic;
    
    std::cout << R"(
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║            ATOMIC ENGINE - C++ ORCHESTRATOR              ║
    ║                                                           ║
    ║     High-Performance LLM Orchestration Core              ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    )" << std::endl;
    
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);
    
    try {
        LOG_INFO("Loading configuration...");
        auto config = utils::ConfigLoader::load_from_env();
        
        if (config.log_level == "DEBUG") {
            utils::Logger::instance().set_level(utils::LogLevel::DEBUG);
        } else if (config.log_level == "WARNING") {
            utils::Logger::instance().set_level(utils::LogLevel::WARNING);
        } else if (config.log_level == "ERROR") {
            utils::Logger::instance().set_level(utils::LogLevel::ERROR);
        } else {
            utils::Logger::instance().set_level(utils::LogLevel::INFO);
        }
        
        LOG_INFO("Configuration loaded successfully");
        LOG_INFO("  HTTP Server: ", config.server.host, ":", config.server.http_port);
        LOG_INFO("  WebSocket: ", config.server.host, ":", config.server.ws_port);
        LOG_INFO("  IPC Socket: ", config.ipc.socket_path);
        LOG_INFO("  Docker Image: ", config.sandbox.base_image);
        LOG_INFO("  Max Queue Size: ", config.queue.max_queue_size);
        LOG_INFO("  Max Concurrent: ", config.queue.max_concurrent_requests);
        
        LOG_INFO("Creating orchestrator...");
        orchestrator::Orchestrator orchestrator(config);
        
        LOG_INFO("Starting orchestrator...");
        if (!orchestrator.start()) {
            LOG_ERROR("Failed to start orchestrator");
            return 1;
        }
        
        LOG_INFO("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        LOG_INFO("✓ Atomic Orchestrator is running");
        LOG_INFO("  REST API:   http://", config.server.host, ":", config.server.http_port, "/api/generate");
        LOG_INFO("  WebSocket:  ws://", config.server.host, ":", config.server.ws_port, "/ws/stream");
        LOG_INFO("  Health:     http://", config.server.host, ":", config.server.http_port, "/health");
        LOG_INFO("  Metrics:    http://", config.server.host, ":", config.server.http_port, "/api/metrics");
        LOG_INFO("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        LOG_INFO("");
        LOG_INFO("Press Ctrl+C to shutdown gracefully");
        
        while (!shutdown_requested) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
        
        LOG_INFO("Shutting down gracefully...");
        orchestrator.stop();
        
        LOG_INFO("✓ Atomic Orchestrator shut down successfully");
        return 0;
        
    } catch (const std::exception& e) {
        LOG_CRITICAL("Fatal error: ", e.what());
        return 1;
    }
}