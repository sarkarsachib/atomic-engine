#pragma once

#include <string>
#include <iostream>
#include <sstream>
#include <mutex>
#include <chrono>
#include <ctime>
#include <iomanip>

namespace atomic {
namespace utils {

enum class LogLevel {
    DEBUG = 0,
    INFO = 1,
    WARNING = 2,
    ERROR = 3,
    CRITICAL = 4
};

class Logger {
public:
    static Logger& instance() {
        static Logger logger;
        return logger;
    }

    void set_level(LogLevel level) {
        std::lock_guard<std::mutex> lock(mutex_);
        level_ = level;
    }

    LogLevel get_level() const {
        return level_;
    }

    template<typename... Args>
    void log(LogLevel level, const Args&... args) {
        if (level < level_) return;

        std::lock_guard<std::mutex> lock(mutex_);
        auto now = std::chrono::system_clock::now();
        auto time = std::chrono::system_clock::to_time_t(now);
        
        std::ostringstream ss;
        ss << "[" << std::put_time(std::localtime(&time), "%Y-%m-%d %H:%M:%S") << "] ";
        ss << "[" << level_to_string(level) << "] ";
        (ss << ... << args) << std::endl;
        
        std::cout << ss.str();
    }

    template<typename... Args>
    void debug(const Args&... args) { log(LogLevel::DEBUG, args...); }

    template<typename... Args>
    void info(const Args&... args) { log(LogLevel::INFO, args...); }

    template<typename... Args>
    void warning(const Args&... args) { log(LogLevel::WARNING, args...); }

    template<typename... Args>
    void error(const Args&... args) { log(LogLevel::ERROR, args...); }

    template<typename... Args>
    void critical(const Args&... args) { log(LogLevel::CRITICAL, args...); }

private:
    Logger() : level_(LogLevel::INFO) {}
    
    std::string level_to_string(LogLevel level) const {
        switch (level) {
            case LogLevel::DEBUG: return "DEBUG";
            case LogLevel::INFO: return "INFO";
            case LogLevel::WARNING: return "WARN";
            case LogLevel::ERROR: return "ERROR";
            case LogLevel::CRITICAL: return "CRIT";
            default: return "UNKNOWN";
        }
    }

    LogLevel level_;
    mutable std::mutex mutex_;
};

#define LOG_DEBUG(...) atomic::utils::Logger::instance().debug(__VA_ARGS__)
#define LOG_INFO(...) atomic::utils::Logger::instance().info(__VA_ARGS__)
#define LOG_WARNING(...) atomic::utils::Logger::instance().warning(__VA_ARGS__)
#define LOG_ERROR(...) atomic::utils::Logger::instance().error(__VA_ARGS__)
#define LOG_CRITICAL(...) atomic::utils::Logger::instance().critical(__VA_ARGS__)

} // namespace utils
} // namespace atomic
