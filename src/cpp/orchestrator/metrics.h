#pragma once

#include <string>
#include <map>
#include <atomic>
#include <mutex>
#include <chrono>
#include <vector>

namespace atomic {
namespace orchestrator {

struct RequestMetrics {
    std::atomic<int64_t> total_requests{0};
    std::atomic<int64_t> successful_requests{0};
    std::atomic<int64_t> failed_requests{0};
    std::atomic<int64_t> total_latency_ms{0};
    std::atomic<int64_t> min_latency_ms{INT64_MAX};
    std::atomic<int64_t> max_latency_ms{0};
    
    double average_latency_ms() const {
        int64_t total = total_requests.load();
        return total > 0 ? static_cast<double>(total_latency_ms.load()) / total : 0.0;
    }
    
    double success_rate() const {
        int64_t total = total_requests.load();
        return total > 0 ? static_cast<double>(successful_requests.load()) / total : 0.0;
    }
};

struct PipelineMetrics {
    std::atomic<int64_t> active_pipelines{0};
    std::atomic<int64_t> completed_pipelines{0};
    std::atomic<int64_t> failed_pipelines{0};
    std::atomic<int64_t> total_pipeline_duration_ms{0};
    
    std::map<std::string, int64_t> stage_durations;
    std::mutex stage_mutex;
    
    double average_pipeline_duration_ms() const {
        int64_t completed = completed_pipelines.load();
        return completed > 0 ? static_cast<double>(total_pipeline_duration_ms.load()) / completed : 0.0;
    }
};

struct ResourceMetrics {
    std::atomic<int64_t> active_connections{0};
    std::atomic<int64_t> queue_size{0};
    std::atomic<int64_t> memory_usage_bytes{0};
    std::atomic<int64_t> cpu_usage_percent{0};
};

struct SystemMetrics {
    RequestMetrics requests;
    PipelineMetrics pipelines;
    ResourceMetrics resources;
    
    int64_t start_time_ms;
    
    SystemMetrics() : start_time_ms(std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count()) {}
    
    int64_t uptime_ms() const {
        auto now = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        return now - start_time_ms;
    }
    
    std::map<std::string, double> to_map() const {
        std::map<std::string, double> result;
        
        result["total_requests"] = requests.total_requests.load();
        result["successful_requests"] = requests.successful_requests.load();
        result["failed_requests"] = requests.failed_requests.load();
        result["average_latency_ms"] = requests.average_latency_ms();
        result["min_latency_ms"] = requests.min_latency_ms.load();
        result["max_latency_ms"] = requests.max_latency_ms.load();
        result["success_rate"] = requests.success_rate();
        
        result["active_pipelines"] = pipelines.active_pipelines.load();
        result["completed_pipelines"] = pipelines.completed_pipelines.load();
        result["failed_pipelines"] = pipelines.failed_pipelines.load();
        result["average_pipeline_duration_ms"] = pipelines.average_pipeline_duration_ms();
        
        result["active_connections"] = resources.active_connections.load();
        result["queue_size"] = resources.queue_size.load();
        result["memory_usage_bytes"] = resources.memory_usage_bytes.load();
        result["cpu_usage_percent"] = resources.cpu_usage_percent.load();
        
        result["uptime_ms"] = uptime_ms();
        
        return result;
    }
};

class MetricsCollector {
public:
    static MetricsCollector& instance() {
        static MetricsCollector collector;
        return collector;
    }
    
    void record_request(int64_t latency_ms, bool success) {
        auto& m = metrics_.requests;
        m.total_requests++;
        
        if (success) {
            m.successful_requests++;
        } else {
            m.failed_requests++;
        }
        
        m.total_latency_ms += latency_ms;
        
        int64_t current_min = m.min_latency_ms.load();
        while (latency_ms < current_min && 
               !m.min_latency_ms.compare_exchange_weak(current_min, latency_ms)) {}
        
        int64_t current_max = m.max_latency_ms.load();
        while (latency_ms > current_max && 
               !m.max_latency_ms.compare_exchange_weak(current_max, latency_ms)) {}
    }
    
    void record_pipeline_start() {
        metrics_.pipelines.active_pipelines++;
    }
    
    void record_pipeline_complete(int64_t duration_ms, bool success) {
        metrics_.pipelines.active_pipelines--;
        
        if (success) {
            metrics_.pipelines.completed_pipelines++;
        } else {
            metrics_.pipelines.failed_pipelines++;
        }
        
        metrics_.pipelines.total_pipeline_duration_ms += duration_ms;
    }
    
    void record_stage_duration(const std::string& stage, int64_t duration_ms) {
        std::lock_guard<std::mutex> lock(metrics_.pipelines.stage_mutex);
        metrics_.pipelines.stage_durations[stage] = duration_ms;
    }
    
    void update_resource_metrics(int64_t connections, int64_t queue_size, 
                                 int64_t memory_bytes, int64_t cpu_percent) {
        metrics_.resources.active_connections = connections;
        metrics_.resources.queue_size = queue_size;
        metrics_.resources.memory_usage_bytes = memory_bytes;
        metrics_.resources.cpu_usage_percent = cpu_percent;
    }
    
    SystemMetrics get_metrics() const {
        return metrics_;
    }
    
    void reset() {
        metrics_ = SystemMetrics();
    }
    
private:
    MetricsCollector() = default;
    SystemMetrics metrics_;
};

} // namespace orchestrator
} // namespace atomic
