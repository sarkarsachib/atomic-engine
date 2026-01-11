#pragma once

#include <queue>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <memory>
#include <atomic>
#include "../utils/helpers.h"

namespace atomic {
namespace queue {

enum class Priority {
    LOW = 0,
    NORMAL = 1,
    HIGH = 2
};

template<typename T>
struct QueuedRequest {
    std::string id;
    T data;
    Priority priority;
    int64_t enqueue_time_ms;
    int retry_count = 0;
    
    bool operator<(const QueuedRequest& other) const {
        if (priority != other.priority) {
            return priority < other.priority;
        }
        return enqueue_time_ms > other.enqueue_time_ms;
    }
};

template<typename T>
class RequestQueue {
public:
    RequestQueue(size_t max_size = 1000, size_t max_concurrent = 10)
        : max_size_(max_size), max_concurrent_(max_concurrent), running_(false) {}
    
    bool enqueue(const T& data, Priority priority = Priority::NORMAL) {
        std::unique_lock<std::mutex> lock(mutex_);
        
        if (queue_.size() >= max_size_) {
            return false;
        }
        
        QueuedRequest<T> request;
        request.id = utils::generate_uuid();
        request.data = data;
        request.priority = priority;
        request.enqueue_time_ms = utils::timestamp_ms();
        
        queue_.push(request);
        cv_.notify_one();
        
        return true;
    }
    
    bool try_dequeue(QueuedRequest<T>& request, int timeout_ms = 1000) {
        std::unique_lock<std::mutex> lock(mutex_);
        
        if (!cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                         [this]() { return !queue_.empty() || !running_; })) {
            return false;
        }
        
        if (queue_.empty()) {
            return false;
        }
        
        request = queue_.top();
        queue_.pop();
        
        return true;
    }
    
    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }
    
    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }
    
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        while (!queue_.empty()) {
            queue_.pop();
        }
    }
    
    void start() {
        running_ = true;
    }
    
    void stop() {
        running_ = false;
        cv_.notify_all();
    }
    
    bool is_running() const {
        return running_;
    }
    
    size_t get_concurrent_limit() const {
        return max_concurrent_;
    }
    
private:
    size_t max_size_;
    size_t max_concurrent_;
    std::atomic<bool> running_;
    
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::priority_queue<QueuedRequest<T>> queue_;
};

template<typename T>
class RequestProcessor {
public:
    using ProcessHandler = std::function<void(const QueuedRequest<T>&)>;
    
    RequestProcessor(RequestQueue<T>& queue, ProcessHandler handler, size_t worker_count = 4)
        : queue_(queue), handler_(handler), worker_count_(worker_count), running_(false) {}
    
    ~RequestProcessor() {
        stop();
    }
    
    void start() {
        if (running_) return;
        
        running_ = true;
        queue_.start();
        
        for (size_t i = 0; i < worker_count_; ++i) {
            workers_.emplace_back([this]() {
                worker_loop();
            });
        }
    }
    
    void stop() {
        if (!running_) return;
        
        running_ = false;
        queue_.stop();
        
        for (auto& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
        workers_.clear();
    }
    
    size_t active_workers() const {
        return active_count_;
    }
    
private:
    void worker_loop() {
        while (running_) {
            QueuedRequest<T> request;
            if (queue_.try_dequeue(request, 1000)) {
                active_count_++;
                
                try {
                    handler_(request);
                } catch (const std::exception& e) {
                    // Log error
                }
                
                active_count_--;
            }
        }
    }
    
    RequestQueue<T>& queue_;
    ProcessHandler handler_;
    size_t worker_count_;
    std::atomic<bool> running_;
    std::atomic<size_t> active_count_{0};
    std::vector<std::thread> workers_;
};

} // namespace queue
} // namespace atomic
