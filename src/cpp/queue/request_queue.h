#pragma once

#include <queue>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <memory>
#include <atomic>
#include "../utils/helpers.h"

/**
 * Priority levels for queued requests.
 *
 * LOW has the lowest precedence, NORMAL is the default, and HIGH has the highest precedence.
 */

/**
 * Represents a request placed into the priority queue.
 *
 * Holds an identifier, payload, priority, enqueue timestamp (ms), and a retry counter.
 */

/**
 * Comparison operator used by the priority queue to order requests.
 *
 * Higher `Priority` values are ordered before lower ones. For equal priorities,
 * earlier `enqueue_time_ms` is ordered before later ones.
 *
 * @param other The request to compare against.
 * @returns `true` if this request should be considered lower priority than `other`, `false` otherwise.
 */

/**
 * Thread-safe, priority-based queue for requests with an optional concurrency limit.
 *
 * The queue enforces a maximum capacity and supports timed dequeueing with shutdown semantics.
 */

/**
 * Create a RequestQueue.
 *
 * @param max_size Maximum number of requests the queue will accept before rejecting new enqueues.
 * @param max_concurrent Maximum concurrent requests allowed (informational; not enforced by the queue itself).
 */

/**
 * Enqueue a new request into the queue.
 *
 * @param data The request payload to enqueue.
 * @param priority The priority level for the request; defaults to NORMAL.
 * @returns `true` if the request was accepted, `false` if the queue is full.
 */

/**
 * Attempt to dequeue the highest-priority request within a timeout.
 *
 * Waits up to `timeout_ms` milliseconds for an item to become available or for the queue to be stopped.
 *
 * @param request Output parameter that will receive the dequeued request on success.
 * @param timeout_ms Maximum time in milliseconds to wait for a request (default 1000).
 * @returns `true` if a request was dequeued into `request`, `false` on timeout or if the queue is stopped/empty.
 */

/**
 * Return the current number of requests in the queue.
 *
 * @returns The number of queued requests.
 */

/**
 * Check whether the queue is empty.
 *
 * @returns `true` if the queue contains no requests, `false` otherwise.
 */

/**
 * Remove all requests from the queue.
 */

/**
 * Mark the queue as running, allowing waiting dequeue operations to proceed.
 */

/**
 * Stop the queue and wake all waiting threads.
 */

/**
 * Query whether the queue is currently running.
 *
 * @returns `true` if the queue has been started and not yet stopped, `false` otherwise.
 */

/**
 * Get the configured concurrent processing limit.
 *
 * @returns The maximum concurrent request count configured for this queue.
 */

/**
 * Processor that spawns worker threads to consume requests from a RequestQueue using a handler.
 *
 * Workers repeatedly dequeue requests and invoke the provided handler until stopped.
 */

/**
 * Construct a RequestProcessor.
 *
 * @param queue Reference to the RequestQueue to consume from.
 * @param handler Function called for each dequeued request.
 * @param worker_count Number of worker threads to spawn when `start()` is called (default 4).
 */

/**
 * Destructor that stops processing and joins worker threads.
 */

/**
 * Start processing by marking running state, starting the queue, and spawning worker threads.
 */

/**
 * Stop processing, signal the queue to stop, join all worker threads, and clear them.
 */

/**
 * Query the number of workers that are actively processing requests.
 *
 * @returns The current count of active workers.
 */
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