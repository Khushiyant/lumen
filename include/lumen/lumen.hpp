#pragma once
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <chrono>
#include <list>
#include <mutex>

namespace lumen {

// --- Pillar 1: Foundational Memory Pool ---
// Manages a cache of reusable memory blocks to avoid expensive syscalls
class MemoryPool {
public:
    struct Block {
        void* ptr;
        size_t size;
    };

    void* acquire(size_t size) {
        std::lock_guard<std::mutex> lock(mutex_);
        // Look for an exact match or a slightly larger block
        for (auto it = free_blocks_.begin(); it != free_blocks_.end(); ++it) {
            if (it->size >= size && it->size <= size * 1.2) { // 20% tolerance
                void* ptr = it->ptr;
                free_blocks_.erase(it);
                return ptr;
            }
        }
        return nullptr; // No suitable block found
    }

    void release(void* ptr, size_t size) {
        std::lock_guard<std::mutex> lock(mutex_);
        free_blocks_.push_back({ptr, size});
        // Optional: Keep pool size in check by pruning old blocks
        if (free_blocks_.size() > 64) {
            // In a real implementation, you'd trigger an actual free here
        }
    }

    ~MemoryPool() {
        // Note: Actual freeing of remaining blocks must be handled 
        // by the Backend that owns the pointers.
    }

private:
    std::list<Block> free_blocks_;
    std::mutex mutex_;
};

class Buffer;
class Backend;
class Router;
class Runtime; 

enum class BufferLocation { HOST_ONLY, DEVICE_ONLY, BOTH_SYNCED };

struct QueuedOp {
    std::string op_name;
    std::vector<Buffer*> inputs;
    Buffer* output;
    std::string target_backend; 
};

struct BackendMetrics {
    double kernel_latency_ms;
    double throughput_mops;
};

class Backend {
public:
    virtual ~Backend() = default;
    virtual Buffer* create_buffer(const std::vector<size_t>& shape) = 0;
    virtual void free_buffer(void* device_ptr, size_t size) = 0; // Updated to include size
    virtual void execute(const std::string& op_name, const std::vector<Buffer*>& inputs, Buffer* output) = 0;
    virtual void sync(std::vector<QueuedOp>& queue) = 0;

protected:
    MemoryPool pool_; // Each backend owns its specific memory pool
};

class Buffer {
public:
    Buffer(const std::vector<size_t>& shape, const std::vector<size_t>& strides, 
           void* device_ptr, void* host_ptr, Backend* creator, size_t offset = 0);
    ~Buffer();
    
    void* data(); 
    void set_runtime(Runtime* rt) { runtime_context_ = rt; }
    
    const std::vector<size_t>& shape() const { return shape_; }
    const std::vector<size_t>& strides() const { return strides_; }
    size_t size_bytes() const;
    void* device_handle() const { return (char*)device_ptr_ + (offset_ * sizeof(float)); }
    Backend* creator() const { return creator_; }

    Buffer* view(const std::vector<size_t>& new_shape, const std::vector<size_t>& new_strides, size_t new_offset = 0);

    void set_location(BufferLocation loc) { location_ = loc; }
    BufferLocation location() const { return location_; }

private:
    std::vector<size_t> shape_;
    std::vector<size_t> strides_;
    void* device_ptr_; 
    void* host_ptr_;   
    size_t offset_; 
    Backend* creator_; 
    Runtime* runtime_context_ = nullptr; 
    BufferLocation location_ = BufferLocation::BOTH_SYNCED;
    bool is_view_ = false; 
};

class Router {
public:
    Backend* select_backend(const std::string& op_name, 
                            const std::vector<size_t>& shape, 
                            const std::map<std::string, std::unique_ptr<Backend>>& backends,
                            const std::map<std::string, std::map<std::string, BackendMetrics>>& metrics);
};

class Runtime {
public:
    Runtime();
    ~Runtime();
    
    Buffer* alloc(const std::vector<size_t>& shape);
    void execute(const std::string& op_name, const std::vector<Buffer*>& inputs, Buffer* output);
    void submit(); 

    void set_backend(const std::string& name);
    std::string current_backend() const;

private:
    void run_startup_benchmarks();
    
    std::map<std::string, std::unique_ptr<Backend>> backends_;
    std::map<std::string, std::map<std::string, BackendMetrics>> metrics_;
    std::vector<QueuedOp> queue_;
    
    Backend* active_backend_; 
    std::string active_backend_name_;
    std::unique_ptr<Router> router_; 
};

} // namespace lumen