#pragma once
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <chrono>

namespace lumen {

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
    virtual void free_buffer(void* device_ptr) = 0;
    virtual void execute(const std::string& op_name, const std::vector<Buffer*>& inputs, Buffer* output) = 0;
    virtual void sync(std::vector<QueuedOp>& queue) = 0;
};

class Buffer {
public:
    // Added strides and offset for Zero-Copy Views
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

    // Pillar 5: Zero-Copy Views
    // Returns a new Buffer sharing the same data but with different shape/strides
    Buffer* view(const std::vector<size_t>& new_shape, const std::vector<size_t>& new_strides, size_t new_offset = 0);

    void set_location(BufferLocation loc) { location_ = loc; }
    BufferLocation location() const { return location_; }

private:
    std::vector<size_t> shape_;
    std::vector<size_t> strides_;
    void* device_ptr_; 
    void* host_ptr_;   
    size_t offset_; // Offset from the start of the memory block
    Backend* creator_; 
    Runtime* runtime_context_ = nullptr; 
    BufferLocation location_ = BufferLocation::BOTH_SYNCED;
    bool is_view_ = false; // Prevents double-freeing shared memory
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