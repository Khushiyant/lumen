#pragma once
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <functional>

namespace lumen {

class Buffer;
class Backend;
class Router; 

struct QueuedOp {
    std::string op_name;
    std::vector<Buffer*> inputs;
    Buffer* output;
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
    Buffer(const std::vector<size_t>& shape, void* device_ptr, void* host_ptr, Backend* creator);
    ~Buffer();
    
    void* data() { return host_ptr_; }
    const std::vector<size_t>& shape() const { return shape_; }
    size_t size_bytes() const;
    void* device_handle() const { return device_ptr_; }
    Backend* creator() const { return creator_; }
    void migrate(void* new_device_ptr, void* new_host_ptr, Backend* new_creator) {
        if (creator_ && device_ptr_) creator_->free_buffer(device_ptr_);
        device_ptr_ = new_device_ptr;
        host_ptr_ = new_host_ptr;
        creator_ = new_creator;
    }
private:
    std::vector<size_t> shape_;
    void* device_ptr_; 
    void* host_ptr_;   
    Backend* creator_; 
};

// --- FIX: Class Definition is here, Implementation is in router.cpp ---
class Router {
public:
    Backend* select_backend(const std::string& op_name, 
                            const std::vector<size_t>& shape, 
                            const std::map<std::string, std::unique_ptr<Backend>>& backends);
};

class Runtime {
public:
    Runtime();
    ~Runtime();
    
    Buffer* alloc(const std::vector<size_t>& shape);
    void execute(const std::string& op_name, const std::vector<Buffer*>& inputs, Buffer* output);

    // Helpers for testing
    void set_backend(const std::string& backend_name);
    std::string current_backend() const;

private:
    std::map<std::string, std::unique_ptr<Backend>> backends_;
    Backend* active_backend_; 
    std::string active_backend_name_;
    std::unique_ptr<Router> router_; 
};

} // namespace lumen