#pragma once
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <functional>

namespace lumen {

class Buffer;
class Backend;

struct QueuedOp {
    std::string op_name;
    std::vector<Buffer*> inputs;
    Buffer* output;
};

// Abstract Hardware Interface
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
private:
    std::vector<size_t> shape_;
    void* device_ptr_; 
    void* host_ptr_;   
    Backend* creator_; 
};

class Runtime {
public:
    Runtime();
    ~Runtime();
    
    // Core API
    Buffer* alloc(const std::vector<size_t>& shape);
    void record(const std::string& op_name, const std::vector<Buffer*>& inputs, Buffer* output);
    void sync(); 
    void execute(const std::string& op_name, const std::vector<Buffer*>& inputs, Buffer* output);

    // New: Backend Management API
    void set_backend(const std::string& backend_name);
    std::string current_backend() const;

private:
    std::map<std::string, std::unique_ptr<Backend>> backends_;
    Backend* active_backend_; // Pointer to the currently selected backend
    std::string active_backend_name_;
    std::vector<QueuedOp> op_queue_;
};

} // namespace lumen