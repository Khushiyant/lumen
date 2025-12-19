#pragma once
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <functional>

namespace lumen {

class Buffer;

struct QueuedOp {
    std::string op_name;
    std::vector<Buffer*> inputs;
    Buffer* output;
};

// Abstract Hardware Interface
class Backend {
public:
    virtual ~Backend() = default;
    // CHANGED: Returns Buffer* directly so Core doesn't need to know about Handles
    virtual Buffer* create_buffer(size_t size) = 0;
    virtual void execute(const std::string& op_name, const std::vector<Buffer*>& inputs, Buffer* output) = 0;
    virtual void sync(std::vector<QueuedOp>& queue) = 0;
};

class Buffer {
public:
    Buffer(size_t size, void* device_ptr, void* host_ptr);
    ~Buffer(); // Defined in runtime.cpp
    void* data() { return host_ptr_; }
    size_t size() const { return size_; }
    void* device_handle() const { return device_ptr_; }
private:
    size_t size_;
    void* device_ptr_; 
    void* host_ptr_;   
};

class Runtime {
public:
    Runtime();
    ~Runtime();
    
    Buffer* alloc(size_t size);
    void record(const std::string& op_name, const std::vector<Buffer*>& inputs, Buffer* output);
    void sync(); 
    void execute(const std::string& op_name, const std::vector<Buffer*>& inputs, Buffer* output);

private:
    std::unique_ptr<Backend> gpu_backend_;
    std::vector<QueuedOp> op_queue_;
};

} // namespace lumen