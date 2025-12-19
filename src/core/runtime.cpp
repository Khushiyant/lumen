#include "lumen/lumen.hpp"
#include <iostream>

namespace lumen {
    // Forward declare the factory defined in metal_backend.mm
    std::unique_ptr<Backend> create_metal_backend();

    // Buffer Implementation
    Buffer::Buffer(size_t size, void* device_ptr, void* host_ptr) 
        : size_(size), device_ptr_(device_ptr), host_ptr_(host_ptr) {}
    
    Buffer::~Buffer() { 
        // In a real system, we would call backend->free(this) here.
        // For now, we rely on the backend/OS cleaning up.
    }

    // Runtime Implementation
    Runtime::Runtime() {
#ifdef __APPLE__
        gpu_backend_ = create_metal_backend();
        std::cout << "[Lumen] Core Initialized with Metal Backend" << std::endl;
#endif
    }

    Runtime::~Runtime() = default;

    Buffer* Runtime::alloc(size_t size) {
        // CLEAN: We just ask the backend for a Buffer. 
        // No Objective-C code here!
        return gpu_backend_->create_buffer(size);
    }

    void Runtime::record(const std::string& op_name, const std::vector<Buffer*>& inputs, Buffer* output) {
        op_queue_.push_back({op_name, inputs, output});
    }

    void Runtime::sync() {
        if (gpu_backend_) gpu_backend_->sync(op_queue_);
        op_queue_.clear();
    }

    void Runtime::execute(const std::string& op_name, const std::vector<Buffer*>& inputs, Buffer* output) {
        if (gpu_backend_) gpu_backend_->execute(op_name, inputs, output);
    }

} // namespace lumen