#include "lumen/lumen.hpp"
#include <iostream>
#include <numeric>
#include <cstring>

namespace lumen {
    // Forward declarations
    std::unique_ptr<Backend> create_cpu_backend();
    
    #ifdef LUMEN_USE_METAL
    std::unique_ptr<Backend> create_metal_backend();
    #endif

    #ifdef LUMEN_USE_CUDA
    std::unique_ptr<Backend> create_cuda_backend();
    #endif

    // Buffer Implementation
    Buffer::Buffer(const std::vector<size_t>& shape, void* device_ptr, void* host_ptr, Backend* creator) 
        : shape_(shape), device_ptr_(device_ptr), host_ptr_(host_ptr), creator_(creator) {}
    
    Buffer::~Buffer() { 
        if (creator_ && device_ptr_) creator_->free_buffer(device_ptr_);
    }

    size_t Buffer::size_bytes() const {
        size_t total = std::accumulate(shape_.begin(), shape_.end(), 1ULL, std::multiplies<size_t>());
        return total * sizeof(float);
    }

    // Runtime Implementation
    Runtime::Runtime() {
        router_ = std::make_unique<Router>();
        
        backends_["cpu"] = create_cpu_backend();

        #ifdef LUMEN_USE_METAL
            backends_["metal"] = create_metal_backend();
        #endif

        #ifdef LUMEN_USE_CUDA
            try { 
                backends_["cuda"] = create_cuda_backend(); 
            } catch (const std::exception& e) { 
                std::cerr << "[Lumen] CUDA Init Failed: " << e.what() << std::endl; 
            }
        #endif

        // Default start backend
        set_backend("cpu");
    }

    Runtime::~Runtime() = default;

    void Runtime::set_backend(const std::string& name) {
        if (backends_.count(name)) {
            active_backend_ = backends_[name].get();
            active_backend_name_ = name;
        }
    }

    std::string Runtime::current_backend() const { return active_backend_name_; }

    Buffer* Runtime::alloc(const std::vector<size_t>& shape) {
        // Simple strategy: Alloc on currently active backend
        // (Advanced migration logic would go here in Phase 2)
        return active_backend_->create_buffer(shape);
    }

    // lumen/src/core/runtime.cpp
void Runtime::execute(const std::string& op_name, const std::vector<Buffer*>& inputs, Buffer* output) {
    Backend* best_backend = router_->select_backend(op_name, output->shape(), backends_);

    // 1. Switch the active backend if the router found a better one
    if (best_backend != active_backend_) {
        active_backend_ = best_backend;
        for (auto const& [name, ptr] : backends_) {
            if (ptr.get() == best_backend) {
                active_backend_name_ = name;
                break;
            }
        }
    }

    // 2. MIGRATE: Ensure all buffers are compatible with the selected backend
    auto ensure_backend = [this](Buffer* buf) {
        if (buf->creator() != active_backend_) {
            // Create a new buffer on the correct hardware
            Buffer* new_storage = active_backend_->create_buffer(buf->shape());
            
            // Copy existing data from host memory
            std::memcpy(new_storage->data(), buf->data(), buf->size_bytes());
            
            // Update the buffer's internal state to the new device handle
            buf->migrate(new_storage->device_handle(), new_storage->data(), active_backend_);
            
            // Cleanup the temporary container (but not the memory we just moved)
            // Note: In a production system, you'd use a more robust hand-off
        }
    };

    for (auto* in : inputs) ensure_backend(in);
    ensure_backend(output);

    // 3. Now safe to execute on the GPU
    active_backend_->execute(op_name, inputs, output);
}

} // namespace lumen