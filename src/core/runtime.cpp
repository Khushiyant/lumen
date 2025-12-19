#include "lumen/lumen.hpp"
#include <iostream>
#include <numeric>

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

    // --- FIX: Full Implementation of Execute using Router ---
    void Runtime::execute(const std::string& op_name, const std::vector<Buffer*>& inputs, Buffer* output) {
        // 1. Ask Router for the best backend
        Backend* best_backend = router_->select_backend(op_name, output->shape(), backends_);

        // 2. Check if we are switching backends (for debug/demo)
        if (best_backend != active_backend_) {
            // In a real system, you might want to log this verbose switch
            // std::cout << "[Router] Switching to " << (best_backend == backends_["cpu"].get() ? "CPU" : "GPU") << std::endl;
            active_backend_ = best_backend;
            active_backend_name_ = (best_backend == backends_["cpu"].get() ? "cpu" : "cuda");
        }

        // 3. Execute
        // Since we use Unified Memory (cudaMallocManaged), pointers work on both devices
        active_backend_->execute(op_name, inputs, output);
    }

} // namespace lumen