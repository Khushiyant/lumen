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
        
        // 1. Initialize CPU as the fallback
        backends_["cpu"] = create_cpu_backend();

        // 2. Try to initialize GPU backends
        #ifdef LUMEN_USE_METAL
            backends_["metal"] = create_metal_backend();
        #endif

        #ifdef LUMEN_USE_CUDA
            try { 
                backends_["cuda"] = create_cuda_backend(); 
            } catch (const std::exception& e) { 
                // Log failure but keep running on CPU
                std::cerr << "[Lumen] CUDA Init Failed: " << e.what() << std::endl; 
            }
        #endif

        // 3. Set initial active backend
        if (backends_.count("cuda")) active_backend_name_ = "cuda";
        else if (backends_.count("metal")) active_backend_name_ = "metal";
        else active_backend_name_ = "cpu";
        
        active_backend_ = backends_[active_backend_name_].get();
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
        // ZERO-COPY ALLOCATOR:
        // We always use the most capable backend for allocation. 
        // Managed/Shared memory is visible to ALL backends without copying.
        Backend* allocator = nullptr;
        if (backends_.count("cuda")) allocator = backends_["cuda"].get();
        else if (backends_.count("metal")) allocator = backends_["metal"].get();
        else allocator = backends_["cpu"].get();
        
        return allocator->create_buffer(shape);
    }

    void Runtime::execute(const std::string& op_name, const std::vector<Buffer*>& inputs, Buffer* output) {
        // Router chooses based on what is actually available in the map
        Backend* best_backend = router_->select_backend(op_name, output->shape(), backends_);

        if (best_backend != active_backend_) {
            active_backend_ = best_backend;
            for (auto const& [name, ptr] : backends_) {
                if (ptr.get() == best_backend) {
                    active_backend_name_ = name;
                    break;
                }
            }
        }

        // Execution is zero-copy because all backends share the same memory pointer
        active_backend_->execute(op_name, inputs, output);
    }

} // namespace lumen