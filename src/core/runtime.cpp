#include "lumen/lumen.hpp"
#include <iostream>
#include <numeric>
#ifdef LUMEN_USE_CUDA
    std::unique_ptr<Backend> create_cuda_backend();
#endif
namespace lumen {
    // Forward declarations of factories
    std::unique_ptr<Backend> create_metal_backend();
    std::unique_ptr<Backend> create_cpu_backend();

    Buffer::Buffer(const std::vector<size_t>& shape, void* device_ptr, void* host_ptr, Backend* creator) 
        : shape_(shape), device_ptr_(device_ptr), host_ptr_(host_ptr), creator_(creator) {}
    
    Buffer::~Buffer() { 
        if (creator_ && device_ptr_) {
            creator_->free_buffer(device_ptr_);
        }
    }

    size_t Buffer::size_bytes() const {
        size_t total = std::accumulate(shape_.begin(), shape_.end(), 1ULL, std::multiplies<size_t>());
        return total * sizeof(float);
    }

    Runtime::Runtime() {
        // 1. CPU (Always available)
        backends_["cpu"] = create_cpu_backend();

        // 2. Metal (Apple only)
        #ifdef LUMEN_USE_METAL
            backends_["metal"] = create_metal_backend();
        #endif

        // 3. CUDA (NVIDIA only) - NEW!
        #ifdef LUMEN_USE_CUDA
            backends_["cuda"] = create_cuda_backend();
        #endif

        // 4. Default Selection Priority: CUDA -> Metal -> CPU
        if (backends_.count("cuda")) {
            set_backend("cuda");
        } else if (backends_.count("metal")) {
            set_backend("metal");
        } else {
            set_backend("cpu");
        }
    }

    Runtime::~Runtime() = default;

    void Runtime::set_backend(const std::string& name) {
        if (backends_.find(name) != backends_.end()) {
            active_backend_ = backends_[name].get();
            active_backend_name_ = name;
            std::cout << "[Lumen] Switched to backend: " << name << std::endl;
        } else {
            std::cerr << "[Lumen] Warning: Backend '" << name << "' not found!" << std::endl;
        }
    }

    std::string Runtime::current_backend() const {
        return active_backend_name_;
    }

    Buffer* Runtime::alloc(const std::vector<size_t>& shape) {
        // Allocate on the *currently active* backend
        return active_backend_->create_buffer(shape);
    }

    void Runtime::record(const std::string& op_name, const std::vector<Buffer*>& inputs, Buffer* output) {
        op_queue_.push_back({op_name, inputs, output});
    }

    void Runtime::sync() {
        if (active_backend_) active_backend_->sync(op_queue_);
        op_queue_.clear();
    }

    void Runtime::execute(const std::string& op_name, const std::vector<Buffer*>& inputs, Buffer* output) {
        if (active_backend_) active_backend_->execute(op_name, inputs, output);
    }

} // namespace lumen