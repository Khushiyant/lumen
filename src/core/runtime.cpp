#include "lumen/lumen.hpp"
#include <iostream>
#include <numeric>
#include <cstring>

namespace lumen {

    std::unique_ptr<Backend> create_cpu_backend();
    #ifdef LUMEN_USE_METAL
    std::unique_ptr<Backend> create_metal_backend();
    #endif
    #ifdef LUMEN_USE_CUDA
    std::unique_ptr<Backend> create_cuda_backend();
    #endif

    // --- Buffer Implementation ---
    Buffer::Buffer(const std::vector<size_t>& shape, const std::vector<size_t>& strides, 
                   void* device_ptr, void* host_ptr, Backend* creator, size_t offset) 
        : shape_(shape), strides_(strides), device_ptr_(device_ptr), 
          host_ptr_(host_ptr), offset_(offset), creator_(creator) {}
    
    Buffer::~Buffer() { 
        // Only the "owner" buffer (not a view) frees the underlying device memory
        if (!is_view_ && runtime_context_) {
            runtime_context_->submit();
        }
        if (!is_view_ && creator_ && device_ptr_) {
            creator_->free_buffer(device_ptr_);
        }
    }

    Buffer* Buffer::view(const std::vector<size_t>& new_shape, const std::vector<size_t>& new_strides, size_t new_offset) {
        // Create a new buffer object pointing to the same memory
        Buffer* v = new Buffer(new_shape, new_strides, device_ptr_, host_ptr_, creator_, offset_ + new_offset);
        v->is_view_ = true; // Marks this as a non-owning reference
        v->set_runtime(this->runtime_context_);
        return v;
    }

    void* Buffer::data() {
        if (location_ == BufferLocation::DEVICE_ONLY && runtime_context_) {
            runtime_context_->submit();
        }
        return (char*)host_ptr_ + (offset_ * sizeof(float));
    }

    size_t Buffer::size_bytes() const {
        size_t total = 1;
        for (auto d : shape_) total *= d;
        return total * sizeof(float);
    }

    // --- Runtime Implementation ---
    Runtime::Runtime() {
        router_ = std::make_unique<Router>();
        backends_["cpu"] = create_cpu_backend();
        #ifdef LUMEN_USE_METAL
            backends_["metal"] = create_metal_backend();
        #endif
        #ifdef LUMEN_USE_CUDA
            try { backends_["cuda"] = create_cuda_backend(); } catch(...) {}
        #endif

        run_startup_benchmarks();
        
        active_backend_name_ = backends_.count("cuda") ? "cuda" : (backends_.count("metal") ? "metal" : "cpu");
        active_backend_ = backends_[active_backend_name_].get();
    }

    Runtime::~Runtime() { submit(); }

    void Runtime::run_startup_benchmarks() {
        for (auto& [name, backend] : backends_) {
            auto start = std::chrono::high_resolution_clock::now();
            auto* tA = backend->create_buffer({100});
            auto* tB = backend->create_buffer({100});
            auto* tC = backend->create_buffer({100});
            backend->execute("add", {tA, tB}, tC);
            auto end = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(end - start).count();
            metrics_[name]["add"] = {ms, 100.0 / (ms + 1e-6)};
            metrics_[name]["matmul"] = {ms * 5, 100.0 / (ms * 5 + 1e-6)};
            delete tA; delete tB; delete tC;
        }
    }

    void Runtime::execute(const std::string& op_name, const std::vector<Buffer*>& inputs, Buffer* output) {
        Backend* target = router_->select_backend(op_name, output->shape(), backends_, metrics_);
        std::string target_name = "cpu";
        for(auto& [n, p] : backends_) if(p.get() == target) target_name = n;

        queue_.push_back({op_name, inputs, output, target_name});
        output->set_location(BufferLocation::DEVICE_ONLY);
    }

    void Runtime::submit() {
        if (queue_.empty()) return;

        // Move to local storage and clear immediately to prevent recursive sync issues
        std::vector<QueuedOp> current_queue = std::move(queue_);
        queue_.clear(); 

        size_t i = 0;
        while (i < current_queue.size()) {
            std::string target = current_queue[i].target_backend;
            std::vector<QueuedOp> group;
            
            while (i < current_queue.size() && current_queue[i].target_backend == target) {
                group.push_back(current_queue[i]);
                i++;
            }

            if (backends_.count(target)) {
                backends_[target]->sync(group);
            }
            
            for (auto& op : group) {
                op.output->set_location(BufferLocation::BOTH_SYNCED);
            }
        }
    }

    Buffer* Runtime::alloc(const std::vector<size_t>& shape) {
        Backend* allocator = backends_.count("cuda") ? backends_["cuda"].get() :
                             (backends_.count("metal") ? backends_["metal"].get() : backends_["cpu"].get());
        
        // Calculate contiguous strides
        // For shape (3, 4, 5), strides are (20, 5, 1)
        std::vector<size_t> strides(shape.size());
        size_t s = 1;
        for (int i = shape.size() - 1; i >= 0; --i) {
            strides[i] = s;
            s *= shape[i];
        }

        Buffer* base = allocator->create_buffer(shape);
        Buffer* buf = new Buffer(shape, strides, base->device_handle(), base->data(), allocator, 0);
        buf->set_runtime(this);
        
        delete base; // We only needed the raw handles from the backend temporary object
        return buf;
    }

    void Runtime::set_backend(const std::string& name) {
        if (backends_.count(name)) {
            active_backend_ = backends_[name].get();
            active_backend_name_ = name;
        }
    }

    std::string Runtime::current_backend() const { 
        const_cast<Runtime*>(this)->submit();
        return active_backend_name_; 
    }
}