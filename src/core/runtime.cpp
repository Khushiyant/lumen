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
    Buffer::Buffer(const std::vector<size_t>& shape, void* device_ptr, void* host_ptr, Backend* creator, Runtime* rt) 
        : shape_(shape), device_ptr_(device_ptr), host_ptr_(host_ptr), creator_(creator), runtime_context_(rt) {}
    
    Buffer::~Buffer() { 
        // SAFETY: If a buffer is deleted while operations are pending, flush the queue.
        // This ensures ops finish before memory handles are invalidated.
        if (runtime_context_) {
            runtime_context_->submit();
        }
        if (creator_ && device_ptr_) creator_->free_buffer(device_ptr_);
    }

    void* Buffer::data() {
        if (location_ == BufferLocation::DEVICE_ONLY && runtime_context_) {
            runtime_context_->submit();
        }
        return host_ptr_;
    }

    size_t Buffer::size_bytes() const {
        if (shape_.empty()) return 0;
        size_t total = std::accumulate(shape_.begin(), shape_.end(), 1ULL, std::multiplies<size_t>());
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
        Buffer* buf = allocator->create_buffer(shape);
        buf->set_runtime(this);
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