#include "lumen/lumen.hpp"
#include <iostream>
#include <numeric>
#include <cstring>
#include <chrono>

namespace lumen {

    // Forward declarations for backend creation functions
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
        if (!is_view_ && runtime_context_) {
            runtime_context_->wait_all(); // Ensure work is done before freeing
        }
        if (!is_view_ && creator_ && device_ptr_) {
            // Pass size to the pool for categorization
            creator_->free_buffer(device_ptr_, this->size_bytes());
        }
    }

    Buffer* Buffer::view(const std::vector<size_t>& new_shape, const std::vector<size_t>& new_strides, size_t new_offset) {
        Buffer* v = new Buffer(new_shape, new_strides, device_ptr_, host_ptr_, creator_, offset_ + new_offset);
        v->is_view_ = true; 
        v->set_runtime(this->runtime_context_);
        return v;
    }

    void* Buffer::data() {
        if (location_ == BufferLocation::DEVICE_ONLY && runtime_context_) {
            runtime_context_->wait_all(); // Sync before CPU access
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
        backends_["cpu"] = create_cpu_backend();
        #ifdef LUMEN_USE_METAL
            backends_["metal"] = create_metal_backend();
        #endif
        #ifdef LUMEN_USE_CUDA
            try { 
                backends_["cuda"] = create_cuda_backend(); 
            } catch(const std::exception& e) {
                // Now reports why CUDA isn't loading
                std::cerr << "[Lumen] CUDA Initialization Failed: " << e.what() << std::endl;
            }
        #endif

        run_startup_benchmarks();
        active_backend_name_ = "dynamic";
        active_backend_ = nullptr; 
        router_ = std::make_unique<Router>();
    }

    Runtime::~Runtime() { wait_all(); }

    void Runtime::run_startup_benchmarks() {
        for (auto& [name, backend] : backends_) {
            size_t dim = 256; 
            auto start = std::chrono::high_resolution_clock::now();
            auto* tA = backend->create_buffer({dim, dim});
            auto* tB = backend->create_buffer({dim, dim});
            auto* tC = backend->create_buffer({dim, dim});
            
            // FIXED: Create a named variable to resolve the lvalue reference error
            std::vector<QueuedOp> startup_queue = {{ "matmul", {tA, tB}, tC, name }};
            auto ev = backend->sync(startup_queue);
            
            if (ev) ev->wait();
            
            auto end = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(end - start).count();
            
            // Store real throughput metrics
            metrics_[name]["matmul"] = {ms, (double)(dim * dim * dim) / (ms * 1e3)};
            delete tA; delete tB; delete tC;
        }
    }

    void Runtime::execute(const std::string& op_name, const std::vector<Buffer*>& inputs, Buffer* output) {
        Backend* target = active_backend_ ? active_backend_ : 
                          router_->select_backend(op_name, output->shape(), backends_, metrics_);
        
        std::string target_name = active_backend_name_;
        if (!active_backend_) {
            for(auto& [n, p] : backends_) if(p.get() == target) target_name = n;
        }

        queue_.push_back({op_name, inputs, output, target_name});
        output->set_location(BufferLocation::DEVICE_ONLY);
    }

    std::vector<std::shared_ptr<Event>> Runtime::submit() {
        if (queue_.empty()) return {};

        std::vector<QueuedOp> current_queue = std::move(queue_);
        queue_.clear(); 

        std::vector<std::shared_ptr<Event>> new_events;
        size_t i = 0;
        while (i < current_queue.size()) {
            // FIXED: 'target' and 'group' are now correctly declared within the loop scope
            std::string target = current_queue[i].target_backend;
            std::vector<QueuedOp> group;
            
            while (i < current_queue.size() && current_queue[i].target_backend == target) {
                group.push_back(current_queue[i]);
                i++;
            }

            if (backends_.count(target)) {
                auto ev = backends_[target]->sync(group);
                if (ev) {
                    new_events.push_back(ev);
                    inflight_events_.push_back(ev);
                }
            }
            
            for (auto& op : group) {
                op.output->set_location(BufferLocation::BOTH_SYNCED);
            }
        }
        return new_events;
    }

    void Runtime::wait_all() {
        for (auto& ev : inflight_events_) {
            if (ev) ev->wait();
        }
        inflight_events_.clear();
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
        } else if (name == "dynamic") {
            active_backend_ = nullptr;
            active_backend_name_ = "dynamic";
        }
    }

    std::string Runtime::current_backend() const { 
        return active_backend_ ? active_backend_name_ : "dynamic"; 
    }
}