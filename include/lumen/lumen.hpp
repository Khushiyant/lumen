#pragma once
#include <string>
#include <vector>
#include <cstddef>
#include <map>
#include <functional> // Required for std::function

namespace lumen {

enum class DeviceType { CPU, GPU, NPU };

struct DeviceInfo {
    DeviceType type;
    std::string name;
};

class Buffer {
public:
    Buffer(size_t size, void* device_ptr, void* host_ptr);
    ~Buffer();
    void* data() { return host_ptr_; }
    size_t size() const { return size_; }
    void* device_handle() const { return device_ptr_; }

private:
    size_t size_;
    void* device_ptr_; 
    void* host_ptr_;   
};

struct TuningConfig {
    size_t cpu_to_npu_threshold = 1000;
    size_t npu_to_gpu_threshold = 50000;
};

// Define the worker type for the registry
using OpWorker = std::function<void(const std::vector<Buffer*>&, Buffer*)>;

class Runtime {
public:
    Runtime();
    ~Runtime();
    
    Buffer* alloc(size_t size);

    void execute(const std::string& op_name, 
                 const std::vector<Buffer*>& inputs, 
                 Buffer* output);
    void calibrate_thresholds();

private:
    void run_cpu(const std::string& op_name, const std::vector<Buffer*>& inputs, Buffer* output);
    void run_gpu(const std::string& op_name, const std::vector<Buffer*>& inputs, Buffer* output);

    void* mtl_device_;
    std::map<std::string, void*> pipeline_cache_;
    TuningConfig config_;

    // Registry maps for "Plug-and-Play" operations
    std::map<std::string, OpWorker> cpu_ops_;
    std::map<std::string, OpWorker> gpu_ops_;
    
    void register_ops();
};

} // namespace lumen