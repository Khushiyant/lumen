#pragma once
#include <string>
#include <vector>
#include <cstddef>
#include <map>

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

class Runtime {
public:
    Runtime();
    ~Runtime();
    
    std::vector<DeviceInfo> discover_hardware();
    Buffer* alloc(size_t size);

    void run_task(const std::string& name);
    void smart_dispatch(Buffer* buffer, const std::string& shader_source);
    
    // Supporting backends
    void dispatch_cpu(Buffer* buffer);
    void dispatch_compute(Buffer* buffer, const std::string& shader_source);
    void dispatch_npu(Buffer* buffer, float multiplier);
    void run_task_smart(Buffer* buffer, const std::string& shader_source);

    void autotune();

private:
    void probe_gpu(std::vector<DeviceInfo>& devices);
    void probe_npu(std::vector<DeviceInfo>& devices);
    void* mtl_device_; 
    TuningConfig config_;


    std::map<std::string, void*> pipeline_cache_;
};

} // namespace lumen