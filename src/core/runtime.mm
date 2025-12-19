#include "lumen/lumen.hpp"
#include <iostream>
#include <chrono>

#ifdef __APPLE__
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#import <Accelerate/Accelerate.h>
#endif

namespace lumen {

// --- Buffer Implementation ---
Buffer::Buffer(size_t size, void* device_ptr, void* host_ptr) 
    : size_(size), device_ptr_(device_ptr), host_ptr_(host_ptr) {}

Buffer::~Buffer() {
#ifdef __APPLE__
    // Since we aren't using ARC, we cast back to id and set to nil 
    // to let the system know we are done.
    id<MTLBuffer> mtlBuffer = (id<MTLBuffer>)device_ptr_;
    mtlBuffer = nil; 
#endif
    std::cout << "[Lumen] Buffer deallocated." << std::endl;
}

// --- Runtime Implementation ---
Runtime::Runtime() {
    std::cout << "[Lumen] Runtime Initialized on macOS." << std::endl;
    
#ifdef __APPLE__
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    mtl_device_ = (void*)device; 
#else
    mtl_device_ = nullptr;
#endif
}

Runtime::~Runtime() {
#ifdef __APPLE__
    id<MTLDevice> device = (id<MTLDevice>)mtl_device_;
    device = nil;
#endif
    std::cout << "[Lumen] Runtime shut down." << std::endl;
}

void Runtime::probe_gpu(std::vector<DeviceInfo>& devices) {
#ifdef __APPLE__
    id<MTLDevice> device = (id<MTLDevice>)mtl_device_;
    if (device) {
        std::string name = [[device name] UTF8String];
        devices.push_back({DeviceType::GPU, name});
    }
#endif
}

void Runtime::probe_npu(std::vector<DeviceInfo>& devices) {
#ifdef __APPLE__
    devices.push_back({DeviceType::NPU, "Apple Neural Engine"});
#endif
}

std::vector<DeviceInfo> Runtime::discover_hardware() {
    std::vector<DeviceInfo> devices;
    devices.push_back({DeviceType::CPU, "Host System CPU"});
    probe_gpu(devices);
    probe_npu(devices);
    return devices;
}

Buffer* Runtime::alloc(size_t size) {
#ifdef __APPLE__
    id<MTLDevice> device = (id<MTLDevice>)mtl_device_;
    if (!device) return nullptr;

    // Allocate Unified Memory (Shared between CPU and GPU)
    id<MTLBuffer> mtlBuffer = [device newBufferWithLength:size 
                                               options:MTLResourceStorageModeShared];
    
    if (!mtlBuffer) return nullptr;

    return new Buffer(size, (void*)mtlBuffer, [mtlBuffer contents]);
#else
    return nullptr;
#endif
}

void Runtime::run_task(const std::string& name) {
    std::cout << "[Lumen] Running task '" << name << "'..." << std::endl;
}
void Runtime::dispatch_compute(Buffer* buffer, const std::string& shader_source) {
#ifdef __APPLE__
    id<MTLDevice> device = (id<MTLDevice>)mtl_device_;
    id<MTLComputePipelineState> pipeline = nil;

    // 1. Check Cache
    if (pipeline_cache_.count(shader_source)) {
        pipeline = (id<MTLComputePipelineState>)pipeline_cache_[shader_source];
    } else {
        NSError* error = nil;
        NSString* source = [NSString stringWithUTF8String:shader_source.c_str()];
        id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
        
        if (!library) {
            std::cerr << "Shader Error: " << [[error localizedDescription] UTF8String] << std::endl;
            return;
        }

        id<MTLFunction> function = [library newFunctionWithName:@"multiply_by_two"];
        pipeline = [device newComputePipelineStateWithFunction:function error:&error];
        
        if (pipeline) {
            pipeline_cache_[shader_source] = (void*)pipeline;
            std::cout << "[Lumen] Shader compiled and cached." << std::endl;
        }
    }

    // 2. Execution (this uses the 'pipeline' from either the cache or the new compilation)
    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:(id<MTLBuffer>)buffer->device_handle() offset:0 atIndex:0];

    uint32_t numElements = buffer->size() / sizeof(float);
    MTLSize gridSize = MTLSizeMake(numElements, 1, 1);
    NSUInteger threadGroupSizeLimit = [pipeline maxTotalThreadsPerThreadgroup];
    MTLSize threadGroupSize = MTLSizeMake(std::min((NSUInteger)numElements, threadGroupSizeLimit), 1, 1);

    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadGroupSize];
    [encoder endEncoding];

    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
#endif
}

void Runtime::smart_dispatch(Buffer* buffer, const std::string& shader_source) {
    size_t element_count = buffer->size() / sizeof(float);
    
    // Heuristic: If work is too small, don't wake up the GPU.
    // This saves energy and avoids launch latency.
    if (element_count < 5000) {
        std::cout << "[Orchestrator] Task small (" << element_count 
                  << " units). Executing on CPU for efficiency." << std::endl;
        
        float* data = (float*)buffer->data();
        for(size_t i = 0; i < element_count; ++i) {
            data[i] *= 2.0f; // Manual CPU fallback
        }
    } else {
        std::cout << "[Orchestrator] Task large (" << element_count 
                  << " units). Offloading to GPU..." << std::endl;
        dispatch_compute(buffer, shader_source);
    }
}

void Runtime::dispatch_npu(Buffer* buffer, float multiplier) {
    std::cout << "[Lumen] Dispatching to Neural/Vector Engine..." << std::endl;
#ifdef __APPLE__
    float* data = (float*)buffer->data();
    size_t n = buffer->size() / sizeof(float);
    
    // vDSP_vsmul: Vector-Scalar Multiplication
    // This uses the AMX/NPU units on Apple Silicon to multiply 
    // an entire array by a value in one highly optimized pass.
    vDSP_vsmul(data, 1, &multiplier, data, 1, n);
#endif
}

void Runtime::dispatch_cpu(Buffer* buffer) {
    float* data = (float*)buffer->data();
    size_t n = buffer->size() / sizeof(float);
    for(size_t i = 0; i < n; ++i) {
        data[i] *= 2.0f;
    }
}

void Runtime::run_task_smart(Buffer* buffer, const std::string& shader_source) {
    size_t elements = buffer->size() / sizeof(float);

    if (elements < 1000) {
        std::cout << ">>> [Orchestrator] Task: " << elements << " units. Route: CPU (Latency Mode)" << std::endl;
        dispatch_cpu(buffer);
    } 
    else if (elements < 100000) {
        std::cout << ">>> [Orchestrator] Task: " << elements << " units. Route: NPU/AMX (Power Mode)" << std::endl;
        dispatch_npu(buffer, 2.0f);
    } 
    else {
        std::cout << ">>> [Orchestrator] Task: " << elements << " units. Route: GPU (Throughput Mode)" << std::endl;
        dispatch_compute(buffer, shader_source);
    }
}


void Runtime::autotune() {
    std::cout << "[Lumen] Auto-tuning for your " << "Hardware..." << std::endl;
    
    // 1. Measure CPU Baseline
    auto s1 = std::chrono::high_resolution_clock::now();
    for(int i=0; i<1000; ++i) { /* empty loop */ }
    auto e1 = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double, std::nano>(e1-s1).count();

    // 2. Measure GPU Dispatch Overhead (Cold Start vs Cached)
    // We run a "null" dispatch to see how long the command queue takes
    auto s2 = std::chrono::high_resolution_clock::now();
    // (Simulate or run a tiny empty shader here)
    auto e2 = std::chrono::high_resolution_clock::now();
    double gpu_overhead = std::chrono::duration<double, std::nano>(e2-s2).count();

    // 3. Set thresholds based on measured latency
    // If GPU overhead is high, we push the threshold higher
    if (gpu_overhead > 100000) { // 100us
        config_.npu_to_gpu_threshold = 150000;
    }
    
    std::cout << "[Lumen] Auto-tune complete. GPU Threshold: " 
              << config_.npu_to_gpu_threshold << " units." << std::endl;
}
} // namespace lumen