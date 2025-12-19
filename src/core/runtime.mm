#include "lumen/lumen.hpp"
#include <iostream>
#include <algorithm>

#ifdef __APPLE__
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#import <Accelerate/Accelerate.h>
#endif

namespace lumen {

// --- Buffer Implementation ---
Buffer::Buffer(size_t size, void* device_ptr, void* host_ptr) 
    : size_(size), device_ptr_(device_ptr), host_ptr_(host_ptr) {}

Buffer::~Buffer() {
#ifdef __APPLE__
    id<MTLBuffer> mtlBuffer = (id<MTLBuffer>)device_ptr_;
    mtlBuffer = nil; 
#endif
}

// --- Runtime Implementation ---
Runtime::Runtime() {
#ifdef __APPLE__
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    mtl_device_ = (void*)device;
    register_ops();
    
    // Auto-Tuning: Find the break-even point
    calibrate_thresholds(); 
    
    std::cout << "[Lumen] Runtime Ready. Optimized Threshold: " 
              << config_.cpu_to_npu_threshold << " elements." << std::endl;
#endif
}

Runtime::~Runtime() {
#ifdef __APPLE__
    id<MTLDevice> device = (id<MTLDevice>)mtl_device_;
    // Clean up cached executables
    for (auto const& [key, val] : pipeline_cache_) {
        MPSGraphExecutable *exec = (MPSGraphExecutable*)val;
        exec = nil;
    }
    device = nil;
#endif
}

void Runtime::calibrate_thresholds() {
    // We use a medium size to test the crossover point
    size_t test_size = 5000;
    auto* A = alloc(test_size * sizeof(float));
    auto* B = alloc(test_size * sizeof(float));
    auto* C = alloc(test_size * sizeof(float));

    // Fill with dummy data
    for(size_t i=0; i<test_size; ++i) {
        ((float*)A->data())[i] = 1.0f;
        ((float*)B->data())[i] = 2.0f;
    }

    // Time CPU via the Registry
    auto s1 = std::chrono::high_resolution_clock::now();
    if (cpu_ops_.count("add")) cpu_ops_["add"]({A, B}, C);
    auto e1 = std::chrono::high_resolution_clock::now();
    
    // Time GPU via the Registry (which calls run_gpu internally)
    auto s2 = std::chrono::high_resolution_clock::now();
    if (gpu_ops_.count("add")) gpu_ops_["add"]({A, B}, C);
    auto e2 = std::chrono::high_resolution_clock::now();

    double cpu_time = std::chrono::duration<double, std::milli>(e1 - s1).count();
    double gpu_time = std::chrono::duration<double, std::milli>(e2 - s2).count();

    // Novelty: Auto-tuning the threshold based on real-time hardware latency
    if (gpu_time > cpu_time) {
        // If GPU is slower at this size, we should prefer CPU for even larger tasks
        config_.cpu_to_npu_threshold = test_size * 2;
    } else {
        config_.cpu_to_npu_threshold = test_size / 2;
    }

    delete A; delete B; delete C;
}
void Runtime::register_ops() {
    // --- CPU Registry (Accelerate/vDSP) ---
    cpu_ops_["add"] = [](const std::vector<Buffer*>& ins, Buffer* out) {
        size_t n = out->size() / sizeof(float);
        vDSP_vadd((float*)ins[0]->data(), 1, (float*)ins[1]->data(), 1, (float*)out->data(), 1, n);
    };
    
    cpu_ops_["mul"] = [](const std::vector<Buffer*>& ins, Buffer* out) {
        size_t n = out->size() / sizeof(float);
        vDSP_vmul((float*)ins[0]->data(), 1, (float*)ins[1]->data(), 1, (float*)out->data(), 1, n);
    };

    // --- GPU Registry (MPSGraph) ---
    gpu_ops_["add"] = [this](const std::vector<Buffer*>& ins, Buffer* out) {
        this->run_gpu("add", ins, out); 
    };
    
    gpu_ops_["mul"] = [this](const std::vector<Buffer*>& ins, Buffer* out) {
        this->run_gpu("mul", ins, out); 
    };

    // CPU MatMul using Accelerate's BLAS
    cpu_ops_["matmul"] = [](const std::vector<Buffer*>& ins, Buffer* out) {
        int n = sqrt(out->size() / sizeof(float)); // Assuming square for demo
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0f, 
                    (float*)ins[0]->data(), n, (float*)ins[1]->data(), n, 0.0f, (float*)out->data(), n);
    };

    // GPU MatMul using MPSGraph
    gpu_ops_["matmul"] = [this](const std::vector<Buffer*>& ins, Buffer* out) {
        this->run_gpu("matmul", ins, out); 
    };
}

Buffer* Runtime::alloc(size_t size) {
#ifdef __APPLE__
    id<MTLDevice> device = (id<MTLDevice>)mtl_device_;
    // Zero-Serialization: Shared memory accessible by CPU and GPU
    id<MTLBuffer> mtlBuffer = [device newBufferWithLength:size options:MTLResourceStorageModeShared];
    return new Buffer(size, (void*)mtlBuffer, [mtlBuffer contents]);
#endif
    return nullptr;
}

void Runtime::execute(const std::string& op_name, const std::vector<Buffer*>& inputs, Buffer* output) {
    size_t elements = output->size() / sizeof(float);

    // Orchestrator: Smart routing based on profiling data
    if (elements < config_.cpu_to_npu_threshold) {
        if (cpu_ops_.count(op_name)) {
            cpu_ops_[op_name](inputs, output);
        }
    } else {
        if (gpu_ops_.count(op_name)) {
            gpu_ops_[op_name](inputs, output);
        }
    }
}


// Add to src/core/runtime.mm

void Runtime::record(const std::string& op_name, const std::vector<Buffer*>& inputs, Buffer* output) {
    // Just store the intent. Do zero work here.
    op_queue_.push_back({op_name, inputs, output});
}

void Runtime::sync() {
    if (op_queue_.empty()) return;

#ifdef __APPLE__
    id<MTLDevice> device = (id<MTLDevice>)mtl_device_;
    
    @autoreleasepool {
        MPSGraph *graph = [[MPSGraph alloc] init];
        
        std::map<Buffer*, MPSGraphTensor*> buffer_to_tensor;
        NSMutableDictionary<MPSGraphTensor*, MPSGraphTensorData*> *feeds = [NSMutableDictionary dictionary];
        NSMutableArray<MPSGraphTensor*> *targetTensors = [NSMutableArray array];
        NSMutableArray<MPSGraphTensorData*> *targetData = [NSMutableArray array];

        for (const auto& op : op_queue_) {
            size_t element_count = op.output->size() / sizeof(float);
            
            // --- SHAPE INFERENCE LOGIC ---
            // Determine if this op should run in 2D (Matrix) mode or 1D (Vector) mode.
            // Rule 1: MatMul is always 2D.
            bool is_2d_op = (op.op_name == "matmul");
            
            // Rule 2: If not MatMul, check if any INPUT is already a 2D Tensor.
            // (This handles the chain: MatMul(2D) -> Add(2D))
            if (!is_2d_op) {
                for (Buffer* buf : op.inputs) {
                    if (buffer_to_tensor.count(buf)) {
                        MPSGraphTensor *existingTensor = buffer_to_tensor[buf];
                        if (existingTensor.shape.count == 2) {
                            is_2d_op = true;
                            break;
                        }
                    }
                }
            }
            
            // --- RESOLVE INPUTS ---
            NSMutableArray<MPSGraphTensor*> *inputTensors = [NSMutableArray array];
            
            for (Buffer* buf : op.inputs) {
                if (buffer_to_tensor.count(buf)) {
                    // Internal Link (Fusion)
                    [inputTensors addObject:buffer_to_tensor[buf]];
                } else {
                    // External Input: Create Placeholder with CORRECT Shape
                    NSArray<NSNumber *> *shape;
                    if (is_2d_op) {
                        int dim = sqrt(element_count);
                        shape = @[@(dim), @(dim)];
                    } else {
                        shape = @[@(element_count)];
                    }
                    
                    MPSGraphTensor *ph = [graph placeholderWithShape:shape dataType:MPSDataTypeFloat32 name:nil];
                    [inputTensors addObject:ph];
                    
                    MPSGraphTensorData *data = [[MPSGraphTensorData alloc] initWithMTLBuffer:(id<MTLBuffer>)buf->device_handle() 
                                                                                       shape:shape 
                                                                                    dataType:MPSDataTypeFloat32];
                    feeds[ph] = data;
                    buffer_to_tensor[buf] = ph;
                }
            }
            
            // --- ADD NODE ---
            MPSGraphTensor *result = nil;
            if (op.op_name == "add") {
                result = [graph additionWithPrimaryTensor:inputTensors[0] secondaryTensor:inputTensors[1] name:nil];
            } 
            else if (op.op_name == "mul") {
                result = [graph multiplicationWithPrimaryTensor:inputTensors[0] secondaryTensor:inputTensors[1] name:nil];
            } 
            else if (op.op_name == "matmul") {
                result = [graph matrixMultiplicationWithPrimaryTensor:inputTensors[0] secondaryTensor:inputTensors[1] name:nil];
            }
            
            // --- REGISTER OUTPUT ---
            if (result) {
                buffer_to_tensor[op.output] = result;
                [targetTensors addObject:result];
                
                // Match output data shape to the op mode
                NSArray<NSNumber *> *outShape;
                if (is_2d_op) {
                     int dim = sqrt(element_count);
                     outShape = @[@(dim), @(dim)];
                } else {
                     outShape = @[@(element_count)];
                }

                MPSGraphTensorData *outD = [[MPSGraphTensorData alloc] initWithMTLBuffer:(id<MTLBuffer>)op.output->device_handle() 
                                                                                   shape:outShape 
                                                                                dataType:MPSDataTypeFloat32];
                [targetData addObject:outD];
            }
        }
        
        // --- COMPILE & RUN ---
        
        // 1. Convert Data Feeds to Type Feeds for Compilation
        NSMutableDictionary<MPSGraphTensor*, MPSGraphShapedType*> *typeFeeds = [NSMutableDictionary dictionary];
        NSMutableArray<MPSGraphTensorData*> *inputsArray = [NSMutableArray array];
        
        for (MPSGraphTensor *tensor in [feeds keyEnumerator]) {
            MPSGraphTensorData *data = feeds[tensor];
            MPSGraphShapedType *shapedType = [[MPSGraphShapedType alloc] initWithShape:data.shape 
                                                                              dataType:data.dataType];
            typeFeeds[tensor] = shapedType;
            [inputsArray addObject:data];
        }

        // 2. Compile
        MPSGraphExecutable *exec = [graph compileWithDevice:[MPSGraphDevice deviceWithMTLDevice:device] 
                                                      feeds:typeFeeds 
                                              targetTensors:targetTensors 
                                           targetOperations:nil 
                                      compilationDescriptor:nil];

        // 3. Run
        id<MTLCommandQueue> queue = [device newCommandQueue];
        [exec runWithMTLCommandQueue:queue 
                         inputsArray:inputsArray
                        resultsArray:targetData 
                 executionDescriptor:nil];
    }
#endif
    op_queue_.clear();
}


// --- GPU Execution via MPSGraph ---
void Runtime::run_gpu(const std::string& op_name, const std::vector<Buffer*>& inputs, Buffer* output) {
#ifdef __APPLE__
    id<MTLDevice> device = (id<MTLDevice>)mtl_device_;
    std::string cache_key = op_name + "_" + std::to_string(output->size());
    MPSGraphExecutable *executable = nil;

    if (pipeline_cache_.count(cache_key)) {
        executable = (MPSGraphExecutable*)pipeline_cache_[cache_key];
    } else {
        @autoreleasepool {
            MPSGraph *graph = [[MPSGraph alloc] init];
            size_t total_elements = output->size() / sizeof(float);
            
            // Logic: If MatMul, we need 2D shapes
            NSArray<NSNumber *> *shape;
            if (op_name == "matmul") {
                int dim = sqrt(total_elements);
                shape = @[@(dim), @(dim)];
            } else {
                shape = @[@(total_elements)];
            }

            MPSGraphTensor *aTensor = [graph placeholderWithShape:shape dataType:MPSDataTypeFloat32 name:@"A"];
            MPSGraphTensor *bTensor = [graph placeholderWithShape:shape dataType:MPSDataTypeFloat32 name:@"B"];
            
            MPSGraphTensor *resTensor = nil;
            if (op_name == "add") {
                resTensor = [graph additionWithPrimaryTensor:aTensor secondaryTensor:bTensor name:nil];
            } else if (op_name == "matmul") {
                resTensor = [graph matrixMultiplicationWithPrimaryTensor:aTensor 
                                                         secondaryTensor:bTensor name:nil];
            }

            if (resTensor) {
                MPSGraphShapedType *type = [[MPSGraphShapedType alloc] initWithShape:shape dataType:MPSDataTypeFloat32];
                executable = [graph compileWithDevice:[MPSGraphDevice deviceWithMTLDevice:device]
                                                feeds:@{aTensor: type, bTensor: type}
                                     targetTensors:@[resTensor]
                                  targetOperations:nil
                               compilationDescriptor:nil];
                pipeline_cache_[cache_key] = (void*)executable;
            }
        }
    }

    // Hot Run: Ensure the Data objects use the SAME SHAPE as the compiled executable
    @autoreleasepool {
        id<MTLCommandQueue> queue = [device newCommandQueue];
        size_t total_elements = output->size() / sizeof(float);
        NSArray<NSNumber *> *execShape = (op_name == "matmul") ? 
            @[@( (int)sqrt(total_elements) ), @( (int)sqrt(total_elements) )] : @[@(total_elements)];

        MPSGraphTensorData *aData = [[MPSGraphTensorData alloc] initWithMTLBuffer:(id<MTLBuffer>)inputs[0]->device_handle() shape:execShape dataType:MPSDataTypeFloat32];
        MPSGraphTensorData *bData = [[MPSGraphTensorData alloc] initWithMTLBuffer:(id<MTLBuffer>)inputs[1]->device_handle() shape:execShape dataType:MPSDataTypeFloat32];
        MPSGraphTensorData *outData = [[MPSGraphTensorData alloc] initWithMTLBuffer:(id<MTLBuffer>)output->device_handle() shape:execShape dataType:MPSDataTypeFloat32];

        [executable runWithMTLCommandQueue:queue inputsArray:@[aData, bData] resultsArray:@[outData] executionDescriptor:nil];
    }
#endif
}

} // namespace lumen