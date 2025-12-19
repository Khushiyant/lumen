#include "lumen/lumen.hpp"
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#import <Accelerate/Accelerate.h>
#include <cmath>
#include <iostream>
#include <sstream>

namespace lumen {

class MetalBackend : public Backend {
public:
    MetalBackend() {
        device_ = MTLCreateSystemDefaultDevice();
        if (!device_) {
            std::cerr << "[Lumen] Metal Error: No Metal-capable GPU detected." << std::endl;
            return;
        }
        command_queue_ = [device_ newCommandQueue];
    }

    Buffer* create_buffer(const std::vector<size_t>& shape) override {
        size_t total_elements = 1;
        for (auto d : shape) total_elements *= d;
        size_t size = total_elements * sizeof(float);
        
        id<MTLBuffer> buf = [device_ newBufferWithLength:size options:MTLResourceStorageModeShared];
        // FIX: Added nullptr as the 5th argument (Runtime* rt)
        return new Buffer(shape, (__bridge_retained void*)buf, [buf contents], this, nullptr);
    }

    void free_buffer(void* device_ptr) override {
        if (device_ptr) {
            id<MTLBuffer> buf = (__bridge_transfer id<MTLBuffer>)device_ptr;
            buf = nil;
        }
    }

    void execute(const std::string& op_name, const std::vector<Buffer*>& inputs, Buffer* output) override {
        std::vector<QueuedOp> single_op_queue;
        single_op_queue.push_back({op_name, inputs, output});
        this->sync(single_op_queue);
    }

    void sync(std::vector<QueuedOp>& queue) override {
        if (queue.empty()) return;
        
        @autoreleasepool {
            std::stringstream key_builder;
            for (const auto& op : queue) {
                key_builder << op.op_name << "(";
                for (auto* b : op.inputs) {
                    for (auto d : b->shape()) key_builder << d << ",";
                    key_builder << "|";
                }
                key_builder << ")->";
                for (auto d : op.output->shape()) key_builder << d << ",";
                key_builder << " ";
            }
            std::string cache_key = key_builder.str();

            MPSGraphExecutable *executable = (__bridge MPSGraphExecutable*)pipeline_cache_[cache_key];
            
            MPSGraph *graph = [[MPSGraph alloc] init];
            std::map<Buffer*, MPSGraphTensor*> buffer_to_tensor;
            NSMutableArray<MPSGraphTensorData*> *inputsArray = [NSMutableArray array];
            NSMutableArray<MPSGraphTensor*> *orderedPlaceholders = [NSMutableArray array];
            NSMutableArray<MPSGraphTensor*> *targetTensors = [NSMutableArray array];
            NSMutableArray<MPSGraphTensorData*> *targetData = [NSMutableArray array];

            for (const auto& op : queue) {
                NSMutableArray<MPSGraphTensor*> *ins = [NSMutableArray array];
                for (Buffer* buf : op.inputs) {
                    // 1. Convert Lumen shape and strides to NSNumber arrays
                    NSMutableArray<NSNumber *> *ns_shape = [NSMutableArray array];
                    NSMutableArray<NSNumber *> *ns_strides = [NSMutableArray array];
                    for (auto d : buf->shape()) [ns_shape addObject:@(d)];
                    for (auto s : buf->strides()) [ns_strides addObject:@(s * sizeof(float))]; // Metal wants byte strides

                    // 2. Create a "Shaped Type" that includes the strides
                    MPSGraphShapedType *shapedType = [[MPSGraphShapedType alloc] initWithShape:ns_shape 
                                                                                    dataType:MPSDataTypeFloat32];
                    
                    // 3. Create TensorData using the buffer AND the byte strides
                    MPSGraphTensorData *tensorData = [[MPSGraphTensorData alloc] initWithMTLBuffer:(__bridge id<MTLBuffer>)buf->device_handle()
                                                                                            shape:ns_shape
                                                                                        dataType:MPSDataTypeFloat32
                                                                                    rowStrides:ns_strides];
                }

                MPSGraphTensor *res = nil;
                if (op.op_name == "add") res = [graph additionWithPrimaryTensor:ins[0] secondaryTensor:ins[1] name:nil];
                else if (op.op_name == "mul") res = [graph multiplicationWithPrimaryTensor:ins[0] secondaryTensor:ins[1] name:nil];
                else if (op.op_name == "matmul") res = [graph matrixMultiplicationWithPrimaryTensor:ins[0] secondaryTensor:ins[1] name:nil];

                if (res) {
                    buffer_to_tensor[op.output] = res;
                    [targetTensors addObject:res];
                    NSMutableArray<NSNumber *> *out_shape = [NSMutableArray array];
                    for (auto d : op.output->shape()) [out_shape addObject:@(d)];
                    [targetData addObject:[[MPSGraphTensorData alloc] initWithMTLBuffer:(__bridge id<MTLBuffer>)op.output->device_handle() shape:out_shape dataType:MPSDataTypeFloat32]];
                }
            }

            if (!executable) {
                NSMutableDictionary *typeFeeds = [NSMutableDictionary dictionary];
                for (NSUInteger i = 0; i < orderedPlaceholders.count; i++) {
                    MPSGraphTensor *t = orderedPlaceholders[i];
                    MPSGraphTensorData *d = inputsArray[i];
                    typeFeeds[t] = [[MPSGraphShapedType alloc] initWithShape:d.shape dataType:d.dataType];
                }
                executable = [graph compileWithDevice:[MPSGraphDevice deviceWithMTLDevice:device_] 
                                                feeds:typeFeeds targetTensors:targetTensors 
                                     targetOperations:nil compilationDescriptor:nil];
                pipeline_cache_[cache_key] = (__bridge_retained void*)executable;
            }

            [executable runWithMTLCommandQueue:command_queue_ 
                                   inputsArray:inputsArray 
                                  resultsArray:targetData 
                           executionDescriptor:nil];
        }
    }

private:
    id<MTLDevice> device_;
    id<MTLCommandQueue> command_queue_;
    std::map<std::string, void*> pipeline_cache_;
};

std::unique_ptr<Backend> create_metal_backend() {
    return std::make_unique<MetalBackend>();
}

} // namespace lumen