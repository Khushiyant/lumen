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
        // PERFORMANCE: Create queue once to avoid driver overhead
        command_queue_ = [device_ newCommandQueue];
    }

    Buffer* create_buffer(size_t size) override {
        id<MTLBuffer> buf = [device_ newBufferWithLength:size options:MTLResourceStorageModeShared];
        // Use __bridge_retained to prevent ARC from deleting the buffer
        return new Buffer(size, (__bridge_retained void*)buf, [buf contents]);
    }

    void execute(const std::string& op_name, const std::vector<Buffer*>& inputs, Buffer* output) override {
        std::vector<QueuedOp> single_op_queue;
        single_op_queue.push_back({op_name, inputs, output});
        this->sync(single_op_queue);
    }

    void sync(std::vector<QueuedOp>& queue) override {
        if (queue.empty()) return;
        
        @autoreleasepool {
            // 1. Precise Cache Key
            std::stringstream key_builder;
            for (const auto& op : queue) {
                key_builder << op.op_name << "_" << op.output->size() << "|";
            }
            std::string cache_key = key_builder.str();

            MPSGraphExecutable *executable = (__bridge MPSGraphExecutable*)pipeline_cache_[cache_key];
            
            // 2. Setup Graph and Ordered Inputs
            MPSGraph *graph = [[MPSGraph alloc] init];
            std::map<Buffer*, MPSGraphTensor*> buffer_to_tensor;
            NSMutableArray<MPSGraphTensorData*> *inputsArray = [NSMutableArray array];
            NSMutableArray<MPSGraphTensor*> *orderedPlaceholders = [NSMutableArray array];
            NSMutableArray<MPSGraphTensor*> *targetTensors = [NSMutableArray array];
            NSMutableArray<MPSGraphTensorData*> *targetData = [NSMutableArray array];

            for (const auto& op : queue) {
                size_t element_count = op.output->size() / sizeof(float);
                bool is_2d = (op.op_name == "matmul");
                if (!is_2d) {
                    for (auto* b : op.inputs) if (buffer_to_tensor.count(b) && buffer_to_tensor[b].shape.count == 2) is_2d = true;
                }

                NSMutableArray<MPSGraphTensor*> *ins = [NSMutableArray array];
                for (Buffer* buf : op.inputs) {
                    if (buffer_to_tensor.count(buf)) {
                        [ins addObject:buffer_to_tensor[buf]];
                    } else {
                        NSArray<NSNumber *> *shape = is_2d ? @[@((int)sqrt(element_count)), @((int)sqrt(element_count))] : @[@(element_count)];
                        MPSGraphTensor *ph = [graph placeholderWithShape:shape dataType:MPSDataTypeFloat32 name:nil];
                        
                        [ins addObject:ph];
                        [orderedPlaceholders addObject:ph];
                        [inputsArray addObject:[[MPSGraphTensorData alloc] initWithMTLBuffer:(__bridge id<MTLBuffer>)buf->device_handle() shape:shape dataType:MPSDataTypeFloat32]];
                        buffer_to_tensor[buf] = ph;
                    }
                }

                MPSGraphTensor *res = nil;
                if (op.op_name == "add") res = [graph additionWithPrimaryTensor:ins[0] secondaryTensor:ins[1] name:nil];
                else if (op.op_name == "matmul") res = [graph matrixMultiplicationWithPrimaryTensor:ins[0] secondaryTensor:ins[1] name:nil];

                if (res) {
                    buffer_to_tensor[op.output] = res;
                    [targetTensors addObject:res];
                    NSArray<NSNumber *> *outShape = is_2d ? @[@((int)sqrt(element_count)), @((int)sqrt(element_count))] : @[@(element_count)];
                    [targetData addObject:[[MPSGraphTensorData alloc] initWithMTLBuffer:(__bridge id<MTLBuffer>)op.output->device_handle() shape:outShape dataType:MPSDataTypeFloat32]];
                }
            }

            // 3. JIT Compilation (Cache Miss)
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

            // 4. Execution (Ordered Alignment)
            // MPSGraphExecutable.feedTensors returns placeholders in the order they were baked in.
            // Since our logic is deterministic, our inputsArray already matches that order.
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