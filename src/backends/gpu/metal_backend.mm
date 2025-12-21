#include "lumen/lumen.hpp"
#import <Accelerate/Accelerate.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#include <cmath>
#include <iostream>
#include <map>
#include <sstream>

namespace lumen {

// Event implementation for Metal to track command buffer status asynchronously
class MetalEvent : public Event {
public:
  MetalEvent(id<MTLCommandBuffer> cb) : cb_(cb) {}
  void wait() override { [cb_ waitUntilCompleted]; }
  bool is_completed() override {
    return cb_.status >= MTLCommandBufferStatusCompleted;
  }

private:
  id<MTLCommandBuffer> cb_;
};

struct CachedPipeline {
  MPSGraphExecutable *executable;
  NSMutableArray<MPSGraphTensor *> *orderedPlaceholders;
};

class MetalBackend : public Backend {
public:
  MetalBackend() {
    device_ = MTLCreateSystemDefaultDevice();
    if (!device_) {
      std::cerr << "[Lumen] Metal Error: No Metal-capable GPU detected."
                << std::endl;
      return;
    }
    command_queue_ = [device_ newCommandQueue];
    std::cout << "[Lumen] Metal Backend Initialized with Memory Pooling"
              << std::endl;
  }

  Buffer *create_buffer(const std::vector<size_t> &shape) override {
    size_t total_elements = 1;
    for (auto d : shape)
      total_elements *= d;
    size_t size = total_elements * sizeof(float);

    // Try to acquire from the memory pool first
    void *device_ptr = pool_.acquire(size);
    id<MTLBuffer> buf = nil;

    if (device_ptr) {
      buf = (__bridge id<MTLBuffer>)device_ptr;
    } else {
      // Allocate new buffer if pool is empty
      buf = [device_ newBufferWithLength:size
                                 options:MTLResourceStorageModeShared];
      device_ptr = (__bridge_retained void *)buf;
    }

    std::vector<size_t> strides(shape.size());
    size_t s = 1;
    for (int i = (int)shape.size() - 1; i >= 0; --i) {
      strides[i] = s;
      s *= shape[i];
    }

    return new Buffer(shape, strides, device_ptr, [buf contents], this, 0);
  }

  // FIXED: Updated signature to match the virtual method in Backend base class
  void free_buffer(void *device_ptr, size_t size) override {
    if (device_ptr) {
      // Return the block to the pool for reuse instead of immediate release
      pool_.release(device_ptr, size);
    }
  }

  void execute(const std::string &op_name, const std::vector<Buffer *> &inputs,
               Buffer *output) override {
    std::vector<QueuedOp> single_op_queue = {{op_name, inputs, output}};
    this->sync(single_op_queue);
  }

  // FIXED: Signature changed to return std::shared_ptr<Event>
  std::shared_ptr<Event> sync(std::vector<QueuedOp> &queue) override {
    if (queue.empty())
      return nullptr;

    @autoreleasepool {
      std::string cache_key = build_cache_key(queue);
      NSMutableArray<MPSGraphTensorData *> *inputsArray =
          [NSMutableArray array];
      NSMutableArray<MPSGraphTensorData *> *targetData = [NSMutableArray array];

      if (pipeline_cache_.find(cache_key) == pipeline_cache_.end()) {
        MPSGraph *graph = [[MPSGraph alloc] init];
        std::map<Buffer *, MPSGraphTensor *> buffer_to_tensor;
        NSMutableArray<MPSGraphTensor *> *orderedPlaceholders =
            [NSMutableArray array];
        NSMutableArray<MPSGraphTensor *> *targetTensors =
            [NSMutableArray array];

        for (const auto &op : queue) {
          NSMutableArray<MPSGraphTensor *> *ins = [NSMutableArray array];
          for (Buffer *buf : op.inputs) {
            if (buffer_to_tensor.count(buf)) {
              [ins addObject:buffer_to_tensor[buf]];
            } else {
              NSMutableArray<NSNumber *> *ns_shape = [NSMutableArray array];
              for (auto d : buf->shape())
                [ns_shape addObject:@(d)];

              MPSGraphTensor *ph =
                  [graph placeholderWithShape:ns_shape
                                     dataType:MPSDataTypeFloat32
                                         name:nil];
              [ins addObject:ph];
              [orderedPlaceholders addObject:ph];
              buffer_to_tensor[buf] = ph;
            }
          }

          MPSGraphTensor *res = nil;
          if (op.op_name == "add")
            res = [graph additionWithPrimaryTensor:ins[0]
                                   secondaryTensor:ins[1]
                                              name:nil];
          else if (op.op_name == "mul")
            res = [graph multiplicationWithPrimaryTensor:ins[0]
                                         secondaryTensor:ins[1]
                                                    name:nil];
          else if (op.op_name == "matmul")
            res = [graph matrixMultiplicationWithPrimaryTensor:ins[0]
                                               secondaryTensor:ins[1]
                                                          name:nil];
          else if (op.op_name == "softmax") {
            // MPSGraph uses the 'axis' parameter. Last dimension is -1.
            res = [graph softMaxWithTensor:ins[0] axis:-1 name:nil];
          } else if (op.op_name == "conv2d") {
            auto strides = op.attrs.get_int_array("stride");
            auto padding = op.attrs.get_int_array("padding");

            // FIX: Selector order is X then Y
            MPSGraphConvolution2DOpDescriptor *desc =
                [MPSGraphConvolution2DOpDescriptor
                    descriptorWithStrideInX:(strides.size() > 1 ? strides[1]
                                                                : 1)
                                  strideInY:(strides.empty()
                                                 ? 1
                                                 : strides[0])dilationRateInX:1
                            dilationRateInY:1
                                     groups:1
                               paddingStyle:MPSGraphPaddingStyleExplicit
                                 dataLayout:MPSGraphTensorNamedDataLayoutNCHW
                              weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];

            [desc
                setExplicitPaddingWithPaddingLeft:(padding.size() > 1
                                                       ? padding[1]
                                                       : 0)
                                     paddingRight:(padding.size() > 1
                                                       ? padding[1]
                                                       : 0)
                                       paddingTop:(padding.empty() ? 0
                                                                   : padding[0])
                                       paddingBottom:(padding.empty()
                                                          ? 0
                                                          : padding[0])];

            res = [graph convolution2DWithSourceTensor:ins[0]
                                         weightsTensor:ins[1]
                                            descriptor:desc
                                                  name:nil];
          } else if (op.op_name == "global_average_pool") {
            res = [graph meanOfTensor:ins[0] axes:@[ @2, @3 ] name:nil];
          }
          if (res) {
            buffer_to_tensor[op.output] = res;
            [targetTensors addObject:res];
          }
        }

        NSMutableDictionary *typeFeeds = [NSMutableDictionary dictionary];
        for (MPSGraphTensor *ph in orderedPlaceholders) {
          typeFeeds[ph] =
              [[MPSGraphShapedType alloc] initWithShape:ph.shape
                                               dataType:ph.dataType];
        }

        MPSGraphExecutable *exe = [graph
                compileWithDevice:[MPSGraphDevice deviceWithMTLDevice:device_]
                            feeds:typeFeeds
                    targetTensors:targetTensors
                 targetOperations:nil
            compilationDescriptor:nil];

        pipeline_cache_[cache_key] = {exe, orderedPlaceholders};
      }

      const auto &pipeline = pipeline_cache_[cache_key];
      std::map<Buffer *, MPSGraphTensorData *> current_data_map;

      for (const auto &op : queue) {
        for (Buffer *buf : op.inputs) {
          if (current_data_map.find(buf) == current_data_map.end()) {
            current_data_map[buf] = create_tensor_data(buf);
          }
        }
        [targetData addObject:create_tensor_data(op.output)];
      }

      std::map<Buffer *, bool> seen;
      for (const auto &op : queue) {
        for (Buffer *buf : op.inputs) {
          if (!seen[buf]) {
            [inputsArray addObject:current_data_map[buf]];
            seen[buf] = true;
          }
        }
      }

      [pipeline.executable runWithMTLCommandQueue:command_queue_
                                      inputsArray:inputsArray
                                     resultsArray:targetData
                              executionDescriptor:nil];

      id<MTLCommandBuffer> commandBuffer = [command_queue_ commandBuffer];
      [commandBuffer commit];

      // Returns the event handler instead of blocking the CPU with
      // waitUntilCompleted
      return std::make_shared<MetalEvent>(commandBuffer);
    }
  }

private:
  id<MTLDevice> device_;
  id<MTLCommandQueue> command_queue_;
  std::map<std::string, CachedPipeline> pipeline_cache_;

  std::string build_cache_key(const std::vector<QueuedOp> &queue) {
    std::stringstream ss;
    for (const auto &op : queue) {
      ss << op.op_name << "(";
      for (auto *b : op.inputs) {
        for (auto d : b->shape())
          ss << d << ",";
        ss << "|";
      }
      ss << ")->";
      for (auto d : op.output->shape())
        ss << d << ",";
      ss << " ";
    }
    return ss.str();
  }

  MPSGraphTensorData *create_tensor_data(Buffer *buf) {
    NSMutableArray<NSNumber *> *ns_shape = [NSMutableArray array];
    for (auto d : buf->shape())
      [ns_shape addObject:@(d)];
    return [[MPSGraphTensorData alloc]
        initWithMTLBuffer:(__bridge id<MTLBuffer>)buf->device_handle()
                    shape:ns_shape
                 dataType:MPSDataTypeFloat32];
  }
};

std::unique_ptr<Backend> create_metal_backend() {
  return std::make_unique<MetalBackend>();
}

} // namespace lumen