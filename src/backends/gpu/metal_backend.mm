#include "lumen/lumen.hpp"
#import <Accelerate/Accelerate.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#include <cmath>
#include <iostream>
#include <map>
#include <set>
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

  std::shared_ptr<Event> sync(std::vector<QueuedOp> &queue) override {
    if (queue.empty())
      return nullptr;

    @autoreleasepool {
      std::string cache_key = build_cache_key(queue);
      if (pipeline_cache_.find(cache_key) == pipeline_cache_.end()) {
        MPSGraph *graph = [[MPSGraph alloc] init];
        std::map<Buffer *, MPSGraphTensor *> buffer_to_tensor;
        NSMutableArray<MPSGraphTensor *> *orderedPlaceholders =
            [NSMutableArray array];
        NSMutableArray<MPSGraphTensor *> *targetTensors =
            [NSMutableArray array];

        for (const auto &op : queue) {
          // Parse "matmul_relu" -> ["matmul", "relu"]
          std::vector<std::string> sub_ops;
          std::string segment;
          std::stringstream ss(op.op_name);
          while (std::getline(ss, segment, '_'))
            sub_ops.push_back(segment);

          MPSGraphTensor *current_res = nil;
          for (size_t i = 0; i < sub_ops.size(); ++i) {
            const std::string &sub_op = sub_ops[i];
            NSMutableArray<MPSGraphTensor *> *ins = [NSMutableArray array];

            if (i == 0) {
              for (Buffer *buf : op.inputs) {
                if (buffer_to_tensor.count(buf))
                  [ins addObject:buffer_to_tensor[buf]];
                else {
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
            } else {
              [ins addObject:current_res];
            }

            if (sub_op == "add")
              current_res = [graph additionWithPrimaryTensor:ins[0]
                                             secondaryTensor:ins[1]
                                                        name:nil];
            else if (sub_op == "mul")
              current_res = [graph multiplicationWithPrimaryTensor:ins[0]
                                                   secondaryTensor:ins[1]
                                                              name:nil];
            else if (sub_op == "matmul")
              current_res = [graph matrixMultiplicationWithPrimaryTensor:ins[0]
                                                         secondaryTensor:ins[1]
                                                                    name:nil];
            else if (sub_op == "relu")
              current_res = [graph reLUWithTensor:ins[0] name:nil];
            else if (sub_op == "softmax")
              current_res = [graph softMaxWithTensor:ins[0] axis:-1 name:nil];
            else if (sub_op == "conv2d") {
              auto strides = op.attrs.get_int_array("stride");
              auto padding = op.attrs.get_int_array("padding");
              MPSGraphConvolution2DOpDescriptor *desc =
                  [MPSGraphConvolution2DOpDescriptor
                      descriptorWithStrideInX:(strides.size() > 1 ? strides[1]
                                                                  : 1)
                                    strideInY:(strides.empty() ? 1 : strides[0])
                                    dilationRateInX:1
                              dilationRateInY:1
                                       groups:1
                                 paddingStyle:MPSGraphPaddingStyleExplicit
                                   dataLayout:MPSGraphTensorNamedDataLayoutNCHW
                                weightsLayout:
                                    MPSGraphTensorNamedDataLayoutOIHW];
              [desc setExplicitPaddingWithPaddingLeft:(padding.size() > 1
                                                           ? padding[1]
                                                           : 0)
                                         paddingRight:(padding.size() > 1
                                                           ? padding[1]
                                                           : 0)
                                           paddingTop:(padding.empty()
                                                           ? 0
                                                           : padding[0])
                                           paddingBottom:(padding.empty()
                                                              ? 0
                                                              : padding[0])];
              current_res = [graph convolution2DWithSourceTensor:ins[0]
                                                   weightsTensor:ins[1]
                                                      descriptor:desc
                                                            name:nil];
            } else if (sub_op == "global_average_pool") {
              current_res = [graph meanOfTensor:ins[0]
                                           axes:@[ @2, @3 ]
                                           name:nil];
            }
          }
          if (current_res) {
            buffer_to_tensor[op.output] = current_res;
            [targetTensors addObject:current_res];
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
      NSMutableArray<MPSGraphTensorData *> *inputsArray =
          [NSMutableArray array];
      NSMutableArray<MPSGraphTensorData *> *resultsArray =
          [NSMutableArray array];
      std::map<Buffer *, MPSGraphTensorData *> data_map;

      for (const auto &op : queue) {
        for (Buffer *buf : op.inputs) {
          if (!data_map.count(buf))
            data_map[buf] = create_tensor_data(buf);
        }
        [resultsArray addObject:create_tensor_data(op.output)];
      }
      std::set<Buffer *> seen;
      for (const auto &op : queue) {
        for (Buffer *buf : op.inputs) {
          if (seen.find(buf) == seen.end()) {
            [inputsArray addObject:data_map[buf]];
            seen.insert(buf);
          }
        }
      }
      [pipeline.executable runWithMTLCommandQueue:command_queue_
                                      inputsArray:inputsArray
                                     resultsArray:resultsArray
                              executionDescriptor:nil];
      id<MTLCommandBuffer> commandBuffer = [command_queue_ commandBuffer];
      [commandBuffer commit];
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

  // Change this method to correctly handle offsets
  // lumen/src/backends/gpu/metal_backend.mm

  // lumen/src/backends/gpu/metal_backend.mm

  MPSGraphTensorData *create_tensor_data(Buffer *buf) {
    NSMutableArray<NSNumber *> *ns_shape = [NSMutableArray array];
    for (auto d : buf->shape())
      [ns_shape addObject:@(d)];

    // FIXED: Correctly cast the device_ptr back to an id<MTLBuffer> object
    // without performing arithmetic on the object's memory address.
    id<MTLBuffer> mtl_buf = (__bridge id<MTLBuffer>)buf->device_handle_base();

    return [[MPSGraphTensorData alloc] initWithMTLBuffer:mtl_buf
                                                   shape:ns_shape
                                                dataType:MPSDataTypeFloat32];
  }
};

std::unique_ptr<Backend> create_metal_backend() {
  return std::make_unique<MetalBackend>();
}

} // namespace lumen