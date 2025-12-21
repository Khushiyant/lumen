#include "lumen/lumen.hpp"
#import <Accelerate/Accelerate.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#include <cmath>
#include <iostream>
#include <map>
#include <set>
#include <sstream>

namespace lumen {

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

  virtual ~MetalBackend() {
    auto blocks = pool_.drain();
    for (auto &block : blocks) {
      // Hands control back to Objective-C ARC to trigger deallocation
      id<MTLBuffer> buf = (__bridge_transfer id<MTLBuffer>)block.second;
      buf = nil;
    }
    std::cout << "[Lumen] Metal Backend: Memory Pool Drained." << std::endl;
  }

  Buffer *create_buffer(const std::vector<size_t> &shape) override {
    size_t total_elements = 1;
    for (auto d : shape)
      total_elements *= d;
    size_t size = total_elements * sizeof(float);

    // Try to acquire from the memory pool
    void *device_ptr = pool_.acquire(size);
    id<MTLBuffer> buf = nil;

    if (device_ptr) {
      // Re-cast the void* back to an Objective-C object
      buf = (__bridge id<MTLBuffer>)device_ptr;
    } else {
      buf = [device_ newBufferWithLength:size
                                 options:MTLResourceStorageModeShared];
      // Bridge Retain: Increments ref count so ARC doesn't dealloc it
      device_ptr = (__bridge_retained void *)buf;
    }

    std::vector<size_t> strides(shape.size());
    size_t s = 1;
    for (int i = (int)shape.size() - 1; i >= 0; --i) {
      strides[i] = s;
      s *= shape[i];
    }

    // Host ptr is contents because storage mode is Shared
    return new Buffer(shape, strides, device_ptr, [buf contents], this, 0);
  }

  void free_buffer(void *device_ptr, size_t size) override {
    if (device_ptr) {
      // Return it to the pool (still retained).
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

      // Helper to create tensor data, handling offsets safely
      // Optimized version in lumen/src/backends/gpu/metal_backend.mm
      // Correct zero-copy implementation in
      // lumen/src/backends/gpu/metal_backend.mm
      auto create_data_safe = [&](Buffer *buf) -> MPSGraphTensorData * {
        id<MTLBuffer> mtl_buf = (__bridge id<MTLBuffer>)buf->device_ptr();

        NSMutableArray<NSNumber *> *ns_shape = [NSMutableArray array];
        for (auto d : buf->shape()) {
          [ns_shape addObject:@(d)];
        }

        // 1. Create a descriptor for the multidimensional array
        MPSNDArrayDescriptor *desc =
            [MPSNDArrayDescriptor descriptorWithDataType:MPSDataTypeFloat32
                                                   shape:ns_shape];

        // 2. Initialize an MPSNDArray using the buffer and the byte offset.
        // This provides a zero-copy "view" into the existing VRAM.
        MPSNDArray *ndarray =
            [[MPSNDArray alloc] initWithBuffer:mtl_buf
                                        offset:buf->offset_bytes()
                                    descriptor:desc];

        // 3. Return the Graph-compatible TensorData initialized from the
        // NDArray
        return [[MPSGraphTensorData alloc] initWithMPSNDArray:ndarray];
      };

      for (const auto &op : queue) {
        for (Buffer *buf : op.inputs) {
          if (!data_map.count(buf))
            data_map[buf] = create_data_safe(buf);
        }
        [resultsArray addObject:create_data_safe(op.output)];
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
};

std::unique_ptr<Backend> create_metal_backend() {
  return std::make_unique<MetalBackend>();
}

} // namespace lumen