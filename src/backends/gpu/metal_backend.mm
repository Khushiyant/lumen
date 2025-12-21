#include "lumen/lumen.hpp"
#include "lumen/op_registry.hpp"
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

class MetalBackend : public Backend, public BackendWithRegistry {
public:
  MetalBackend() {
    device_ = MTLCreateSystemDefaultDevice();
    if (!device_) {
      std::cerr << "[Lumen] Metal Error: No Metal-capable GPU detected."
                << std::endl;
      return;
    }
    command_queue_ = [device_ newCommandQueue];

    register_standard_ops();
    register_backend_ops();
    std::cout << "[Lumen] Metal Backend Initialized - "
              << supported_ops().size() << " ops" << std::endl;
  }

  virtual ~MetalBackend() {
    auto blocks = pool_.drain();
    for (auto &block : blocks) {
      id<MTLBuffer> buf = (__bridge_transfer id<MTLBuffer>)block.second;
      buf = nil;
    }
  }

  Buffer *create_buffer(const std::vector<size_t> &shape) override {
    size_t total_elements = 1;
    for (auto d : shape)
      total_elements *= d;
    size_t size = total_elements * sizeof(float);
    void *device_ptr = pool_.acquire(size);
    id<MTLBuffer> buf = nil;
    if (device_ptr) {
      buf = (__bridge id<MTLBuffer>)device_ptr;
    } else {
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

  void free_buffer(void *device_ptr, size_t size) override {
    if (device_ptr)
      pool_.release(device_ptr, size);
  }

  void execute(const std::string &op_name,
               const std::vector<std::shared_ptr<Buffer>> &inputs,
               std::shared_ptr<Buffer> output) override {
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
              for (const auto &buf_ptr : op.inputs) {
                Buffer *buf = buf_ptr.get();
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
            } else {
              [ins addObject:current_res];
            }

            if (sub_op == "add") {
              current_res = [graph additionWithPrimaryTensor:ins[0]
                                             secondaryTensor:ins[1]
                                                        name:nil];
            } else if (sub_op == "mul") {
              current_res = [graph multiplicationWithPrimaryTensor:ins[0]
                                                   secondaryTensor:ins[1]
                                                              name:nil];
            } else if (sub_op == "matmul") {
              current_res = [graph matrixMultiplicationWithPrimaryTensor:ins[0]
                                                         secondaryTensor:ins[1]
                                                                    name:nil];
            } else if (sub_op == "relu") {
              current_res = [graph reLUWithTensor:ins[0] name:nil];
            } else if (sub_op == "sigmoid") {
              current_res = [graph sigmoidWithTensor:ins[0] name:nil];
            } else if (sub_op == "tanh") {
              current_res = [graph tanhWithTensor:ins[0] name:nil];
            } else if (sub_op == "softmax") {
              current_res = [graph softMaxWithTensor:ins[0] axis:-1 name:nil];
            } else if (sub_op == "flatten") {
              current_res = [graph reshapeTensor:ins[0]
                                       withShape:@[
                                         @(op.output.get()->shape()[0]),
                                         @(op.output.get()->shape()[1])
                                       ]
                                            name:nil];
            } else if (sub_op == "reshape") {
              auto shape_ints = op.attrs.get_int_array("shape");
              NSMutableArray<NSNumber *> *ns_shape = [NSMutableArray array];
              for (auto d : shape_ints)
                [ns_shape addObject:@(d)];
              current_res = [graph reshapeTensor:ins[0]
                                       withShape:ns_shape
                                            name:nil];
            } else if (sub_op == "layer_norm") {
              MPSGraphTensor *mean = [graph meanOfTensor:ins[0]
                                                    axes:@[ @(-1) ]
                                                    name:nil];
              MPSGraphTensor *centered =
                  [graph subtractionWithPrimaryTensor:ins[0]
                                      secondaryTensor:mean
                                                 name:nil];
              MPSGraphTensor *squared = [graph squareWithTensor:centered
                                                           name:nil];
              MPSGraphTensor *variance = [graph meanOfTensor:squared
                                                        axes:@[ @(-1) ]
                                                        name:nil];
              float eps = op.attrs.get_float("epsilon", 1e-5f);
              MPSGraphTensor *eps_tensor =
                  [graph constantWithScalar:eps dataType:MPSDataTypeFloat32];
              MPSGraphTensor *variance_plus_eps =
                  [graph additionWithPrimaryTensor:variance
                                   secondaryTensor:eps_tensor
                                              name:nil];
              MPSGraphTensor *std_dev =
                  [graph squareRootWithTensor:variance_plus_eps name:nil];
              current_res = [graph divisionWithPrimaryTensor:centered
                                             secondaryTensor:std_dev
                                                        name:nil];
            } else if (sub_op == "conv2d") {
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
            } else if (sub_op == "maxpool2d") {
              auto kernel = op.attrs.get_int_array("kernel_size");
              auto strides = op.attrs.get_int_array("stride");
              int kh = kernel.empty() ? 2 : kernel[0];
              int kw = kernel.size() > 1 ? kernel[1] : kh;
              int sh = strides.empty() ? kh : strides[0];
              int sw = strides.size() > 1 ? strides[1] : sh;
              MPSGraphPooling2DOpDescriptor *desc =
                  [MPSGraphPooling2DOpDescriptor
                      descriptorWithKernelWidth:kw
                                   kernelHeight:kh
                                      strideInX:sw
                                      strideInY:sh
                                   paddingStyle:MPSGraphPaddingStyleExplicit
                                     dataLayout:
                                         MPSGraphTensorNamedDataLayoutNCHW];
              current_res = [graph maxPooling2DWithSourceTensor:ins[0]
                                                     descriptor:desc
                                                           name:nil];
            } else if (sub_op == "avgpool2d") {
              auto kernel = op.attrs.get_int_array("kernel_size");
              auto strides = op.attrs.get_int_array("stride");
              int kh = kernel.empty() ? 2 : kernel[0];
              int kw = kernel.size() > 1 ? kernel[1] : kh;
              int sh = strides.empty() ? kh : strides[0];
              int sw = strides.size() > 1 ? strides[1] : sh;
              MPSGraphPooling2DOpDescriptor *desc =
                  [MPSGraphPooling2DOpDescriptor
                      descriptorWithKernelWidth:kw
                                   kernelHeight:kh
                                      strideInX:sw
                                      strideInY:sh
                                   paddingStyle:MPSGraphPaddingStyleExplicit
                                     dataLayout:
                                         MPSGraphTensorNamedDataLayoutNCHW];
              current_res = [graph avgPooling2DWithSourceTensor:ins[0]
                                                     descriptor:desc
                                                           name:nil];
            } else if (sub_op == "reduce_mean") {
              current_res = [graph meanOfTensor:ins[0]
                                           axes:@[ @(-1) ]
                                           name:nil];
            } else if (sub_op == "reduce_sum") {
              current_res = [graph reductionSumWithTensor:ins[0]
                                                     axes:@[ @(-1) ]
                                                     name:nil];
            } else if (sub_op == "batchnorm") {
              // Simplified batch norm
              MPSGraphTensor *mean = [graph meanOfTensor:ins[0]
                                                    axes:@[ @0, @2, @3 ]
                                                    name:nil];
              MPSGraphTensor *centered =
                  [graph subtractionWithPrimaryTensor:ins[0]
                                      secondaryTensor:mean
                                                 name:nil];
              MPSGraphTensor *squared = [graph squareWithTensor:centered
                                                           name:nil];
              MPSGraphTensor *variance = [graph meanOfTensor:squared
                                                        axes:@[ @0, @2, @3 ]
                                                        name:nil];
              float eps = op.attrs.get_float("epsilon", 1e-5f);
              MPSGraphTensor *eps_tensor =
                  [graph constantWithScalar:eps dataType:MPSDataTypeFloat32];
              MPSGraphTensor *variance_plus_eps =
                  [graph additionWithPrimaryTensor:variance
                                   secondaryTensor:eps_tensor
                                              name:nil];
              MPSGraphTensor *std_dev =
                  [graph squareRootWithTensor:variance_plus_eps name:nil];
              current_res = [graph divisionWithPrimaryTensor:centered
                                             secondaryTensor:std_dev
                                                        name:nil];
            }
          }
          if (current_res) {
            buffer_to_tensor[op.output.get()] = current_res;
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

      auto create_data_safe =
          [&](const std::shared_ptr<Buffer> &buf) -> MPSGraphTensorData * {
        id<MTLBuffer> mtl_buf = (__bridge id<MTLBuffer>)buf->device_ptr();
        NSMutableArray<NSNumber *> *ns_shape = [NSMutableArray array];
        for (auto d : buf->shape())
          [ns_shape addObject:@(d)];
        MPSNDArrayDescriptor *desc =
            [MPSNDArrayDescriptor descriptorWithDataType:MPSDataTypeFloat32
                                                   shape:ns_shape];
        MPSNDArray *ndarray =
            [[MPSNDArray alloc] initWithBuffer:mtl_buf
                                        offset:buf->offset_bytes()
                                    descriptor:desc];
        return [[MPSGraphTensorData alloc] initWithMPSNDArray:ndarray];
      };

      for (const auto &op : queue) {
        for (const auto &buf_sh : op.inputs) {
          // Buffer *buf = buf_sh.get(); 
          if (!data_map.count(buf_sh.get()))
            data_map[buf_sh.get()] = create_data_safe(buf_sh);
        }
        [resultsArray addObject:create_data_safe(op.output)];
      }

      std::set<Buffer *> seen;
      for (const auto &op : queue) {
        for (const auto &buf_sh : op.inputs) {
          if (seen.find(buf_sh.get()) == seen.end()) {
            [inputsArray addObject:data_map[buf_sh.get()]];
            seen.insert(buf_sh.get());
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

protected:
  void register_backend_ops() override {
    // Metal uses MPSGraph for all operations, so the default graph-building
    // in sync() handles everything. No need to override individual ops.
    // All 15 operations are supported through the graph compilation.
  }

private:
  id<MTLDevice> device_;
  id<MTLCommandQueue> command_queue_;
  std::map<std::string, CachedPipeline> pipeline_cache_;

  std::string build_cache_key(const std::vector<QueuedOp> &queue) {
    std::stringstream ss;
    for (const auto &op : queue) {
      ss << op.op_name << "(";
      for (const auto &b : op.inputs) {
        Buffer *raw_buf = b.get();
        for (auto d : raw_buf->shape())
          ss << d << ",";
        ss << "|";
      }
      ss << ")->";
      for (auto d : op.output.get()->shape())
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