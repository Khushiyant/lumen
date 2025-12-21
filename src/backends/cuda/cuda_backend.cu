#include "lumen/lumen.hpp"
#include "lumen/op_registry.hpp"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace lumen {

__global__ void add_kernel(const float *a, const float *b, float *c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    c[i] = a[i] + b[i];
}

__global__ void mul_kernel(const float *a, const float *b, float *c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    c[i] = a[i] * b[i];
}

__global__ void relu_kernel(const float *input, float *output, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    output[i] = fmaxf(0.0f, input[i]);
}

__global__ void sigmoid_kernel(const float *input, float *output, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    output[i] = 1.0f / (1.0f + expf(-input[i]));
}

__global__ void tanh_kernel(const float *input, float *output, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    output[i] = tanhf(input[i]);
}

__global__ void softmax_kernel(const float *input, float *output,
                               int outer_size, int inner_size) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < outer_size) {
    const float *in_row = input + (row * inner_size);
    float *out_row = output + (row * inner_size);
    float max_val = -1e38f;
    for (int i = 0; i < inner_size; ++i)
      max_val = fmaxf(max_val, in_row[i]);
    float sum = 0.0f;
    for (int i = 0; i < inner_size; ++i) {
      float e = expf(in_row[i] - max_val);
      out_row[i] = e;
      sum += e;
    }
    for (int i = 0; i < inner_size; ++i)
      out_row[i] /= sum;
  }
}

__global__ void layer_norm_kernel(const float *input, float *output,
                                  int outer_size, int inner_size, float eps) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < outer_size) {
    const float *in_row = input + (row * inner_size);
    float *out_row = output + (row * inner_size);
    float mean = 0.0f;
    for (int i = 0; i < inner_size; ++i)
      mean += in_row[i];
    mean /= inner_size;
    float variance = 0.0f;
    for (int i = 0; i < inner_size; ++i) {
      float diff = in_row[i] - mean;
      variance += diff * diff;
    }
    variance /= inner_size;
    float inv_std = rsqrtf(variance + eps);
    for (int i = 0; i < inner_size; ++i)
      out_row[i] = (in_row[i] - mean) * inv_std;
  }
}

__global__ void conv2d_kernel(const float *input, const float *weight,
                              float *output, int N, int Ci, int H, int W,
                              int Co, int Kh, int Kw, int Ho, int Wo, int sh,
                              int sw, int ph, int pw) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = N * Co * Ho * Wo;
  if (idx < total) {
    int w_idx = idx % Wo;
    int h_idx = (idx / Wo) % Ho;
    int c_idx = (idx / (Wo * Ho)) % Co;
    int n_idx = idx / (Wo * Ho * Co);
    float sum = 0;
    for (int ic = 0; ic < Ci; ++ic) {
      for (int kh = 0; kh < Kh; ++kh) {
        for (int kw = 0; kw < Kw; ++kw) {
          int ih = h_idx * sh - ph + kh;
          int iw = w_idx * sw - pw + kw;
          if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
            sum += input[((n_idx * Ci + ic) * H + ih) * W + iw] *
                   weight[((c_idx * Ci + ic) * Kh + kh) * Kw + kw];
          }
        }
      }
    }
    output[idx] = sum;
  }
}

class CUDAEvent : public Event {
public:
  void wait() override { cudaDeviceSynchronize(); }
  bool is_completed() override { return cudaStreamQuery(0) == cudaSuccess; }
};

class CUDABackend : public Backend, public BackendWithRegistry {
public:
  CUDABackend() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess)
      throw std::runtime_error("CUDA Driver Error: " +
                               std::string(cudaGetErrorString(err)));
    if (deviceCount == 0)
      throw std::runtime_error("No CUDA-capable devices found.");
    cudaSetDevice(0);
    cublasStatus_t stat = cublasCreate(&cublas_handle_);
    if (stat != CUBLAS_STATUS_SUCCESS)
      throw std::runtime_error("Failed to initialize cuBLAS.");

    register_standard_ops();
    register_backend_ops();
    std::cout << "[Lumen] CUDA Backend Initialized - " << supported_ops().size()
              << " ops" << std::endl;
  }

  ~CUDABackend() {
    if (cublas_handle_)
      cublasDestroy(cublas_handle_);
    auto blocks = pool_.drain();
    for (auto &block : blocks)
      cudaFree(block.second);
  }

  Buffer *create_buffer(const std::vector<size_t> &shape) override {
    size_t total_elements = 1;
    for (auto d : shape)
      total_elements *= d;
    size_t size = total_elements * sizeof(float);
    void *ptr = pool_.acquire(size);
    if (!ptr)
      cudaMallocManaged(&ptr, size);
    std::vector<size_t> strides(shape.size());
    size_t s = 1;
    for (int i = (int)shape.size() - 1; i >= 0; --i) {
      strides[i] = s;
      s *= shape[i];
    }
    return new Buffer(shape, strides, ptr, ptr, this, 0);
  }

  void free_buffer(void *device_ptr, size_t size) override {
    if (device_ptr)
      pool_.release(device_ptr, size);
  }

  void execute(const std::string &op_name,
               const std::vector<std::shared_ptr<Buffer>> &inputs,
               std::shared_ptr<Buffer> output) override {
    std::vector<QueuedOp> q = {{op_name, inputs, output}};
    sync(q);
  }

  float *get_device_ptr(Buffer *b) {
    return (float *)((char *)b->device_ptr() + b->offset_bytes());
  }
void copy_h2d(void *host_ptr, void *device_ptr, size_t size) override {
    cudaMemcpy(device_ptr, host_ptr, size, cudaMemcpyHostToDevice);
  }

  void copy_d2h(void *device_ptr, void *host_ptr, size_t size) override {
    cudaMemcpy(host_ptr, device_ptr, size, cudaMemcpyDeviceToHost);
  }
  std::shared_ptr<Event> sync(std::vector<QueuedOp> &queue) override {
    if (queue.empty())
      return nullptr;
    for (const auto &op : queue) {
      std::vector<std::string> sub_ops;
      std::string segment;
      std::stringstream ss(op.op_name);
      while (std::getline(ss, segment, '_'))
        sub_ops.push_back(segment);

      for (size_t i = 0; i < sub_ops.size(); ++i) {
        OpContext ctx;
        if (i == 0) {
          for (auto &in_sh : op.inputs)
            ctx.inputs.push_back(in_sh.get());
        } else {
          ctx.inputs = {op.output.get()};
        }
        ctx.output = op.output.get();
        ctx.attrs = op.attrs;

        try {
          execute_op(sub_ops[i], ctx);
        } catch (const std::exception &e) {
          std::cerr << "[CUDA] Op '" << sub_ops[i] << "' failed: " << e.what()
                    << std::endl;
          throw;
        }
      }
    }
    return std::make_shared<CUDAEvent>();
  }

protected:
  void register_backend_ops() override {
    registry_.register_op("add", [this](const OpContext &ctx) {
      float *d_a = get_device_ptr(ctx.inputs[0]);
      float *d_b = get_device_ptr(ctx.inputs[1]);
      float *d_out = get_device_ptr(ctx.output);
      size_t n = ctx.output_size();
      int threads = 256, blocks = (n + threads - 1) / threads;
      add_kernel<<<blocks, threads>>>(d_a, d_b, d_out, (int)n);
    });

    registry_.register_op("mul", [this](const OpContext &ctx) {
      float *d_a = get_device_ptr(ctx.inputs[0]);
      float *d_b = get_device_ptr(ctx.inputs[1]);
      float *d_out = get_device_ptr(ctx.output);
      size_t n = ctx.output_size();
      int threads = 256, blocks = (n + threads - 1) / threads;
      mul_kernel<<<blocks, threads>>>(d_a, d_b, d_out, (int)n);
    });

    registry_.register_op("relu", [this](const OpContext &ctx) {
      float *d_in = get_device_ptr(ctx.inputs[0]);
      float *d_out = get_device_ptr(ctx.output);
      size_t n = ctx.output_size();
      int threads = 256, blocks = (n + threads - 1) / threads;
      relu_kernel<<<blocks, threads>>>(d_in, d_out, (int)n);
    });

    registry_.register_op("sigmoid", [this](const OpContext &ctx) {
      float *d_in = get_device_ptr(ctx.inputs[0]);
      float *d_out = get_device_ptr(ctx.output);
      size_t n = ctx.output_size();
      int threads = 256, blocks = (n + threads - 1) / threads;
      sigmoid_kernel<<<blocks, threads>>>(d_in, d_out, (int)n);
    });

    registry_.register_op("tanh", [this](const OpContext &ctx) {
      float *d_in = get_device_ptr(ctx.inputs[0]);
      float *d_out = get_device_ptr(ctx.output);
      size_t n = ctx.output_size();
      int threads = 256, blocks = (n + threads - 1) / threads;
      tanh_kernel<<<blocks, threads>>>(d_in, d_out, (int)n);
    });

    registry_.register_op("matmul", [this](const OpContext &ctx) {
      float *d_a = get_device_ptr(ctx.inputs[0]);
      float *d_b = get_device_ptr(ctx.inputs[1]);
      float *d_out = get_device_ptr(ctx.output);
      int M = (int)ctx.inputs[0]->shape()[0];
      int K = (int)ctx.inputs[0]->shape()[1];
      int N = (int)ctx.inputs[1]->shape()[1];
      const float alpha = 1.0f, beta = 0.0f;
      cublasSgemm(cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                  d_b, N, d_a, K, &beta, d_out, N);
    });

    registry_.register_op("softmax", [this](const OpContext &ctx) {
      float *d_in = get_device_ptr(ctx.inputs[0]);
      float *d_out = get_device_ptr(ctx.output);
      int inner = (int)ctx.inputs[0]->shape().back();
      int outer = (int)(ctx.output_size() / inner);
      int threads = 256, blocks = (outer + threads - 1) / threads;
      softmax_kernel<<<blocks, threads>>>(d_in, d_out, outer, inner);
    });

    registry_.register_op("layer_norm", [this](const OpContext &ctx) {
      float *d_in = get_device_ptr(ctx.inputs[0]);
      float *d_out = get_device_ptr(ctx.output);
      float eps = ctx.attrs.get_float("epsilon", 1e-5f);
      int inner = (int)ctx.inputs[0]->shape().back();
      int outer = (int)(ctx.output_size() / inner);
      int threads = 256, blocks = (outer + threads - 1) / threads;
      layer_norm_kernel<<<blocks, threads>>>(d_in, d_out, outer, inner, eps);
    });

    registry_.register_op("conv2d", [this](const OpContext &ctx) {
      float *d_in = get_device_ptr(ctx.inputs[0]);
      float *d_w = get_device_ptr(ctx.inputs[1]);
      float *d_out = get_device_ptr(ctx.output);
      auto &in_s = ctx.inputs[0]->shape();
      auto &w_s = ctx.inputs[1]->shape();
      auto &out_s = ctx.output->shape();
      auto stride = ctx.attrs.get_int_array("stride");
      auto padding = ctx.attrs.get_int_array("padding");
      int sh = stride.empty() ? 1 : stride[0],
          sw = stride.size() > 1 ? stride[1] : sh;
      int ph = padding.empty() ? 0 : padding[0],
          pw = padding.size() > 1 ? padding[1] : ph;
      int total = out_s[0] * out_s[1] * out_s[2] * out_s[3];
      int threads = 256, blocks = (total + threads - 1) / threads;
      conv2d_kernel<<<blocks, threads>>>(
          d_in, d_w, d_out, (int)in_s[0], (int)in_s[1], (int)in_s[2],
          (int)in_s[3], (int)out_s[1], (int)w_s[2], (int)w_s[3], (int)out_s[2],
          (int)out_s[3], sh, sw, ph, pw);
    });

    registry_.register_op("flatten", [this](const OpContext &ctx) {
      cudaMemcpyAsync(get_device_ptr(ctx.output), get_device_ptr(ctx.inputs[0]),
                      ctx.output_size() * sizeof(float),
                      cudaMemcpyDeviceToDevice);
    });

    registry_.register_op("reshape", [this](const OpContext &ctx) {
      cudaMemcpyAsync(get_device_ptr(ctx.output), get_device_ptr(ctx.inputs[0]),
                      ctx.output_size() * sizeof(float),
                      cudaMemcpyDeviceToDevice);
    });
  }

private:
  cublasHandle_t cublas_handle_ = nullptr;
};

std::unique_ptr<Backend> create_cuda_backend() {
  return std::make_unique<CUDABackend>();
}

} // namespace lumen