#include "lumen/lumen.hpp"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace lumen {

// --- CUDA Kernels ---
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

class CUDABackend : public Backend {
public:
  CUDABackend() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    if (err != cudaSuccess) {
      throw std::runtime_error("CUDA Driver Error: " +
                               std::string(cudaGetErrorString(err)));
    }

    if (deviceCount == 0) {
      throw std::runtime_error("No CUDA-capable devices found on this system.");
    }

    cudaSetDevice(0);

    cublasStatus_t stat = cublasCreate(&cublas_handle_);
    if (stat != CUBLAS_STATUS_SUCCESS) {
      throw std::runtime_error("Failed to initialize cuBLAS.");
    }

    std::cout << "[Lumen] CUDA Backend Initialized with Memory Pooling"
              << std::endl;
  }

  ~CUDABackend() {
    if (cublas_handle_)
      cublasDestroy(cublas_handle_);

    auto blocks = pool_.drain();
    for (auto &block : blocks) {
      // Explicitly free the managed memory pointers
      cudaFree(block.second);
    }
    std::cout << "[Lumen] CUDA Backend: Memory Pool Drained." << std::endl;
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

  void execute(const std::string &op_name, const std::vector<Buffer *> &inputs,
               Buffer *output) override {
    std::vector<QueuedOp> q = {{op_name, inputs, output}};
    sync(q);
  }

  // Helper to resolve device pointer with offset safely
  float *get_device_ptr(Buffer *b) {
    return (float *)((char *)b->device_ptr() + b->offset_bytes());
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
        const std::string &current_op = sub_ops[i];

        // Output resolution
        float *d_out = get_device_ptr(op.output);
        size_t n = op.output->num_elements();

        std::vector<Buffer *> effective_inputs =
            (i == 0) ? op.inputs : std::vector<Buffer *>{op.output};

        if (current_op == "add" || current_op == "mul") {
          float *d_a = get_device_ptr(effective_inputs[0]);
          float *d_b = get_device_ptr(effective_inputs[1]);
          int threads = 256;
          int blocks = (n + threads - 1) / threads;
          if (current_op == "add")
            add_kernel<<<blocks, threads>>>(d_a, d_b, d_out, (int)n);
          else
            mul_kernel<<<blocks, threads>>>(d_a, d_b, d_out, (int)n);
        } else if (current_op == "matmul") {
          float *d_a = get_device_ptr(effective_inputs[0]);
          float *d_b = get_device_ptr(effective_inputs[1]);
          int M = (int)effective_inputs[0]->shape()[0],
              K = (int)effective_inputs[0]->shape()[1],
              N = (int)effective_inputs[1]->shape()[1];
          const float alpha = 1.0f, beta = 0.0f;
          cublasSgemm(cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                      d_b, N, d_a, K, &beta, d_out, N);
        } else if (current_op == "relu") {
          // Basic inplace relu
          float *d_in = get_device_ptr(effective_inputs[0]);
          // Placeholder: in production, launch a relu kernel here
          // Using simple mul with 1/0 is not efficient, needs custom kernel
        } else if (current_op == "softmax") {
          float *d_in = get_device_ptr(effective_inputs[0]);
          int inner = (int)effective_inputs[0]->shape().back();
          int outer = (int)(n / inner);
          int threads = 256, blocks = (outer + threads - 1) / threads;
          softmax_kernel<<<blocks, threads>>>(d_in, d_out, outer, inner);
        } else if (current_op == "conv2d") {
          float *d_in = get_device_ptr(effective_inputs[0]);
          float *d_w = get_device_ptr(effective_inputs[1]);
          auto &in_s = effective_inputs[0]->shape();
          auto &w_s = effective_inputs[1]->shape();
          auto &out_s = op.output->shape();
          auto stride = op.attrs.get_int_array("stride");
          auto padding = op.attrs.get_int_array("padding");
          int sh = stride.empty() ? 1 : stride[0],
              sw = stride.size() > 1 ? stride[1] : sh;
          int ph = padding.empty() ? 0 : padding[0],
              pw = padding.size() > 1 ? padding[1] : ph;
          int total = out_s[0] * out_s[1] * out_s[2] * out_s[3];
          int threads = 256, blocks = (total + threads - 1) / threads;
          conv2d_kernel<<<blocks, threads>>>(
              d_in, d_w, d_out, (int)in_s[0], (int)in_s[1], (int)in_s[2],
              (int)in_s[3], (int)out_s[1], (int)w_s[2], (int)w_s[3],
              (int)out_s[2], (int)out_s[3], sh, sw, ph, pw);
        }
      }
    }
    return std::make_shared<CUDAEvent>();
  }

private:
  cublasHandle_t cublas_handle_ = nullptr;
};

std::unique_ptr<Backend> create_cuda_backend() {
  return std::make_unique<CUDABackend>();
}

} // namespace lumen