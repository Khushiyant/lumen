#include "lumen/lumen.hpp"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
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

// --- Pillar 2: CUDA Event Implementation ---
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
  }

  Buffer *create_buffer(const std::vector<size_t> &shape) override {
    size_t total_elements = 1;
    for (auto d : shape)
      total_elements *= d;
    size_t size = total_elements * sizeof(float); // DEFINES 'size'

    void *ptr = pool_.acquire(size);
    if (!ptr)
      cudaMallocManaged(&ptr, size);

    // FIXED: 'strides' logic implemented correctly
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

  // FIXED: Signature returns std::shared_ptr<Event>
  std::shared_ptr<Event> sync(std::vector<QueuedOp> &queue) override {
    if (queue.empty())
      return nullptr;

    for (const auto &op : queue) {
      float *d_out = static_cast<float *>(op.output->device_handle());
      size_t n = op.output->size_bytes() / sizeof(float);

      if (op.op_name == "add" || op.op_name == "mul") {
        float *d_a = static_cast<float *>(op.inputs[0]->device_handle());
        float *d_b = static_cast<float *>(op.inputs[1]->device_handle());

        int threads = 256;
        int blocks = (n + threads - 1) / threads;

        if (op.op_name == "add") {
          add_kernel<<<blocks, threads>>>(d_a, d_b, d_out, n);
        } else {
          mul_kernel<<<blocks, threads>>>(d_a, d_b, d_out, n);
        }
      } else if (op.op_name == "matmul") {
        float *d_a = static_cast<float *>(op.inputs[0]->device_handle());
        float *d_b = static_cast<float *>(op.inputs[1]->device_handle());

        int M = (int)op.inputs[0]->shape()[0];
        int K = (int)op.inputs[0]->shape()[1];
        int N = (int)op.inputs[1]->shape()[1];

        const float alpha = 1.0f;
        const float beta = 0.0f;

        // cuBLAS matmul (A * B)
        cublasSgemm(cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                    d_b, N, d_a, K, &beta, d_out, N);
      } else if (op.op_name == "softmax") {
        float *d_in = (float *)op.inputs[0]->device_handle();
        float *d_out = (float *)op.output->device_handle();
        int inner = (int)op.inputs[0]->shape().back();
        int outer = (int)(op.inputs[0]->num_elements() / inner);

        int threads = 256;
        int blocks = (outer + threads - 1) / threads;
        softmax_kernel<<<blocks, threads>>>(d_in, d_out, outer, inner);
      } else if (op.op_name == "conv2d") {
        float *d_in = (float *)op.inputs[0]->device_handle();
        float *d_w = (float *)op.inputs[1]->device_handle();
        float *d_out = (float *)op.output->device_handle();

        auto &in_s = op.inputs[0]->shape();
        auto &w_s = op.inputs[1]->shape();
        auto &out_s = op.output->shape();

        auto stride = op.attrs.get_int_array("stride");
        auto padding = op.attrs.get_int_array("padding");
        int sh = stride.empty() ? 1 : stride[0];
        int sw = stride.size() > 1 ? stride[1] : sh;
        int ph = padding.empty() ? 0 : padding[0];
        int pw = padding.size() > 1 ? padding[1] : ph;

        int total = out_s[0] * out_s[1] * out_s[2] * out_s[3];
        int threads = 256;
        int blocks = (total + threads - 1) / threads;

        conv2d_kernel<<<blocks, threads>>>(
            d_in, d_w, d_out, in_s[0], (int)in_s[1], (int)in_s[2], (int)in_s[3],
            (int)out_s[1], (int)w_s[2], (int)w_s[3], (int)out_s[2],
            (int)out_s[3], sh, sw, ph, pw);
      } else {
        throw std::runtime_error("Unsupported operation: " + op.op_name);
      }
    }

    // Return event for asynchronous tracking
    return std::make_shared<CUDAEvent>();
  }

private:
  cublasHandle_t cublas_handle_ = nullptr;
};

std::unique_ptr<Backend> create_cuda_backend() {
  return std::make_unique<CUDABackend>();
}

} // namespace lumen