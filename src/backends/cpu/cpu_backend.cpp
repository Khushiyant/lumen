#include "lumen/lumen.hpp"
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#elif defined(LUMEN_USE_OPENBLAS)
#include <cblas.h>
#endif

namespace lumen {
class CPUEvent : public Event {
public:
  void wait() override {} // Already done
  bool is_completed() override { return true; }
};
class CPUBackend : public Backend {
public:
  CPUBackend() {
#ifdef LUMEN_USE_OPENBLAS
    std::cout << "[Lumen] CPU Backend Initialized (OpenBLAS Optimized)"
              << std::endl;
#elif defined(__APPLE__)
    std::cout << "[Lumen] CPU Backend Initialized (Apple Accelerate)"
              << std::endl;
#else
    std::cout << "[Lumen] CPU Backend Initialized (Naive Fallback)"
              << std::endl;
#endif
  }

  Buffer *create_buffer(const std::vector<size_t> &shape) override {
    size_t total_elements = 1;
    for (auto d : shape)
      total_elements *= d;
    size_t size = total_elements * sizeof(float);

    // 1. Try to acquire from pool
    void *ptr = pool_.acquire(size);

    // 2. Fallback to allocation if pool is empty
    if (!ptr) {
      ptr = new float[total_elements];
    }

    std::vector<size_t> strides(shape.size());
    size_t s = 1;
    for (int i = (int)shape.size() - 1; i >= 0; --i) {
      strides[i] = s;
      s *= shape[i];
    }
    return new Buffer(shape, strides, ptr, ptr, this, 0);
  }

  void free_buffer(void *device_ptr, size_t size) override {
    if (device_ptr) {
      // Instead of deleting, return to pool
      pool_.release(device_ptr, size);
    }
  }

  void execute(const std::string &op_name, const std::vector<Buffer *> &inputs,
               Buffer *output) override {
    std::vector<QueuedOp> q = {{op_name, inputs, output}};
    sync(q);
  }

  std::shared_ptr<Event> sync(std::vector<QueuedOp> &queue) override {
    for (const auto &op : queue) {
      float *out = (float *)op.output->data();
      size_t n = op.output->size_bytes() / sizeof(float);
      if (op.op_name == "relu") {
        Buffer *in_buf = op.inputs[0];
        Buffer *out_buf = op.output;

        float *in_data = (float *)in_buf->data();
        float *out_data = (float *)out_buf->data();

        auto &shape = in_buf->shape();
        auto &in_strides = in_buf->strides();
        auto &out_strides = out_buf->strides();

        // For a 2D Tensor (generalized to N-D would use recursion or a flat
        // iterator)
        for (size_t i = 0; i < shape[0]; ++i) {
          for (size_t j = 0; j < shape[1]; ++j) {
            // Calculate exact memory positions using strides
            size_t in_idx = i * in_strides[0] + j * in_strides[1];
            size_t out_idx = i * out_strides[0] + j * out_strides[1];

            out_data[out_idx] = std::max(0.0f, in_data[in_idx]);
          }
        }
      } else if (op.op_name == "add") {
        float *a = (float *)op.inputs[0]->data();
        float *b = (float *)op.inputs[1]->data();
        for (size_t i = 0; i < n; ++i)
          out[i] = a[i] + b[i];
      } else if (op.op_name == "mul") {
        float *a = (float *)op.inputs[0]->data();
        float *b = (float *)op.inputs[1]->data();
        for (size_t i = 0; i < n; ++i)
          out[i] = a[i] * b[i];
      } else if (op.op_name == "matmul") {
        float *A = (float *)op.inputs[0]->data();
        float *B = (float *)op.inputs[1]->data();
        int M = (int)op.inputs[0]->shape()[0];
        int K = (int)op.inputs[0]->shape()[1];
        int N = (int)op.inputs[1]->shape()[1];
        

#if defined(__APPLE__) || defined(LUMEN_USE_OPENBLAS)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f, A,
                    K, B, N, 0.0f, out, N);
#else
        // Naive fallback only if no BLAS is present
        for (int i = 0; i < M; ++i)
          for (int j = 0; j < N; ++j) {
            float sum = 0;
            for (int k = 0; k < K; ++k)
              sum += A[i * K + k] * B[k * N + j];
            out[i * N + j] = sum;
          }
#endif
      }
      // Add to CPUBackend::sync in lumen/src/backends/cpu/cpu_backend.cpp
      else if (op.op_name == "softmax") {
        float *in_data = (float *)op.inputs[0]->data();
        float *out_data = (float *)op.output->data();
        auto &shape = op.inputs[0]->shape();

        size_t last_dim = shape.back();
        size_t outer_size = op.inputs[0]->num_elements() / last_dim;

        for (size_t i = 0; i < outer_size; ++i) {
          float *in_row = in_data + (i * last_dim);
          float *out_row = out_data + (i * last_dim);

          // 1. Find Max for numerical stability
          float max_val = in_row[0];
          for (size_t j = 1; j < last_dim; ++j)
            max_val = std::max(max_val, in_row[j]);

          // 2. Compute Sum of Exponentials
          float sum = 0.0f;
          for (size_t j = 0; j < last_dim; ++j) {
            out_row[j] = std::exp(in_row[j] - max_val);
            sum += out_row[j];
          }

          // 3. Normalize
          float inv_sum = 1.0f / sum;
          for (size_t j = 0; j < last_dim; ++j) {
            out_row[j] *= inv_sum;
          }
        }
      } else if (op.op_name == "conv2d") {
        float *in = (float *)op.inputs[0]->data();
        float *weight = (float *)op.inputs[1]->data();
        float *out = (float *)op.output->data();

        auto &in_s = op.inputs[0]->shape(); // [N, C_in, H, W]
        auto &w_s = op.inputs[1]->shape();  // [C_out, C_in, KH, KW]
        auto &out_s = op.output->shape();   // [N, C_out, HO, WO]

        auto stride = op.attrs.get_int_array("stride");
        auto padding = op.attrs.get_int_array("padding");
        int sh = stride.empty() ? 1 : stride[0];
        int sw = stride.size() > 1 ? stride[1] : sh;
        int ph = padding.empty() ? 0 : padding[0];
        int pw = padding.size() > 1 ? padding[1] : ph;

        for (int n = 0; n < out_s[0]; ++n) {
          for (int oc = 0; oc < out_s[1]; ++oc) {
            for (int oh = 0; oh < out_s[2]; ++oh) {
              for (int ow = 0; ow < out_s[3]; ++ow) {
                float sum = 0;
                for (int ic = 0; ic < in_s[1]; ++ic) {
                  for (int kh = 0; kh < w_s[2]; ++kh) {
                    for (int kw = 0; kw < w_s[3]; ++kw) {
                      int ih = oh * sh - ph + kh;
                      int iw = ow * sw - pw + kw;
                      if (ih >= 0 && ih < in_s[2] && iw >= 0 && iw < in_s[3]) {
                        sum +=
                            in[((n * in_s[1] + ic) * in_s[2] + ih) * in_s[3] +
                               iw] *
                            weight[((oc * w_s[1] + ic) * w_s[2] + kh) * w_s[3] +
                                   kw];
                      }
                    }
                  }
                }
                out[((n * out_s[1] + oc) * out_s[2] + oh) * out_s[3] + ow] =
                    sum;
              }
            }
          }
        }
      } else if (op.op_name == "global_average_pool") {
        float *in = (float *)op.inputs[0]->data();
        float *out = (float *)op.output->data();
        auto &s = op.inputs[0]->shape();
        int N = s[0], C = s[1], HW = s[2] * s[3];
        float inv_hw = 1.0f / HW;

        for (int n = 0; n < N; ++n) {
          for (int c = 0; c < C; ++c) {
            float sum = 0;
            for (int i = 0; i < HW; ++i)
              sum += in[(n * C + c) * HW + i];
            out[n * C + c] = sum * inv_hw;
          }
        }
      } else {
        throw std::runtime_error("Unsupported op: " + op.op_name);
      }
    }
    return std::make_shared<CPUEvent>();
  }
};

std::unique_ptr<Backend> create_cpu_backend() {
  return std::make_unique<CPUBackend>();
}

} // namespace lumen