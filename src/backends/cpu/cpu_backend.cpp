#include "lumen/lumen.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <sstream>
#include <vector>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#elif defined(LUMEN_USE_OPENBLAS)
#include <cblas.h>
#endif

namespace lumen {
class CPUEvent : public Event {
public:
  void wait() override {}
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
    void *ptr = pool_.acquire(size);
    if (!ptr)
      ptr = new float[total_elements];

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

  std::shared_ptr<Event> sync(std::vector<QueuedOp> &queue) override {
    for (const auto &op : queue) {
      std::vector<std::string> sub_ops;
      std::string segment;
      std::stringstream ss(op.op_name);
      while (std::getline(ss, segment, '_'))
        sub_ops.push_back(segment);

      for (size_t i = 0; i < sub_ops.size(); ++i) {
        const std::string &current_op = sub_ops[i];
        std::vector<Buffer *> effective_inputs =
            (i == 0) ? op.inputs : std::vector<Buffer *>{op.output};
        float *out = (float *)op.output->data();
        size_t n = op.output->num_elements();

        if (current_op == "matmul") {
          float *A = (float *)effective_inputs[0]->data();
          float *B = (float *)effective_inputs[1]->data();
          int M = (int)effective_inputs[0]->shape()[0];
          int K = (int)effective_inputs[0]->shape()[1];
          bool trans_b = op.attrs.get_int("transB", 0);
          int N = trans_b ? (int)effective_inputs[1]->shape()[0]
                          : (int)effective_inputs[1]->shape()[1];

#if defined(__APPLE__) || defined(LUMEN_USE_OPENBLAS)
          cblas_sgemm(CblasRowMajor, CblasNoTrans,
                      trans_b ? CblasTrans : CblasNoTrans, M, N, K, 1.0f, A, K,
                      B, trans_b ? K : N, 0.0f, out, N);
#else
          for (int m = 0; m < M; ++m)
            for (int j = 0; j < N; ++j) {
              float sum = 0;
              for (int k = 0; k < K; ++k) {
                float val_b = trans_b ? B[j * K + k] : B[k * N + j];
                sum += A[m * K + k] * val_b;
              }
              out[m * N + j] = sum;
            }
#endif
        } else if (current_op == "relu") {
          float *in_data = (float *)effective_inputs[0]->data();
          for (size_t j = 0; j < n; ++j)
            out[j] = std::max(0.0f, in_data[j]);
        } else if (current_op == "add") {
          float *a = (float *)effective_inputs[0]->data();
          float *b = (float *)effective_inputs[1]->data();
          for (size_t j = 0; j < n; ++j)
            out[j] = a[j] + b[j];
        } else if (current_op == "mul") {
          float *a = (float *)effective_inputs[0]->data();
          float *b = (float *)effective_inputs[1]->data();
          for (size_t j = 0; j < n; ++j)
            out[j] = a[j] * b[j];
        } else if (current_op == "softmax") {
          float *in_data = (float *)effective_inputs[0]->data();
          size_t last_dim = effective_inputs[0]->shape().back();
          size_t outer_size = n / last_dim;
          for (size_t r = 0; r < outer_size; ++r) {
            float *in_row = in_data + (r * last_dim);
            float *out_row = out + (r * last_dim);
            float max_val = in_row[0];
            for (size_t j = 1; j < last_dim; ++j)
              max_val = std::max(max_val, in_row[j]);
            float sum = 0.0f;
            for (size_t j = 0; j < last_dim; ++j) {
              out_row[j] = std::exp(in_row[j] - max_val);
              sum += out_row[j];
            }
            float inv_sum = 1.0f / sum;
            for (size_t j = 0; j < last_dim; ++j)
              out_row[j] *= inv_sum;
          }
        } else if (current_op == "conv2d") {
          float *in = (float *)effective_inputs[0]->data();
          float *weight = (float *)effective_inputs[1]->data();
          auto &in_s = effective_inputs[0]->shape();
          auto &w_s = effective_inputs[1]->shape();
          auto &out_s = op.output->shape();
          auto stride = op.attrs.get_int_array("stride");
          auto padding = op.attrs.get_int_array("padding");
          int sh = stride.empty() ? 1 : stride[0],
              sw = stride.size() > 1 ? stride[1] : sh;
          int ph = padding.empty() ? 0 : padding[0],
              pw = padding.size() > 1 ? padding[1] : ph;
          for (int batch = 0; batch < out_s[0]; ++batch) {
            for (int oc = 0; oc < out_s[1]; ++oc) {
              for (int oh = 0; oh < out_s[2]; ++oh) {
                for (int ow = 0; ow < out_s[3]; ++ow) {
                  float sum = 0;
                  for (int ic = 0; ic < in_s[1]; ++ic) {
                    for (int kh = 0; kh < w_s[2]; ++kh) {
                      for (int kw = 0; kw < w_s[3]; ++kw) {
                        int ih = oh * sh - ph + kh, iw = ow * sw - pw + kw;
                        if (ih >= 0 && ih < in_s[2] && iw >= 0 &&
                            iw < in_s[3]) {
                          sum += in[((batch * in_s[1] + ic) * in_s[2] + ih) *
                                        in_s[3] +
                                    iw] *
                                 weight[((oc * w_s[1] + ic) * w_s[2] + kh) *
                                            w_s[3] +
                                        kw];
                        }
                      }
                    }
                  }
                  out[((batch * out_s[1] + oc) * out_s[2] + oh) * out_s[3] +
                      ow] = sum;
                }
              }
            }
          }
        }
      }
    }
    return std::make_shared<CPUEvent>();
  }
};

std::unique_ptr<Backend> create_cpu_backend() {
  return std::make_unique<CPUBackend>();
}
} // namespace lumen