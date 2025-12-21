#include "lumen/op_registry.hpp"
#include "lumen/lumen.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace lumen {

// ============================================================================
// OPCONTEXT HELPERS
// ============================================================================

float *OpContext::input_ptr(size_t idx) const {
  if (idx >= inputs.size()) {
    throw std::runtime_error("Input index out of range");
  }
  return (float *)inputs[idx]->data();
}

float *OpContext::output_ptr() const { return (float *)output->data(); }

size_t OpContext::output_size() const { return output->num_elements(); }

// ============================================================================
// OPREGISTRY IMPLEMENTATION
// ============================================================================

void OpRegistry::register_op(const std::string &op_name, OpKernel kernel,
                             const OpMetadata &metadata) {
  kernels_[op_name] = kernel;
  metadata_[op_name] =
      metadata.name.empty() ? OpMetadata{op_name, 1, 10, false, {}} : metadata;
}

bool OpRegistry::supports(const std::string &op_name) const {
  return kernels_.find(op_name) != kernels_.end();
}

void OpRegistry::execute(const std::string &op_name,
                         const OpContext &ctx) const {
  auto it = kernels_.find(op_name);
  if (it == kernels_.end()) {
    throw std::runtime_error("Operation not supported: " + op_name);
  }

  // Validate input count
  auto meta_it = metadata_.find(op_name);
  if (meta_it != metadata_.end()) {
    const auto &meta = meta_it->second;
    int n_inputs = static_cast<int>(ctx.inputs.size());
    if (n_inputs < meta.min_inputs || n_inputs > meta.max_inputs) {
      throw std::runtime_error("Invalid number of inputs for " + op_name);
    }
  }

  it->second(ctx);
}

std::vector<std::string> OpRegistry::list_ops() const {
  std::vector<std::string> ops;
  for (const auto &[name, _] : kernels_) {
    ops.push_back(name);
  }
  return ops;
}

const OpMetadata *OpRegistry::get_metadata(const std::string &op_name) const {
  auto it = metadata_.find(op_name);
  return it != metadata_.end() ? &it->second : nullptr;
}

// ============================================================================
// CPU OPERATION IMPLEMENTATIONS
// ============================================================================

namespace ops {

void add_cpu(const OpContext &ctx) {
  float *a = ctx.input_ptr(0);
  float *b = ctx.input_ptr(1);
  float *out = ctx.output_ptr();
  size_t n = ctx.output_size();

  for (size_t i = 0; i < n; ++i) {
    out[i] = a[i] + b[i];
  }
}

void mul_cpu(const OpContext &ctx) {
  float *a = ctx.input_ptr(0);
  float *b = ctx.input_ptr(1);
  float *out = ctx.output_ptr();
  size_t n = ctx.output_size();

  for (size_t i = 0; i < n; ++i) {
    out[i] = a[i] * b[i];
  }
}

void relu_cpu(const OpContext &ctx) {
  float *in = ctx.input_ptr(0);
  float *out = ctx.output_ptr();
  size_t n = ctx.output_size();

  for (size_t i = 0; i < n; ++i) {
    out[i] = std::max(0.0f, in[i]);
  }
}

void sigmoid_cpu(const OpContext &ctx) {
  float *in = ctx.input_ptr(0);
  float *out = ctx.output_ptr();
  size_t n = ctx.output_size();

  for (size_t i = 0; i < n; ++i) {
    out[i] = 1.0f / (1.0f + std::exp(-in[i]));
  }
}

void tanh_cpu(const OpContext &ctx) {
  float *in = ctx.input_ptr(0);
  float *out = ctx.output_ptr();
  size_t n = ctx.output_size();

  for (size_t i = 0; i < n; ++i) {
    out[i] = std::tanh(in[i]);
  }
}

void matmul_cpu(const OpContext &ctx) {
  float *A = ctx.input_ptr(0);
  float *B = ctx.input_ptr(1);
  float *out = ctx.output_ptr();

  auto &shape_a = ctx.inputs[0]->shape();
  auto &shape_b = ctx.inputs[1]->shape();

  bool trans_a = ctx.attrs.get_int("transA", 0);
  bool trans_b = ctx.attrs.get_int("transB", 0);

  int M = trans_a ? shape_a[1] : shape_a[0];
  int K = trans_a ? shape_a[0] : shape_a[1];
  int N = trans_b ? shape_b[0] : shape_b[1];

#if defined(__APPLE__) || defined(LUMEN_USE_OPENBLAS)
#include <Accelerate/Accelerate.h>
  cblas_sgemm(CblasRowMajor, trans_a ? CblasTrans : CblasNoTrans,
              trans_b ? CblasTrans : CblasNoTrans, M, N, K, 1.0f, A,
              trans_a ? M : K, B, trans_b ? K : N, 0.0f, out, N);
#else
  // Naive implementation
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      float sum = 0.0f;
      for (int k = 0; k < K; ++k) {
        int a_idx = trans_a ? (k * M + m) : (m * K + k);
        int b_idx = trans_b ? (n * K + k) : (k * N + n);
        sum += A[a_idx] * B[b_idx];
      }
      out[m * N + n] = sum;
    }
  }
#endif
}

void conv2d_cpu(const OpContext &ctx) {
  float *in = ctx.input_ptr(0);
  float *weight = ctx.input_ptr(1);
  float *out = ctx.output_ptr();

  auto &in_s = ctx.inputs[0]->shape();
  auto &w_s = ctx.inputs[1]->shape();
  auto &out_s = ctx.output->shape();

  auto stride = ctx.attrs.get_int_array("stride");
  auto padding = ctx.attrs.get_int_array("padding");

  int sh = stride.empty() ? 1 : stride[0];
  int sw = stride.size() > 1 ? stride[1] : sh;
  int ph = padding.empty() ? 0 : padding[0];
  int pw = padding.size() > 1 ? padding[1] : ph;

  for (int batch = 0; batch < out_s[0]; ++batch) {
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
                      in[((batch * in_s[1] + ic) * in_s[2] + ih) * in_s[3] +
                         iw] *
                      weight[((oc * w_s[1] + ic) * w_s[2] + kh) * w_s[3] + kw];
                }
              }
            }
          }
          out[((batch * out_s[1] + oc) * out_s[2] + oh) * out_s[3] + ow] = sum;
        }
      }
    }
  }
}

void softmax_cpu(const OpContext &ctx) {
  float *in = ctx.input_ptr(0);
  float *out = ctx.output_ptr();

  auto &shape = ctx.inputs[0]->shape();
  size_t last_dim = shape.back();
  size_t outer_size = ctx.output_size() / last_dim;

  for (size_t r = 0; r < outer_size; ++r) {
    float *in_row = in + (r * last_dim);
    float *out_row = out + (r * last_dim);

    // Find max for numerical stability
    float max_val = in_row[0];
    for (size_t j = 1; j < last_dim; ++j) {
      max_val = std::max(max_val, in_row[j]);
    }

    // Compute exp and sum
    float sum = 0.0f;
    for (size_t j = 0; j < last_dim; ++j) {
      out_row[j] = std::exp(in_row[j] - max_val);
      sum += out_row[j];
    }

    // Normalize
    float inv_sum = 1.0f / sum;
    for (size_t j = 0; j < last_dim; ++j) {
      out_row[j] *= inv_sum;
    }
  }
}

void layer_norm_cpu(const OpContext &ctx) {
  float *in = ctx.input_ptr(0);
  float *out = ctx.output_ptr();

  float eps = ctx.attrs.get_float("epsilon", 1e-5f);
  auto &shape = ctx.inputs[0]->shape();
  size_t normalized_size = shape.back();
  size_t outer_size = ctx.output_size() / normalized_size;

  for (size_t i = 0; i < outer_size; ++i) {
    float *in_row = in + (i * normalized_size);
    float *out_row = out + (i * normalized_size);

    // Compute mean
    float mean = 0.0f;
    for (size_t j = 0; j < normalized_size; ++j) {
      mean += in_row[j];
    }
    mean /= normalized_size;

    // Compute variance
    float variance = 0.0f;
    for (size_t j = 0; j < normalized_size; ++j) {
      float diff = in_row[j] - mean;
      variance += diff * diff;
    }
    variance /= normalized_size;

    // Normalize
    float inv_std = 1.0f / std::sqrt(variance + eps);
    for (size_t j = 0; j < normalized_size; ++j) {
      out_row[j] = (in_row[j] - mean) * inv_std;
    }
  }
}

void batch_norm_cpu(const OpContext &ctx) {
  float *in = ctx.input_ptr(0);
  float *out = ctx.output_ptr();

  // For now, simplified batch norm (training mode)
  float eps = ctx.attrs.get_float("epsilon", 1e-5f);
  auto &shape = ctx.inputs[0]->shape();

  // Assuming NCHW format: [batch, channels, height, width]
  size_t N = shape[0];
  size_t C = shape[1];
  size_t spatial_size = ctx.output_size() / (N * C);

  for (size_t c = 0; c < C; ++c) {
    // Compute mean across batch and spatial dims
    float mean = 0.0f;
    size_t count = N * spatial_size;
    for (size_t n = 0; n < N; ++n) {
      for (size_t s = 0; s < spatial_size; ++s) {
        mean += in[(n * C + c) * spatial_size + s];
      }
    }
    mean /= count;

    // Compute variance
    float variance = 0.0f;
    for (size_t n = 0; n < N; ++n) {
      for (size_t s = 0; s < spatial_size; ++s) {
        float diff = in[(n * C + c) * spatial_size + s] - mean;
        variance += diff * diff;
      }
    }
    variance /= count;

    // Normalize
    float inv_std = 1.0f / std::sqrt(variance + eps);
    for (size_t n = 0; n < N; ++n) {
      for (size_t s = 0; s < spatial_size; ++s) {
        size_t idx = (n * C + c) * spatial_size + s;
        out[idx] = (in[idx] - mean) * inv_std;
      }
    }
  }
}

void max_pool2d_cpu(const OpContext &ctx) {
  float *in = ctx.input_ptr(0);
  float *out = ctx.output_ptr();

  auto &in_s = ctx.inputs[0]->shape();
  auto &out_s = ctx.output->shape();

  auto kernel = ctx.attrs.get_int_array("kernel_size");
  auto stride = ctx.attrs.get_int_array("stride");
  auto padding = ctx.attrs.get_int_array("padding");

  int kh = kernel.empty() ? 2 : kernel[0];
  int kw = kernel.size() > 1 ? kernel[1] : kh;
  int sh = stride.empty() ? kh : stride[0];
  int sw = stride.size() > 1 ? stride[1] : sh;
  int ph = padding.empty() ? 0 : padding[0];
  int pw = padding.size() > 1 ? padding[1] : ph;

  for (int n = 0; n < out_s[0]; ++n) {
    for (int c = 0; c < out_s[1]; ++c) {
      for (int oh = 0; oh < out_s[2]; ++oh) {
        for (int ow = 0; ow < out_s[3]; ++ow) {
          float max_val = -INFINITY;
          for (int kh_i = 0; kh_i < kh; ++kh_i) {
            for (int kw_i = 0; kw_i < kw; ++kw_i) {
              int ih = oh * sh - ph + kh_i;
              int iw = ow * sw - pw + kw_i;
              if (ih >= 0 && ih < in_s[2] && iw >= 0 && iw < in_s[3]) {
                float val =
                    in[((n * in_s[1] + c) * in_s[2] + ih) * in_s[3] + iw];
                max_val = std::max(max_val, val);
              }
            }
          }
          out[((n * out_s[1] + c) * out_s[2] + oh) * out_s[3] + ow] = max_val;
        }
      }
    }
  }
}

void avg_pool2d_cpu(const OpContext &ctx) {
  float *in = ctx.input_ptr(0);
  float *out = ctx.output_ptr();

  auto &in_s = ctx.inputs[0]->shape();
  auto &out_s = ctx.output->shape();

  auto kernel = ctx.attrs.get_int_array("kernel_size");
  auto stride = ctx.attrs.get_int_array("stride");
  auto padding = ctx.attrs.get_int_array("padding");

  int kh = kernel.empty() ? 2 : kernel[0];
  int kw = kernel.size() > 1 ? kernel[1] : kh;
  int sh = stride.empty() ? kh : stride[0];
  int sw = stride.size() > 1 ? stride[1] : sh;
  int ph = padding.empty() ? 0 : padding[0];
  int pw = padding.size() > 1 ? padding[1] : ph;

  for (int n = 0; n < out_s[0]; ++n) {
    for (int c = 0; c < out_s[1]; ++c) {
      for (int oh = 0; oh < out_s[2]; ++oh) {
        for (int ow = 0; ow < out_s[3]; ++ow) {
          float sum = 0.0f;
          int count = 0;
          for (int kh_i = 0; kh_i < kh; ++kh_i) {
            for (int kw_i = 0; kw_i < kw; ++kw_i) {
              int ih = oh * sh - ph + kh_i;
              int iw = ow * sw - pw + kw_i;
              if (ih >= 0 && ih < in_s[2] && iw >= 0 && iw < in_s[3]) {
                sum += in[((n * in_s[1] + c) * in_s[2] + ih) * in_s[3] + iw];
                count++;
              }
            }
          }
          out[((n * out_s[1] + c) * out_s[2] + oh) * out_s[3] + ow] =
              count > 0 ? sum / count : 0.0f;
        }
      }
    }
  }
}

void global_avg_pool_cpu(const OpContext &ctx) {
  float *in = ctx.input_ptr(0);
  float *out = ctx.output_ptr();

  auto &shape = ctx.inputs[0]->shape();
  // Assuming NCHW: [batch, channels, height, width]
  size_t N = shape[0];
  size_t C = shape[1];
  size_t spatial = shape[2] * shape[3];

  for (size_t n = 0; n < N; ++n) {
    for (size_t c = 0; c < C; ++c) {
      float sum = 0.0f;
      for (size_t s = 0; s < spatial; ++s) {
        sum += in[(n * C + c) * spatial + s];
      }
      out[n * C + c] = sum / spatial;
    }
  }
}

void reduce_mean_cpu(const OpContext &ctx) {
  float *in = ctx.input_ptr(0);
  float *out = ctx.output_ptr();

  auto axes = ctx.attrs.get_int_array("axes");
  // Simplified: reduce over last dimension
  auto &shape = ctx.inputs[0]->shape();
  size_t last_dim = shape.back();
  size_t outer = ctx.inputs[0]->num_elements() / last_dim;

  for (size_t i = 0; i < outer; ++i) {
    float sum = 0.0f;
    for (size_t j = 0; j < last_dim; ++j) {
      sum += in[i * last_dim + j];
    }
    out[i] = sum / last_dim;
  }
}

void reduce_sum_cpu(const OpContext &ctx) {
  float *in = ctx.input_ptr(0);
  float *out = ctx.output_ptr();

  auto &shape = ctx.inputs[0]->shape();
  size_t last_dim = shape.back();
  size_t outer = ctx.inputs[0]->num_elements() / last_dim;

  for (size_t i = 0; i < outer; ++i) {
    float sum = 0.0f;
    for (size_t j = 0; j < last_dim; ++j) {
      sum += in[i * last_dim + j];
    }
    out[i] = sum;
  }
}

} // namespace ops

// ============================================================================
// BACKEND BASE CLASS
// ============================================================================

void BackendWithRegistry::register_standard_ops() {
  // Element-wise
  registry_.register_op("add", ops::add_cpu, {"add", 2, 2, true, {"relu"}});
  registry_.register_op("mul", ops::mul_cpu, {"mul", 2, 2, true, {"relu"}});
  registry_.register_op("relu", ops::relu_cpu, {"relu", 1, 1, false, {}});
  registry_.register_op("sigmoid", ops::sigmoid_cpu);
  registry_.register_op("tanh", ops::tanh_cpu);

  // Matrix ops
  registry_.register_op("matmul", ops::matmul_cpu,
                        {"matmul", 2, 2, true, {"relu", "add"}});

  // Conv ops
  registry_.register_op("conv2d", ops::conv2d_cpu,
                        {"conv2d", 2, 2, true, {"relu", "batchnorm"}});

  // Normalization
  registry_.register_op("softmax", ops::softmax_cpu);
  registry_.register_op("layer_norm", ops::layer_norm_cpu);
  registry_.register_op("batchnorm", ops::batch_norm_cpu);

  // Pooling
  registry_.register_op("maxpool2d", ops::max_pool2d_cpu);
  registry_.register_op("avgpool2d", ops::avg_pool2d_cpu);
  registry_.register_op("global_average_pool", ops::global_avg_pool_cpu);

  // Reductions
  registry_.register_op("reduce_mean", ops::reduce_mean_cpu);
  registry_.register_op("reduce_sum", ops::reduce_sum_cpu);
}

} // namespace lumen