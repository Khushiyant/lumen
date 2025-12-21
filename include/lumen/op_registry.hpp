#pragma once
#include <functional>
#include <lumen/lumen.hpp>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace lumen {

// Forward declarations
class Buffer;
struct OpAttributes;




// ============================================================================
// OPERATION CONTEXT
// ============================================================================

struct OpContext {
  std::vector<Buffer *> inputs;
  Buffer *output;
  OpAttributes attrs;

  // Helper methods for common patterns
  float *input_ptr(size_t idx) const;
  float *output_ptr() const;
  size_t output_size() const;
};

// ============================================================================
// OPERATION SIGNATURE
// ============================================================================

using OpKernel = std::function<void(const OpContext &)>;

struct OpMetadata {
  std::string name;
  int min_inputs;
  int max_inputs;
  bool supports_fusion;
  std::vector<std::string> fusable_with;
};

// ============================================================================
// BACKEND OPERATION REGISTRY
// ============================================================================

class OpRegistry {
public:
  // Register an operation implementation
  void register_op(const std::string &op_name, OpKernel kernel,
                   const OpMetadata &metadata = {});

  // Check if operation is supported
  bool supports(const std::string &op_name) const;

  // Execute an operation
  void execute(const std::string &op_name, const OpContext &ctx) const;

  // Get list of all supported operations
  std::vector<std::string> list_ops() const;

  // Get metadata for an operation
  const OpMetadata *get_metadata(const std::string &op_name) const;

private:
  std::map<std::string, OpKernel> kernels_;
  std::map<std::string, OpMetadata> metadata_;
};

// ============================================================================
// OPERATION CATALOG (Reference implementations)
// ============================================================================

namespace ops {

// Element-wise operations
void add_cpu(const OpContext &ctx);
void mul_cpu(const OpContext &ctx);
void relu_cpu(const OpContext &ctx);
void sigmoid_cpu(const OpContext &ctx);
void tanh_cpu(const OpContext &ctx);

// Matrix operations
void matmul_cpu(const OpContext &ctx);

// Convolution operations
void conv2d_cpu(const OpContext &ctx);

// Normalization operations
void softmax_cpu(const OpContext &ctx);
void layer_norm_cpu(const OpContext &ctx);
void batch_norm_cpu(const OpContext &ctx);

// Pooling operations
void max_pool2d_cpu(const OpContext &ctx);
void avg_pool2d_cpu(const OpContext &ctx);
void global_avg_pool_cpu(const OpContext &ctx);

// Reduction operations
void reduce_mean_cpu(const OpContext &ctx);
void reduce_sum_cpu(const OpContext &ctx);

void flatten_cpu(const OpContext &ctx);
void reshape_cpu(const OpContext &ctx);

} // namespace ops

// ============================================================================
// BACKEND BASE CLASS WITH REGISTRY
// ============================================================================

class BackendWithRegistry {
protected:
  OpRegistry registry_;

  // Helper to register all standard operations
  void register_standard_ops();

  // Subclasses override this to add backend-specific optimizations
  virtual void register_backend_ops() {}

public:
  virtual ~BackendWithRegistry() = default;

  // Check operation support
  bool supports_op(const std::string &op_name) const {
    return registry_.supports(op_name);
  }

  // Execute operation through registry
  void execute_op(const std::string &op_name, const OpContext &ctx) {
    registry_.execute(op_name, ctx);
  }

  // Get supported operations
  std::vector<std::string> supported_ops() const {
    return registry_.list_ops();
  }
};

} // namespace lumen