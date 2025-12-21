#pragma once
#include <lumen/lumen.hpp>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace lumen {

// ============================================================================
// DATA TYPES & ATTRIBUTES
// ============================================================================

enum class DataType { FLOAT32, FLOAT16, INT8, INT32, INT64, BOOL };

// Flexible attribute system for operation parameters
struct OpAttributes {
  std::map<std::string, int> int_attrs;
  std::map<std::string, float> float_attrs;
  std::map<std::string, std::vector<int>> int_array_attrs;
  std::map<std::string, std::string> string_attrs;
  std::map<std::string, bool> bool_attrs;

  // Convenience getters with defaults
  int get_int(const std::string &key, int default_val = 0) const {
    auto it = int_attrs.find(key);
    return it != int_attrs.end() ? it->second : default_val;
  }

  float get_float(const std::string &key, float default_val = 0.0f) const {
    auto it = float_attrs.find(key);
    return it != float_attrs.end() ? it->second : default_val;
  }

  std::vector<int> get_int_array(const std::string &key) const {
    auto it = int_array_attrs.find(key);
    return it != int_array_attrs.end() ? it->second : std::vector<int>{};
  }

  bool get_bool(const std::string &key, bool default_val = false) const {
    auto it = bool_attrs.find(key);
    return it != bool_attrs.end() ? it->second : default_val;
  }
};

// Extended QueuedOp with attributes for graph execution
struct QueuedOpWithAttrs {
  std::string op_name;
  std::vector<Buffer *> inputs;
  Buffer *output;
  OpAttributes attrs;
  std::string target_backend;
};

// ============================================================================
// TENSOR DESCRIPTOR (Graph-level tensor representation)
// ============================================================================

class TensorDescriptor {
public:
  TensorDescriptor(const std::string &name, const std::vector<size_t> &shape,
                   DataType dtype = DataType::FLOAT32)
      : name_(name), shape_(shape), dtype_(dtype) {}

  const std::string &name() const { return name_; }
  const std::vector<size_t> &shape() const { return shape_; }
  DataType dtype() const { return dtype_; }

  void set_shape(const std::vector<size_t> &shape) { shape_ = shape; }

  size_t num_elements() const {
    size_t total = 1;
    for (auto d : shape_)
      total *= d;
    return total;
  }

  size_t size_bytes() const {
    size_t elem_size = 4; // Default float32
    switch (dtype_) {
    case DataType::FLOAT32:
      elem_size = 4;
      break;
    case DataType::FLOAT16:
      elem_size = 2;
      break;
    case DataType::INT8:
      elem_size = 1;
      break;
    case DataType::INT32:
      elem_size = 4;
      break;
    case DataType::INT64:
      elem_size = 8;
      break;
    case DataType::BOOL:
      elem_size = 1;
      break;
    }
    return num_elements() * elem_size;
  }

private:
  std::string name_;
  std::vector<size_t> shape_;
  DataType dtype_;
};

struct TensorLiveness {
  size_t first_use; // Index of the op that produces this tensor
  size_t last_use;  // Index of the last op that consumes this tensor
  size_t size_bytes;
  std::string name;
};

// ============================================================================
// GRAPH NODE (Represents a single operation)
// ============================================================================

class GraphNode {
public:
  GraphNode(const std::string &name, const std::string &op_type,
            const std::vector<TensorDescriptor *> &inputs,
            TensorDescriptor *output, const OpAttributes &attrs = {})
      : name_(name), op_type_(op_type), inputs_(inputs), output_(output),
        attrs_(attrs) {}

  const std::string &name() const { return name_; }
  const std::string &op_type() const { return op_type_; }
  const std::vector<TensorDescriptor *> &inputs() const { return inputs_; }
  TensorDescriptor *output() const { return output_; }
  const OpAttributes &attrs() const { return attrs_; }

  // Check if this node can be fused with the next node
  bool can_fuse_with(const GraphNode *next) const {
    // Simple fusion rules (expand as needed)
    if (op_type_ == "conv2d" && next->op_type_ == "relu")
      return true;
    if (op_type_ == "conv2d" && next->op_type_ == "batchnorm")
      return true;
    if (op_type_ == "matmul" && next->op_type_ == "relu")
      return true;
    return false;
  }

  // Get fused operation name
  std::string get_fused_name(const GraphNode *next) const {
    return op_type_ + "_" + next->op_type_;
  }

private:
  std::string name_;
  std::string op_type_;
  std::vector<TensorDescriptor *> inputs_;
  TensorDescriptor *output_;
  OpAttributes attrs_;
};

// ============================================================================
// SHAPE INFERENCE ENGINE
// ============================================================================

class ShapeInference {
public:
  static std::vector<size_t>
  infer_shape(const std::string &op_type,
              const std::vector<TensorDescriptor *> &inputs,
              const OpAttributes &attrs);

private:
  static std::vector<size_t>
  infer_conv2d(const std::vector<size_t> &input_shape,
               const std::vector<size_t> &weight_shape,
               const OpAttributes &attrs);

  static std::vector<size_t> infer_matmul(const std::vector<size_t> &a_shape,
                                          const std::vector<size_t> &b_shape);

  static std::vector<size_t>
  infer_elementwise(const std::vector<size_t> &shape);

  static std::vector<size_t>
  infer_pool2d(const std::vector<size_t> &input_shape,
               const OpAttributes &attrs);

  static std::vector<size_t>
  infer_batchnorm(const std::vector<size_t> &input_shape);
};

// ============================================================================
// GRAPH (Computation Graph Builder & Optimizer)
// ============================================================================

class Graph {
public:
  Graph() : node_counter_(0), tensor_counter_(0) {}

  // Build graph: Add inputs
  TensorDescriptor *add_input(const std::string &name,
                              const std::vector<size_t> &shape,
                              DataType dtype = DataType::FLOAT32);

  // Build graph: Add operation
  TensorDescriptor *add_op(const std::string &op_type,
                           const std::vector<TensorDescriptor *> &inputs,
                           const OpAttributes &attrs = {},
                           const std::string &name = "");

  // Build graph: Add weight/parameter
  TensorDescriptor *add_weight(const std::string &name,
                               const std::vector<size_t> &shape,
                               const float *data = nullptr);

  // Mark output tensors
  void mark_output(TensorDescriptor *tensor);

  // Optimization passes
  void optimize();
  void fuse_operations();
  void eliminate_dead_code();
  void optimize_memory_layout();

  // Compilation
  class ExecutableGraph *compile(Runtime *rt);

  // Introspection
  const std::vector<std::unique_ptr<GraphNode>> &nodes() const {
    return nodes_;
  }
  const std::vector<TensorDescriptor *> &inputs() const { return inputs_; }
  const std::vector<TensorDescriptor *> &outputs() const { return outputs_; }
  const std::map<std::string, TensorDescriptor *> &weights() const {
    return weights_;
  }
  const std::map<std::string, std::vector<float>> &weight_data() const {
    return weight_data_;
  }

  // Debugging
  void print_summary() const;
  std::string to_string() const;

private:
  std::vector<std::unique_ptr<GraphNode>> nodes_;
  std::vector<std::unique_ptr<TensorDescriptor>> tensors_;
  std::vector<TensorDescriptor *> inputs_;
  std::vector<TensorDescriptor *> outputs_;
  std::map<std::string, TensorDescriptor *> weights_;
  std::map<std::string, std::vector<float>> weight_data_;

  size_t node_counter_;
  size_t tensor_counter_;

  std::string generate_node_name() {
    return "node_" + std::to_string(node_counter_++);
  }

  std::string generate_tensor_name() {
    return "tensor_" + std::to_string(tensor_counter_++);
  }
};

// ============================================================================
// EXECUTABLE GRAPH (Compiled, optimized, ready-to-run graph)
// ============================================================================

class ExecutableGraph {
public:
  ExecutableGraph(Runtime *rt, const Graph *graph);
  ~ExecutableGraph();

  // Execute inference
  std::vector<Buffer *> execute(const std::vector<Buffer *> &inputs);

  // Execute with custom output buffers (advanced)
  void execute(const std::vector<Buffer *> &inputs,
               const std::vector<Buffer *> &outputs);

  // Memory management
  size_t get_memory_usage() const { return total_memory_bytes_; }

  // Profiling
  struct ProfilingData {
    std::string node_name;
    std::string op_type;
    double time_ms;
    std::string backend;
  };
  std::vector<ProfilingData> profile(const std::vector<Buffer *> &inputs);

private:
  Runtime *runtime_;

  // Execution plan
  std::vector<QueuedOpWithAttrs> execution_plan_;

  // Buffer management
  std::map<std::string, Buffer *> intermediate_buffers_;
  std::map<std::string, Buffer *> weight_buffers_;
  std::vector<Buffer *> output_buffers_;

  size_t total_memory_bytes_;

  void analyze_liveness(const Graph *graph,
                        std::map<std::string, TensorLiveness> &liveness);
  void plan_memory_reuse(const std::map<std::string, TensorLiveness> &liveness);
  void allocate_buffers(const Graph *graph);
  void load_weights(const Graph *graph);
  void build_execution_plan(const Graph *graph);

  Buffer *get_or_create_buffer(const std::string &name,
                               const std::vector<size_t> &shape);
};

// ============================================================================
// EXCEPTIONS
// ============================================================================

class GraphException : public std::runtime_error {
public:
  explicit GraphException(const std::string &msg) : std::runtime_error(msg) {}
};

class ShapeInferenceError : public GraphException {
public:
  explicit ShapeInferenceError(const std::string &msg)
      : GraphException("Shape inference failed: " + msg) {}
};

class GraphCompilationError : public GraphException {
public:
  explicit GraphCompilationError(const std::string &msg)
      : GraphException("Graph compilation failed: " + msg) {}
};

} // namespace lumen