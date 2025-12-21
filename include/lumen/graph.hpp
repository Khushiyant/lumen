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

struct QueuedOpWithAttrs {
  std::string op_name;
  std::vector<Buffer *> inputs;
  Buffer *output;
  OpAttributes attrs;
  std::string target_backend;
};

// ============================================================================
// TENSOR DESCRIPTOR
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
    size_t elem_size = 4;
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
  size_t first_use;
  size_t last_use;
  size_t size_bytes;
  std::string name;
};

// ============================================================================
// GRAPH NODE
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

  bool can_fuse_with(const GraphNode *next) const {
    if (op_type_ == "conv2d" && next->op_type_ == "relu")
      return true;
    if (op_type_ == "conv2d" && next->op_type_ == "batchnorm")
      return true;
    if (op_type_ == "matmul" && next->op_type_ == "relu")
      return true;
    return false;
  }

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
// SHAPE INFERENCE ENGINE (Corrected: Methods made Public)
// ============================================================================

class ShapeInference {
public:
  static std::vector<size_t>
  infer_shape(const std::string &op_type,
              const std::vector<TensorDescriptor *> &inputs,
              const OpAttributes &attrs);

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
// GRAPH
// ============================================================================

class Graph {
public:
  Graph() : node_counter_(0), tensor_counter_(0) {}

  TensorDescriptor *add_input(const std::string &name,
                              const std::vector<size_t> &shape,
                              DataType dtype = DataType::FLOAT32);

  TensorDescriptor *add_op(const std::string &op_type,
                           const std::vector<TensorDescriptor *> &inputs,
                           const OpAttributes &attrs = {},
                           const std::string &name = "");

  TensorDescriptor *add_weight(const std::string &name,
                               const std::vector<size_t> &shape,
                               const float *data = nullptr);

  void mark_output(TensorDescriptor *tensor);
  void optimize();
  void fuse_operations();
  void eliminate_dead_code();
  void optimize_memory_layout();

  class ExecutableGraph *compile(Runtime *rt);

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
// EXECUTABLE GRAPH
// ============================================================================

class ExecutableGraph {
public:
  ExecutableGraph(Runtime *rt, const Graph *graph);
  ~ExecutableGraph();

  std::vector<Buffer *> execute(const std::vector<Buffer *> &inputs);
  void execute(const std::vector<Buffer *> &inputs,
               const std::vector<Buffer *> &outputs);
  size_t get_memory_usage() const { return total_memory_bytes_; }

  struct ProfilingData {
    std::string node_name;
    std::string op_type;
    double time_ms;
    std::string backend;
  };
  std::vector<ProfilingData> profile(const std::vector<Buffer *> &inputs);

private:
  Runtime *runtime_;
  std::vector<QueuedOpWithAttrs> execution_plan_;
  std::map<std::string, Buffer *> intermediate_buffers_;
  std::map<std::string, Buffer *> weight_buffers_;
  std::vector<Buffer *> output_buffers_;
  size_t total_memory_bytes_;

  void analyze_liveness(const Graph *graph,
                        std::map<std::string, TensorLiveness> &liveness);
  void allocate_buffers(const Graph *graph);
  void load_weights(const Graph *graph);
  void build_execution_plan(const Graph *graph);
  Buffer *get_or_create_buffer(const std::string &name,
                               const std::vector<size_t> &shape);
};

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