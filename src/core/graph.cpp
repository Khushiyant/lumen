#include "lumen/graph.hpp"
#include <algorithm>
#include <iostream>
#include <set>
#include <sstream>
#include <unordered_set>

namespace lumen {

// ============================================================================
// SHAPE INFERENCE IMPLEMENTATION
// ============================================================================

std::vector<size_t>
ShapeInference::infer_shape(const std::string &op_type,
                            const std::vector<TensorDescriptor *> &inputs,
                            const OpAttributes &attrs) {

  if (op_type == "conv2d") {
    if (inputs.size() < 2)
      throw ShapeInferenceError("conv2d requires at least 2 inputs");
    return infer_conv2d(inputs[0]->shape(), inputs[1]->shape(), attrs);
  } else if (op_type == "matmul") {
    if (inputs.size() < 2)
      throw ShapeInferenceError("matmul requires 2 inputs");
    return infer_matmul(inputs[0]->shape(), inputs[1]->shape());
  } else if (op_type == "add" || op_type == "mul" || op_type == "relu" ||
             op_type == "sigmoid" || op_type == "tanh") {
    return infer_elementwise(inputs[0]->shape());
  } else if (op_type == "maxpool2d" || op_type == "avgpool2d") {
    return infer_pool2d(inputs[0]->shape(), attrs);
  } else if (op_type == "batchnorm") {
    return infer_batchnorm(inputs[0]->shape());
  } else if (op_type == "reshape") {
    auto shape_ints = attrs.get_int_array("shape");
    std::vector<size_t> shape(shape_ints.begin(), shape_ints.end());
    return shape;
  } else if (op_type == "flatten") {
    auto &in_shape = inputs[0]->shape();
    size_t total = 1;
    for (size_t i = 1; i < in_shape.size(); ++i)
      total *= in_shape[i];
    return {in_shape[0], total};
  } else if (op_type == "softmax") {
    return inputs[0]->shape();
  } else {
    // Default: keep same shape as input
    return inputs[0]->shape();
  }
}

std::vector<size_t>
ShapeInference::infer_conv2d(const std::vector<size_t> &input_shape,
                             const std::vector<size_t> &weight_shape,
                             const OpAttributes &attrs) {

  // Input: [N, C_in, H, W]
  // Weight: [C_out, C_in, K_h, K_w]
  // Output: [N, C_out, H_out, W_out]

  if (input_shape.size() != 4 || weight_shape.size() != 4) {
    throw ShapeInferenceError("conv2d expects 4D tensors");
  }

  auto stride = attrs.get_int_array("stride");
  auto padding = attrs.get_int_array("padding");
  auto dilation = attrs.get_int_array("dilation");

  int stride_h = stride.empty() ? 1 : stride[0];
  int stride_w = stride.size() > 1 ? stride[1] : stride_h;

  int pad_h = padding.empty() ? 0 : padding[0];
  int pad_w = padding.size() > 1 ? padding[1] : pad_h;

  int dil_h = dilation.empty() ? 1 : dilation[0];
  int dil_w = dilation.size() > 1 ? dilation[1] : dil_h;

  size_t N = input_shape[0];
  size_t C_out = weight_shape[0];
  size_t H = input_shape[2];
  size_t W = input_shape[3];
  size_t K_h = weight_shape[2];
  size_t K_w = weight_shape[3];

  size_t H_out = (H + 2 * pad_h - dil_h * (K_h - 1) - 1) / stride_h + 1;
  size_t W_out = (W + 2 * pad_w - dil_w * (K_w - 1) - 1) / stride_w + 1;

  return {N, C_out, H_out, W_out};
}

std::vector<size_t>
ShapeInference::infer_matmul(const std::vector<size_t> &a_shape,
                             const std::vector<size_t> &b_shape) {

  if (a_shape.size() < 2 || b_shape.size() < 2) {
    throw ShapeInferenceError("matmul requires at least 2D tensors");
  }

  // A: [..., M, K]
  // B: [..., K, N]
  // C: [..., M, N]

  size_t M = a_shape[a_shape.size() - 2];
  size_t K_a = a_shape[a_shape.size() - 1];
  size_t K_b = b_shape[b_shape.size() - 2];
  size_t N = b_shape[b_shape.size() - 1];

  if (K_a != K_b) {
    throw ShapeInferenceError("matmul dimension mismatch");
  }

  // Handle batch dimensions (simple version: broadcast not implemented)
  std::vector<size_t> out_shape;
  for (size_t i = 0; i < a_shape.size() - 2; ++i) {
    out_shape.push_back(a_shape[i]);
  }
  out_shape.push_back(M);
  out_shape.push_back(N);

  return out_shape;
}

std::vector<size_t>
ShapeInference::infer_elementwise(const std::vector<size_t> &shape) {
  return shape;
}

std::vector<size_t>
ShapeInference::infer_pool2d(const std::vector<size_t> &input_shape,
                             const OpAttributes &attrs) {

  // Input: [N, C, H, W]
  // Output: [N, C, H_out, W_out]

  if (input_shape.size() != 4) {
    throw ShapeInferenceError("pool2d expects 4D tensor");
  }

  auto kernel_size = attrs.get_int_array("kernel_size");
  auto stride = attrs.get_int_array("stride");
  auto padding = attrs.get_int_array("padding");

  int k_h = kernel_size.empty() ? 2 : kernel_size[0];
  int k_w = kernel_size.size() > 1 ? kernel_size[1] : k_h;

  int s_h = stride.empty() ? k_h : stride[0];
  int s_w = stride.size() > 1 ? stride[1] : s_h;

  int p_h = padding.empty() ? 0 : padding[0];
  int p_w = padding.size() > 1 ? padding[1] : p_h;

  size_t N = input_shape[0];
  size_t C = input_shape[1];
  size_t H = input_shape[2];
  size_t W = input_shape[3];

  size_t H_out = (H + 2 * p_h - k_h) / s_h + 1;
  size_t W_out = (W + 2 * p_w - k_w) / s_w + 1;

  return {N, C, H_out, W_out};
}

std::vector<size_t>
ShapeInference::infer_batchnorm(const std::vector<size_t> &input_shape) {
  return input_shape;
}

// ============================================================================
// GRAPH IMPLEMENTATION
// ============================================================================

TensorDescriptor *Graph::add_input(const std::string &name,
                                   const std::vector<size_t> &shape,
                                   DataType dtype) {
  auto tensor = std::make_unique<TensorDescriptor>(name, shape, dtype);
  auto *ptr = tensor.get();
  tensors_.push_back(std::move(tensor));
  inputs_.push_back(ptr);
  return ptr;
}

TensorDescriptor *Graph::add_op(const std::string &op_type,
                                const std::vector<TensorDescriptor *> &inputs,
                                const OpAttributes &attrs,
                                const std::string &name) {

  if (inputs.empty()) {
    throw GraphException("Operation must have at least one input");
  }

  // Infer output shape
  std::vector<size_t> output_shape;
  try {
    output_shape = ShapeInference::infer_shape(op_type, inputs, attrs);
  } catch (const ShapeInferenceError &e) {
    throw GraphException("Failed to infer shape for op '" + op_type +
                         "': " + e.what());
  }

  // Create output tensor
  std::string tensor_name = generate_tensor_name();
  auto output_tensor = std::make_unique<TensorDescriptor>(
      tensor_name, output_shape, inputs[0]->dtype());
  auto *output_ptr = output_tensor.get();
  tensors_.push_back(std::move(output_tensor));

  // Create node
  std::string node_name = name.empty() ? generate_node_name() : name;
  auto node = std::make_unique<GraphNode>(node_name, op_type, inputs,
                                          output_ptr, attrs);
  nodes_.push_back(std::move(node));

  return output_ptr;
}

TensorDescriptor *Graph::add_weight(const std::string &name,
                                    const std::vector<size_t> &shape,
                                    const float *data) {
  auto tensor = std::make_unique<TensorDescriptor>(name, shape);
  auto *ptr = tensor.get();
  tensors_.push_back(std::move(tensor));
  weights_[name] = ptr;

  // Store weight data if provided
  if (data) {
    size_t num_elements = ptr->num_elements();
    weight_data_[name] = std::vector<float>(data, data + num_elements);
  }

  return ptr;
}

void Graph::mark_output(TensorDescriptor *tensor) {
  outputs_.push_back(tensor);
}

void Graph::optimize() {
  std::cout << "[Graph] Running optimization passes..." << std::endl;

  fuse_operations();
  eliminate_dead_code();
  optimize_memory_layout();

  std::cout << "[Graph] Optimization complete. Nodes: " << nodes_.size()
            << std::endl;
}

void Graph::fuse_operations() {
  // Simple fusion: Conv2d + ReLU, Matmul + ReLU, etc.
  std::vector<std::unique_ptr<GraphNode>> fused_nodes;

  for (size_t i = 0; i < nodes_.size(); ++i) {
    if (i + 1 < nodes_.size() &&
        nodes_[i]->can_fuse_with(nodes_[i + 1].get())) {

      // Check if output of node[i] is only used by node[i+1]
      auto *intermediate = nodes_[i]->output();
      bool single_use = true;
      for (size_t j = i + 2; j < nodes_.size(); ++j) {
        for (auto *input : nodes_[j]->inputs()) {
          if (input == intermediate) {
            single_use = false;
            break;
          }
        }
      }

      if (single_use) {
        // Fuse: Create new node with combined operation
        std::string fused_op = nodes_[i]->get_fused_name(nodes_[i + 1].get());
        std::string fused_name =
            nodes_[i]->name() + "_" + nodes_[i + 1]->name();

        auto fused_node = std::make_unique<GraphNode>(
            fused_name, fused_op, nodes_[i]->inputs(), nodes_[i + 1]->output(),
            nodes_[i]->attrs());

        fused_nodes.push_back(std::move(fused_node));

        std::cout << "[Graph] Fused: " << nodes_[i]->op_type() << " + "
                  << nodes_[i + 1]->op_type() << " -> " << fused_op
                  << std::endl;

        ++i; // Skip next node as it's fused
        continue;
      }
    }

    fused_nodes.push_back(std::move(nodes_[i]));
  }

  nodes_ = std::move(fused_nodes);
}

void Graph::eliminate_dead_code() {
  // Mark all tensors that are reachable from outputs
  std::unordered_set<TensorDescriptor *> live_tensors;

  for (auto *output : outputs_) {
    live_tensors.insert(output);
  }

  // Backward pass: mark all nodes that contribute to outputs
  bool changed = true;
  while (changed) {
    changed = false;
    for (auto &node : nodes_) {
      if (live_tensors.count(node->output())) {
        for (auto *input : node->inputs()) {
          if (live_tensors.insert(input).second) {
            changed = true;
          }
        }
      }
    }
  }

  // Remove dead nodes
  nodes_.erase(std::remove_if(nodes_.begin(), nodes_.end(),
                              [&](const std::unique_ptr<GraphNode> &node) {
                                return !live_tensors.count(node->output());
                              }),
               nodes_.end());
}

void Graph::optimize_memory_layout() {
  // Placeholder: Could implement memory reuse analysis here
  // For now, just log that we're considering it
  std::cout << "[Graph] Memory layout optimization: TODO" << std::endl;
}

ExecutableGraph *Graph::compile(Runtime *rt) {
  std::cout << "[Graph] Compiling graph..." << std::endl;

  if (outputs_.empty()) {
    throw GraphCompilationError("No output tensors marked");
  }

  return new ExecutableGraph(rt, this);
}

void Graph::print_summary() const {
  std::cout << "\n=== Graph Summary ===" << std::endl;
  std::cout << "Inputs: " << inputs_.size() << std::endl;
  std::cout << "Outputs: " << outputs_.size() << std::endl;
  std::cout << "Nodes: " << nodes_.size() << std::endl;
  std::cout << "Weights: " << weights_.size() << std::endl;

  std::cout << "\nNodes:" << std::endl;
  for (const auto &node : nodes_) {
    std::cout << "  " << node->name() << " [" << node->op_type() << "]";
    std::cout << " inputs=" << node->inputs().size();
    std::cout << " output_shape=[";
    for (size_t i = 0; i < node->output()->shape().size(); ++i) {
      if (i > 0)
        std::cout << ",";
      std::cout << node->output()->shape()[i];
    }
    std::cout << "]" << std::endl;
  }
  std::cout << "=====================\n" << std::endl;
}

std::string Graph::to_string() const {
  std::stringstream ss;
  ss << "Graph(nodes=" << nodes_.size() << ", inputs=" << inputs_.size()
     << ", outputs=" << outputs_.size() << ")";
  return ss.str();
}

// ============================================================================
// EXECUTABLE GRAPH IMPLEMENTATION
// ============================================================================

ExecutableGraph::ExecutableGraph(Runtime *rt, const Graph *graph)
    : runtime_(rt), total_memory_bytes_(0) {

  allocate_buffers(graph);
  load_weights(graph);
  build_execution_plan(graph);

  std::cout << "[ExecutableGraph] Compilation complete. Memory: "
            << (total_memory_bytes_ / 1024.0 / 1024.0) << " MB" << std::endl;
}

ExecutableGraph::~ExecutableGraph() {
  // Use a set to collect unique physical buffers.
  // This prevents double-freeing when multiple tensors share a buffer.
  std::set<Buffer *> unique_buffers;

  // 1. Collect from intermediate buffers
  for (auto const &[name, buf] : intermediate_buffers_) {
    if (buf)
      unique_buffers.insert(buf);
  }

  // 2. Collect from weight buffers
  for (auto const &[name, buf] : weight_buffers_) {
    if (buf)
      unique_buffers.insert(buf);
  }

  // 3. Delete each unique buffer exactly once
  for (Buffer *buf : unique_buffers) {
    delete buf;
  }

  // Clear maps to prevent dangling pointers
  intermediate_buffers_.clear();
  weight_buffers_.clear();
}

void ExecutableGraph::allocate_buffers(const Graph *graph) {
  std::map<std::string, TensorLiveness> liveness_map;
  analyze_liveness(graph, liveness_map);

  // Pool of currently "free" buffers: size -> list of buffers
  std::multimap<size_t, Buffer *> free_pool;

  // Track which buffer is assigned to which tensor name
  std::map<std::string, Buffer *> assignments;

  const auto &nodes = graph->nodes();
  for (size_t i = 0; i < nodes.size(); ++i) {
    std::string out_name = nodes[i]->output()->name();
    auto &liveness = liveness_map[out_name];

    // 1. Try to reuse a buffer from the pool
    Buffer *reused_buf = nullptr;
    auto it = free_pool.lower_bound(liveness.size_bytes);

    // Find a buffer that is large enough but not excessively oversized (e.g., <
    // 2x)
    if (it != free_pool.end() && it->first < liveness.size_bytes * 2) {
      reused_buf = it->second;
      free_pool.erase(it);
    } else {
      // No suitable buffer found, allocate a new one
      reused_buf = runtime_->alloc(nodes[i]->output()->shape());
      total_memory_bytes_ += liveness.size_bytes;
    }

    assignments[out_name] = reused_buf;
    intermediate_buffers_[out_name] = reused_buf;

    // 2. Check if any inputs "die" after this operation
    for (auto *input : nodes[i]->inputs()) {
      std::string in_name = input->name();
      if (liveness_map.count(in_name) && liveness_map[in_name].last_use == i) {
        // Return buffer to pool for future ops
        free_pool.insert(
            {liveness_map[in_name].size_bytes, assignments[in_name]});
      }
    }
  }

  std::cout << "[MemoryPlanner] Optimized peak memory: "
            << (total_memory_bytes_ / 1024.0 / 1024.0) << " MB" << std::endl;
}
void ExecutableGraph::analyze_liveness(
    const Graph *graph, std::map<std::string, TensorLiveness> &liveness) {
  const auto &nodes = graph->nodes();

  // 1. Initial pass: Mark production (first_use)
  for (size_t i = 0; i < nodes.size(); ++i) {
    std::string out_name = nodes[i]->output()->name();
    liveness[out_name] = {i, i, nodes[i]->output()->size_bytes(), out_name};
  }

  // 2. Second pass: Update last_use based on consumers
  for (size_t i = 0; i < nodes.size(); ++i) {
    for (auto *input : nodes[i]->inputs()) {
      if (liveness.count(input->name())) {
        liveness[input->name()].last_use = i;
      }
    }
  }

  // 3. Tensors marked as Graph Outputs must live until the very end
  for (auto *graph_output : graph->outputs()) {
    if (liveness.count(graph_output->name())) {
      liveness[graph_output->name()].last_use = nodes.size();
    }
  }
}

void ExecutableGraph::load_weights(const Graph *graph) {
  // Load weight data into buffers
  for (const auto &[name, desc] : graph->weights()) {
    auto *buf = runtime_->alloc(desc->shape());
    weight_buffers_[name] = buf;
    total_memory_bytes_ += desc->size_bytes();

    // Copy weight data if available
    const auto &all_weight_data = graph->weight_data();
    if (all_weight_data.find(name) != all_weight_data.end()) {
      const auto &data = all_weight_data.at(name);
      std::memcpy(buf->data(), data.data(), data.size() * sizeof(float));
    }
  }
}

void ExecutableGraph::build_execution_plan(const Graph *graph) {
  // Convert graph nodes to QueuedOpWithAttrs
  for (const auto &node : graph->nodes()) {
    std::vector<Buffer *> input_buffers;

    // Map input tensors to buffers
    for (auto *input_desc : node->inputs()) {
      std::string name = input_desc->name();

      // Check if it's a weight
      if (weight_buffers_.find(name) != weight_buffers_.end()) {
        input_buffers.push_back(weight_buffers_[name]);
      }
      // Check if it's an intermediate
      else if (intermediate_buffers_.find(name) !=
               intermediate_buffers_.end()) {
        input_buffers.push_back(intermediate_buffers_[name]);
      } else {
        // It's a graph input (will be provided at runtime)
        input_buffers.push_back(nullptr);
      }
    }

    // Get output buffer
    Buffer *output_buffer = intermediate_buffers_[node->output()->name()];

    // Create QueuedOpWithAttrs
    QueuedOpWithAttrs op;
    op.op_name = node->op_type();
    op.inputs = input_buffers;
    op.output = output_buffer;
    op.attrs = node->attrs();
    op.target_backend = ""; // Will be selected by router

    execution_plan_.push_back(op);
  }
}

Buffer *
ExecutableGraph::get_or_create_buffer(const std::string &name,
                                      const std::vector<size_t> &shape) {

  if (intermediate_buffers_.find(name) != intermediate_buffers_.end()) {
    return intermediate_buffers_[name];
  }

  auto *buf = runtime_->alloc(shape);
  intermediate_buffers_[name] = buf;
  return buf;
}

// lumen/src/core/graph.cpp

std::vector<Buffer *>
ExecutableGraph::execute(const std::vector<Buffer *> &inputs) {
  size_t input_idx = 0;

  // Iterate through the plan and inject provided buffers where placeholders
  // (nullptr) exist
  for (auto &op : execution_plan_) {
    std::vector<Buffer *> runtime_inputs = op.inputs;

    for (size_t i = 0; i < runtime_inputs.size(); ++i) {
      if (runtime_inputs[i] == nullptr) {
        if (input_idx < inputs.size()) {
          runtime_inputs[i] = inputs[input_idx++];
        } else {
          throw GraphException(
              "Not enough input buffers provided for execution.");
        }
      }
    }

    // Call execute exactly ONCE with the resolved buffers and attributes
    runtime_->execute(op.op_name, runtime_inputs, op.output, op.attrs);
  }

  // Submit the entire plan to the hardware and wait for completion
  runtime_->submit();
  runtime_->wait_all();

  // Return the output buffers (the final op's output)
  if (!execution_plan_.empty()) {
    return {execution_plan_.back().output};
  }
  return {};
}

void ExecutableGraph::execute(const std::vector<Buffer *> &inputs,
                              const std::vector<Buffer *> &outputs) {
  // Custom output version - similar to above but uses provided output buffers
  auto result = execute(inputs);

  for (size_t i = 0; i < outputs.size() && i < result.size(); ++i) {
    // Copy result to provided output buffer
    // (In production, would avoid this copy by using output buffers directly)
    std::memcpy(outputs[i]->data(), result[i]->data(), result[i]->size_bytes());
  }
}

// lumen/src/core/graph.cpp

std::vector<ExecutableGraph::ProfilingData>
ExecutableGraph::profile(const std::vector<Buffer *> &inputs) {
  std::vector<ProfilingData> results;
  size_t input_idx = 0;

  for (auto &op : execution_plan_) {
    // Resolve inputs for this specific run without modifying op.inputs
    std::vector<Buffer *> actual_inputs = op.inputs;
    for (size_t i = 0; i < actual_inputs.size(); ++i) {
      if (actual_inputs[i] == nullptr) {
        actual_inputs[i] = inputs[input_idx++];
      }
    }

    auto start = std::chrono::high_resolution_clock::now();

    // Use the resolved inputs
    runtime_->execute(op.op_name, actual_inputs, op.output, op.attrs);
    runtime_->submit();
    runtime_->wait_all();

    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();

    ProfilingData data;
    data.node_name = op.op_name;
    data.op_type = op.op_name;
    data.time_ms = ms;
    data.backend = runtime_->current_backend();
    results.push_back(data);
  }

  return results;
}

} // namespace lumen