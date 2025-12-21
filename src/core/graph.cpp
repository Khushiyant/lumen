#include "lumen/graph.hpp"
#include <algorithm>
#include <iostream>
#include <set>
#include <sstream>
#include <unordered_set>
#include <cstring>

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
             op_type == "sigmoid" || op_type == "tanh" ||
             op_type == "softmax") {
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
  } else {
    return inputs[0]->shape();
  }
}

std::vector<size_t>
ShapeInference::infer_conv2d(const std::vector<size_t> &input_shape,
                             const std::vector<size_t> &weight_shape,
                             const OpAttributes &attrs) {
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

  size_t M = a_shape[a_shape.size() - 2];
  size_t K_a = a_shape[a_shape.size() - 1];
  size_t K_b = b_shape[b_shape.size() - 2];
  size_t N = b_shape[b_shape.size() - 1];

  if (K_a != K_b) {
    std::stringstream ss;
    ss << "dimension mismatch: A=[" << M << "," << K_a << "] B=[" << K_b << ","
       << N << "]";
    throw ShapeInferenceError(ss.str());
  }

  std::vector<size_t> out_shape;
  for (size_t i = 0; i < a_shape.size() - 2; ++i)
    out_shape.push_back(a_shape[i]);
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
  if (input_shape.size() != 4)
    throw ShapeInferenceError("pool2d expects 4D tensor");

  auto kernel_size = attrs.get_int_array("kernel_size");
  auto stride = attrs.get_int_array("stride");
  auto padding = attrs.get_int_array("padding");

  int k_h = kernel_size.empty() ? 2 : kernel_size[0];
  int k_w = kernel_size.size() > 1 ? kernel_size[1] : k_h;
  int s_h = stride.empty() ? k_h : stride[0];
  int s_w = stride.size() > 1 ? stride[1] : s_h;
  int p_h = padding.empty() ? 0 : padding[0];
  int p_w = padding.size() > 1 ? padding[1] : p_h;

  size_t N = input_shape[0], C = input_shape[1], H = input_shape[2],
         W = input_shape[3];
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
  if (inputs.empty())
    throw GraphException("Operation must have at least one input");

  std::vector<size_t> output_shape;
  if (op_type == "matmul") {
    bool trans_a = attrs.get_int("transA", 0);
    bool trans_b = attrs.get_int("transB", 0);
    std::vector<size_t> shape_a = inputs[0]->shape();
    std::vector<size_t> shape_b = inputs[1]->shape();

    if (trans_a && shape_a.size() >= 2)
      std::swap(shape_a[shape_a.size() - 1], shape_a[shape_a.size() - 2]);
    if (trans_b && shape_b.size() >= 2)
      std::swap(shape_b[shape_b.size() - 1], shape_b[shape_b.size() - 2]);

    try {
      output_shape = ShapeInference::infer_matmul(shape_a, shape_b);
    } catch (const ShapeInferenceError &e) {
      throw GraphException("Failed to infer shape for op 'matmul': " +
                           std::string(e.what()));
    }
  } else {
    try {
      output_shape = ShapeInference::infer_shape(op_type, inputs, attrs);
    } catch (const ShapeInferenceError &e) {
      throw GraphException("Failed to infer shape for op '" + op_type +
                           "': " + std::string(e.what()));
    }
  }

  std::string tensor_name = generate_tensor_name();
  auto output_tensor = std::make_unique<TensorDescriptor>(
      tensor_name, output_shape, inputs[0]->dtype());
  auto *output_ptr = output_tensor.get();
  tensors_.push_back(std::move(output_tensor));

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
  std::cout << "[Graph] Optimization complete. Nodes: " << nodes_.size()
            << std::endl;
}

void Graph::fuse_operations() {
  std::vector<std::unique_ptr<GraphNode>> fused_nodes;
  for (size_t i = 0; i < nodes_.size(); ++i) {
    if (i + 1 < nodes_.size() &&
        nodes_[i]->can_fuse_with(nodes_[i + 1].get())) {
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
        ++i;
        continue;
      }
    }
    fused_nodes.push_back(std::move(nodes_[i]));
  }
  nodes_ = std::move(fused_nodes);
}

void Graph::eliminate_dead_code() {
  std::unordered_set<TensorDescriptor *> live_tensors;
  for (auto *output : outputs_)
    live_tensors.insert(output);
  bool changed = true;
  while (changed) {
    changed = false;
    for (auto &node : nodes_) {
      if (live_tensors.count(node->output())) {
        for (auto *input : node->inputs()) {
          if (live_tensors.insert(input).second)
            changed = true;
        }
      }
    }
  }
  nodes_.erase(std::remove_if(nodes_.begin(), nodes_.end(),
                              [&](const std::unique_ptr<GraphNode> &node) {
                                return !live_tensors.count(node->output());
                              }),
               nodes_.end());
}

void Graph::optimize_memory_layout() {
  std::cout << "[Graph] Memory layout optimization: TODO" << std::endl;
}

ExecutableGraph *Graph::compile(Runtime *rt) {
  if (outputs_.empty())
    throw GraphCompilationError("No output tensors marked");
  return new ExecutableGraph(rt, this);
}

void Graph::print_summary() const {
  std::cout << "\n=== Graph Summary ===\nNodes: " << nodes_.size()
            << " Weights: " << weights_.size() << std::endl;
  for (const auto &node : nodes_) {
    std::cout << "  " << node->name() << " [" << node->op_type()
              << "] output=" << node->output()->name() << std::endl;
  }
}

std::string Graph::to_string() const {
  return "Graph(nodes=" + std::to_string(nodes_.size()) + ")";
}

// ============================================================================
// EXECUTABLE GRAPH IMPLEMENTATION
// ============================================================================

ExecutableGraph::ExecutableGraph(Runtime *rt, const Graph *graph)
    : runtime_(rt), total_memory_bytes_(0) {
  allocate_buffers(graph);
  load_weights(graph);
  size_t total_weight_bytes = 0;
  for (const auto &[name, desc] : graph->weights()) {
    total_weight_bytes += desc->size_bytes();
  }
  total_memory_bytes_ += total_weight_bytes;
  build_execution_plan(graph);
  std::cout << "[ExecutableGraph] Compilation complete. Memory: "
            << (total_memory_bytes_ / 1024.0 / 1024.0) << " MB" << std::endl;
}

ExecutableGraph::~ExecutableGraph() {
  intermediate_buffers_.clear();
  weight_buffers_.clear();
}

void ExecutableGraph::allocate_buffers(const Graph *graph) {
  std::map<std::string, TensorLiveness> liveness_map;
  analyze_liveness(graph, liveness_map);

  std::multimap<size_t, std::shared_ptr<Buffer>> free_pool;
  std::map<std::string, std::shared_ptr<Buffer>> assignments;

  size_t current_footprint = 0;
  total_memory_bytes_ = 0; // This will track the Peak (High Water Mark)

  for (size_t i = 0; i < graph->nodes().size(); ++i) {
    auto &node = graph->nodes()[i];
    std::string out_name = node->output()->name();
    auto &liveness = liveness_map[out_name];

    std::shared_ptr<Buffer> buf;
    auto it = free_pool.lower_bound(liveness.size_bytes);

    if (it != free_pool.end() && it->first < liveness.size_bytes * 2) {
      buf = it->second;
      free_pool.erase(it);
      // Footprint doesn't increase; we reused a buffer
    } else {
      buf = runtime_->alloc(node->output()->shape());
      current_footprint += liveness.size_bytes;
    }

    // Update high-water mark
    if (current_footprint > total_memory_bytes_) {
      total_memory_bytes_ = current_footprint;
    }

    assignments[out_name] = buf;
    intermediate_buffers_[out_name] = buf;

    // Important: Decrease footprint when a buffer is returned to the pool
    for (auto *input : node->inputs()) {
      std::string in_name = input->name();
      if (liveness_map.count(in_name) && liveness_map[in_name].last_use == i) {
        free_pool.insert(
            {liveness_map[in_name].size_bytes, assignments[in_name]});
        current_footprint -= liveness_map[in_name].size_bytes;
      }
    }
  }
}

void ExecutableGraph::analyze_liveness(
    const Graph *graph, std::map<std::string, TensorLiveness> &liveness) {
  const auto &nodes = graph->nodes();
  for (size_t i = 0; i < nodes.size(); ++i) {
    std::string out_name = nodes[i]->output()->name();
    liveness[out_name] = {i, i, nodes[i]->output()->size_bytes(), out_name};
  }
  for (size_t i = 0; i < nodes.size(); ++i) {
    for (auto *input : nodes[i]->inputs()) {
      if (liveness.count(input->name()))
        liveness[input->name()].last_use = i;
    }
  }
  for (auto *graph_output : graph->outputs()) {
    if (liveness.count(graph_output->name()))
      liveness[graph_output->name()].last_use = nodes.size();
  }
}

void ExecutableGraph::load_weights(const Graph *graph) {
  for (const auto &[name, desc] : graph->weights()) {
    // runtime_->alloc returns shared_ptr
    auto buf = runtime_->alloc(desc->shape());
    weight_buffers_[name] = buf;

    // Use buf.get() to access raw data for memcpy
    const auto &all_weight_data = graph->weight_data();
    if (all_weight_data.count(name)) {
      std::memcpy(buf->data(), all_weight_data.at(name).data(),
                  desc->size_bytes());
    }
  }
}

void ExecutableGraph::build_execution_plan(const Graph *graph) {
  for (const auto &node : graph->nodes()) {
    std::vector<std::shared_ptr<Buffer>> input_buffers;
    for (auto *input_desc : node->inputs()) {
      std::string name = input_desc->name();
      if (weight_buffers_.count(name))
        input_buffers.push_back(weight_buffers_[name]);
      else if (intermediate_buffers_.count(name))
        input_buffers.push_back(intermediate_buffers_[name]);
      else
        input_buffers.push_back(nullptr);
    }

    QueuedOpWithAttrs op;
    op.op_name = node->op_type();
    op.inputs = input_buffers;
    op.output = intermediate_buffers_[node->output()->name()];
    op.attrs = node->attrs();
    execution_plan_.push_back(op);
  }
}

std::vector<std::shared_ptr<Buffer>>
ExecutableGraph::execute(const std::vector<std::shared_ptr<Buffer>> &inputs) {
  size_t input_idx = 0;
  for (auto &op : execution_plan_) {
    std::vector<std::shared_ptr<Buffer>> runtime_inputs = op.inputs;
    for (size_t i = 0; i < runtime_inputs.size(); ++i) {
      if (runtime_inputs[i] == nullptr) {
        if (input_idx < inputs.size())
          runtime_inputs[i] = inputs[input_idx++];
      }
    }
    runtime_->execute(op.op_name, runtime_inputs, op.output, op.attrs);
  }
  runtime_->submit();
  runtime_->wait_all();
  return {execution_plan_.back().output};
}

void ExecutableGraph::execute(
    const std::vector<std::shared_ptr<Buffer>> &inputs,
    const std::vector<std::shared_ptr<Buffer>> &outputs) {
  auto result = execute(inputs);
  for (size_t i = 0; i < outputs.size() && i < result.size(); ++i) {
    std::memcpy(outputs[i]->data(), result[i]->data(), result[i]->size_bytes());
  }
}

// lumen/src/core/graph.cpp

std::vector<ExecutableGraph::ProfilingData>
ExecutableGraph::profile(const std::vector<std::shared_ptr<Buffer>> &inputs) {
  std::vector<ProfilingData> results;
  size_t input_idx = 0;
  for (auto &op : execution_plan_) {
    // Correctly define actual_inputs as a vector of shared_ptrs
    std::vector<std::shared_ptr<Buffer>> actual_inputs = op.inputs;

    for (size_t i = 0; i < actual_inputs.size(); ++i) {
      if (actual_inputs[i] == nullptr) {
        // This assignment now works because both sides are shared_ptr<Buffer>
        actual_inputs[i] = inputs[input_idx++];
      }
    }

    auto start = std::chrono::high_resolution_clock::now();

    // Pass the correct shared_ptr vector to runtime
    runtime_->execute(op.op_name, actual_inputs, op.output, op.attrs);

    runtime_->submit();
    runtime_->wait_all();
    auto end = std::chrono::high_resolution_clock::now();

    ProfilingData data;
    data.node_name = op.op_name;
    data.op_type = op.op_name;
    data.time_ms =
        std::chrono::duration<double, std::milli>(end - start).count();
    data.backend = runtime_->current_backend();
    results.push_back(data);
  }
  return results;
}

} // namespace lumen