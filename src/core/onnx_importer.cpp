#include "lumen/onnx_importer.hpp"
#include <algorithm>
#include <fstream>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <iostream>
#include <map>
#include <onnx/onnx_pb.h>

namespace lumen {

// Helper to convert ONNX DataType to Lumen DataType
DataType map_dtype(int onnx_type) {
  switch (onnx_type) {
  case onnx::TensorProto_DataType_FLOAT:
    return DataType::FLOAT32;
  case onnx::TensorProto_DataType_FLOAT16:
    return DataType::FLOAT16;
  case onnx::TensorProto_DataType_INT8:
    return DataType::INT8;
  case onnx::TensorProto_DataType_INT32:
    return DataType::INT32;
  case onnx::TensorProto_DataType_INT64:
    return DataType::INT64;
  case onnx::TensorProto_DataType_BOOL:
    return DataType::BOOL;
  default:
    return DataType::FLOAT32;
  }
}

// Helper to map ONNX Op Types to Lumen Op Codes
std::string map_onnx_op(const std::string &onnx_op) {
  if (onnx_op == "Gemm" || onnx_op == "MatMul")
    return "matmul";
  if (onnx_op == "Conv")
    return "conv2d";
  if (onnx_op == "Relu")
    return "relu";
  if (onnx_op == "GlobalAveragePool")
    return "global_average_pool";
  if (onnx_op == "AveragePool")
    return "avgpool2d";
  if (onnx_op == "MaxPool")
    return "maxpool2d";
  if (onnx_op == "Softmax")
    return "softmax";
  if (onnx_op == "Flatten")
    return "flatten";
  if (onnx_op == "Reshape")
    return "reshape";
  if (onnx_op == "BatchNormalization")
    return "batchnorm";
  if (onnx_op == "Add")
    return "add";
  if (onnx_op == "Mul")
    return "mul";
  if (onnx_op == "Sigmoid")
    return "sigmoid";
  if (onnx_op == "Tanh")
    return "tanh";

  std::string lower = onnx_op;
  std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
  return lower;
}

std::unique_ptr<Graph>
ONNXImporter::import_model(const std::string &model_path) {
  auto lumen_graph = std::make_unique<Graph>();

  onnx::ModelProto model;
  std::ifstream in(model_path, std::ios::ate | std::ios::binary);
  if (!in.is_open()) {
    throw GraphException("Could not open ONNX model: " + model_path);
  }

  in.seekg(0, std::ios::beg);
  if (!model.ParseFromIstream(&in)) {
    throw GraphException("Failed to parse ONNX protobuf.");
  }

  const auto &onnx_graph = model.graph();
  std::map<std::string, TensorDescriptor *> tensor_map;

  // 1. Import Initializers (Weights)
  for (const auto &initializer : onnx_graph.initializer()) {
    std::vector<size_t> shape;
    for (auto dim : initializer.dims()) {
      shape.push_back(static_cast<size_t>(dim));
    }

    const float *data_ptr = nullptr;
    std::vector<float> data_vec;

    if (initializer.has_raw_data()) {
      data_ptr = reinterpret_cast<const float *>(initializer.raw_data().data());
    } else {
      for (int i = 0; i < initializer.float_data_size(); ++i) {
        data_vec.push_back(initializer.float_data(i));
      }
      data_ptr = data_vec.data();
    }

    auto *t = lumen_graph->add_weight(initializer.name(), shape, data_ptr);
    tensor_map[initializer.name()] = t;
  }

  // 2. Import Inputs
  for (const auto &input : onnx_graph.input()) {
    if (tensor_map.count(input.name()))
      continue;

    std::vector<size_t> shape;
    const auto &type = input.type().tensor_type();
    for (const auto &dim : type.shape().dim()) {
      shape.push_back(dim.has_dim_value() ? dim.dim_value() : 1);
    }

    auto *t = lumen_graph->add_input(input.name(), shape,
                                     map_dtype(type.elem_type()));
    tensor_map[input.name()] = t;
  }

  // 3. Import Nodes
  for (const auto &node : onnx_graph.node()) {
    std::vector<TensorDescriptor *> inputs;
    for (const auto &input_name : node.input()) {
      if (tensor_map.count(input_name)) {
        inputs.push_back(tensor_map[input_name]);
      }
    }

    OpAttributes attrs;
    for (const auto &attr : node.attribute()) {
      // Map ONNX attribute names to Lumen expected names
      std::string lumen_attr_name = attr.name();
      if (lumen_attr_name == "pads")
        lumen_attr_name = "padding";
      else if (lumen_attr_name == "strides")
        lumen_attr_name = "stride";
      else if (lumen_attr_name == "dilations")
        lumen_attr_name = "dilation";
      else if (lumen_attr_name == "kernel_shape")
        lumen_attr_name = "kernel_size";

      if (attr.has_i())
        attrs.int_attrs[lumen_attr_name] = attr.i();
      else if (attr.has_f())
        attrs.float_attrs[lumen_attr_name] = attr.f();
      else if (attr.has_s())
        attrs.string_attrs[lumen_attr_name] = attr.s();
      else if (attr.ints_size() > 0) {
        std::vector<int> vals;
        for (auto v : attr.ints())
          vals.push_back(static_cast<int>(v));
        attrs.int_array_attrs[lumen_attr_name] = vals;
      }
    }

    std::string op_type = map_onnx_op(node.op_type());

    if ((op_type == "matmul" || op_type == "conv2d") && inputs.size() > 2) {
      // 1. Add the core operation with first two inputs (Data, Weights)
      std::vector<TensorDescriptor *> core_inputs = {inputs[0], inputs[1]};
      auto *core_output = lumen_graph->add_op(op_type, core_inputs, attrs,
                                              node.name() + "_core");

      // 2. Add an explicit 'add' operation for the bias (input[2])
      // Broadcasting implemented in Step 1 will handle the bias shape
      // automatically.
      auto *final_output =
          lumen_graph->add_op("add", {core_output, inputs[2]}, {}, node.name());

      if (node.output_size() > 0) {
        tensor_map[node.output(0)] = final_output;
      }
    } else {
      // Standard path for ops without extra bias inputs
      auto *output_tensor =
          lumen_graph->add_op(op_type, inputs, attrs, node.name());
      if (node.output_size() > 0) {
        tensor_map[node.output(0)] = output_tensor;
      }
    }

    auto *output_tensor =
        lumen_graph->add_op(op_type, inputs, attrs, node.name());

    if (node.output_size() > 0) {
      tensor_map[node.output(0)] = output_tensor;
    }
  }

  // 4. Mark Outputs
  for (const auto &output : onnx_graph.output()) {
    if (tensor_map.count(output.name())) {
      lumen_graph->mark_output(tensor_map[output.name()]);
    }
  }

  std::cout << "[ONNX] Successfully imported model: " << model_path
            << std::endl;
  return lumen_graph;
}

} // namespace lumen