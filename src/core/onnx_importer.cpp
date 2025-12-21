#include "lumen/onnx_importer.hpp"
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

std::unique_ptr<Graph>
ONNXImporter::import_model(const std::string &model_path) {
  auto lumen_graph = std::make_unique<Graph>();

  // 1. Load ONNX Model using Protobuf
  onnx::ModelProto model;
  std::ifstream in(model_path, std::ios::ate | std::ios::binary);
  if (!in.is_open()) {
    throw GraphException("Could not open ONNX model: " + model_path);
  }

  std::streamsize size = in.tellg();
  in.seekg(0, std::ios::beg);
  if (!model.ParseFromIstream(&in)) {
    throw GraphException("Failed to parse ONNX protobuf.");
  }

  const auto &onnx_graph = model.graph();
  std::map<std::string, TensorDescriptor *> tensor_map;

  // 2. Import Initializers (Weights)
  for (const auto &initializer : onnx_graph.initializer()) {
    std::vector<size_t> shape;
    for (auto dim : initializer.dims()) {
      shape.push_back(static_cast<size_t>(dim));
    }

    // Handle raw data or float_data depending on ONNX storage
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

  // 3. Import Inputs (excluding those that are weights)
  for (const auto &input : onnx_graph.input()) {
    if (tensor_map.count(input.name()))
      continue;

    std::vector<size_t> shape;
    const auto &type = input.type().tensor_type();
    for (const auto &dim : type.shape().dim()) {
      // ONNX may use -1 for dynamic dims; default to 1 for static runtime
      shape.push_back(dim.has_dim_value() ? dim.dim_value() : 1);
    }

    auto *t = lumen_graph->add_input(input.name(), shape,
                                     map_dtype(type.elem_type()));
    tensor_map[input.name()] = t;
  }

  // 4. Import Nodes (Operations)
  for (const auto &node : onnx_graph.node()) {
    std::vector<TensorDescriptor *> inputs;
    for (const auto &input_name : node.input()) {
      if (tensor_map.count(input_name)) {
        inputs.push_back(tensor_map[input_name]);
      }
    }

    // Map Attributes
    OpAttributes attrs;
    for (const auto &attr : node.attribute()) {
      if (attr.has_i())
        attrs.int_attrs[attr.name()] = attr.i();
      else if (attr.has_f())
        attrs.float_attrs[attr.name()] = attr.f();
      else if (attr.has_s())
        attrs.string_attrs[attr.name()] = attr.s();
      else if (attr.ints_size() > 0) {
        std::vector<int> vals;
        for (auto v : attr.ints())
          vals.push_back(static_cast<int>(v));
        attrs.int_array_attrs[attr.name()] = vals;
      }
    }

    // Lowercase op_type to match Lumen's internal registry (e.g., "Relu" ->
    // "relu")
    std::string op_type = node.op_type();
    for (auto &c : op_type)
      c = tolower(c);

    auto *output_tensor =
        lumen_graph->add_op(op_type, inputs, attrs, node.name());

    // Register the first output of the node in the map
    if (node.output_size() > 0) {
      tensor_map[node.output(0)] = output_tensor;
    }
  }

  // 5. Mark Outputs
  for (const auto &output : onnx_graph.output()) {
    if (tensor_map.count(output.name())) {
      lumen_graph->mark_output(tensor_map[output.name()]);
    }
  }

  std::cout << "[ONNX] Successfully imported model from: " << model_path
            << std::endl;
  return lumen_graph;
}

} // namespace lumen