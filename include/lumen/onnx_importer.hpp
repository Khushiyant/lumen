#pragma once
#include <lumen/graph.hpp>
#include <memory>
#include <string>

namespace lumen {

class ONNXImporter {
public:
  /**
   * @brief Parses an ONNX model file and constructs a Lumen Graph.
   * @param model_path Path to the .onnx file.
   * @return A unique_ptr to the constructed Graph.
   * @throws GraphException if parsing fails or an unsupported op is found.
   */
  static std::unique_ptr<Graph> import_model(const std::string &model_path);
};

} // namespace lumen