#include <iostream>
#include <lumen/lumen.hpp>
#include <lumen/onnx_importer.hpp>

void test_onnx_loading() {
  lumen::Runtime rt;

  // 1. Import from file
  auto graph = lumen::ONNXImporter::import_model("./models/mnist.onnx");

  // 2. Run your existing optimization passes
  graph->optimize(); // Fuses ops, removes dead code

  // 3. Compile for execution
  auto *executable = graph->compile(&rt);

  // 4. Run inference
  auto input_buf = rt.alloc({1, 1, 28, 28});
  auto outputs = executable->execute({input_buf});

  std::cout << "Successfully ran inference on ONNX model!" << std::endl;

  delete executable;
  // input_buf deleted automatically
}