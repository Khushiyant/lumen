#include <iostream>
#include <lumen/graph.hpp>
#include <lumen/lumen.hpp>

// Example 1: Simple Linear Model (y = Wx + b)
void example_linear_model() {
  std::cout << "\n=== Example 1: Linear Model ===" << std::endl;

  lumen::Runtime rt;
  lumen::Graph graph;

  auto *input = graph.add_input("input", {1, 10});
  auto *weight = graph.add_weight("weight", {10, 5});
  auto *bias = graph.add_weight("bias", {5});

  auto *matmul_out = graph.add_op("matmul", {input, weight});
  auto *add_out = graph.add_op("add", {matmul_out, bias});

  graph.mark_output(add_out);
  graph.optimize();
  auto *executable = graph.compile(&rt);

  auto *input_buffer = rt.alloc({1, 10});
  float *input_data = (float *)input_buffer->data();
  for (int i = 0; i < 10; i++)
    input_data[i] = 1.0f;

  auto outputs = executable->execute({input_buffer});

  std::cout << "Output shape: [";
  for (auto d : outputs[0]->shape())
    std::cout << d << " ";
  std::cout << "]" << std::endl;

  delete executable;
  delete input_buffer;
}

// Example 2: Conv2d + ReLU (with fusion)
void example_conv_relu() {
  std::cout << "\n=== Example 2: Conv2d + ReLU (Fusion) ===" << std::endl;

  lumen::Runtime rt;
  lumen::Graph graph;

  auto *input = graph.add_input("input", {1, 3, 224, 224});
  auto *weight = graph.add_weight("conv_weight", {64, 3, 3, 3});

  lumen::OpAttributes conv_attrs;
  conv_attrs.int_array_attrs["stride"] = {1, 1};
  conv_attrs.int_array_attrs["padding"] = {1, 1};

  auto *conv_out = graph.add_op("conv2d", {input, weight}, conv_attrs);
  auto *relu_out = graph.add_op("relu", {conv_out});

  graph.mark_output(relu_out);
  graph.optimize();

  auto *executable = graph.compile(&rt);
  std::cout << "Compilation successful! Memory usage: "
            << (executable->get_memory_usage() / 1024.0 / 1024.0) << " MB"
            << std::endl;

  delete executable;
}

int main() {
  std::cout << "=================================================="
            << std::endl;
  std::cout << "Lumen Graph IR Examples" << std::endl;
  std::cout << "=================================================="
            << std::endl;

  example_linear_model();
  example_conv_relu();

  std::cout << "\n=================================================="
            << std::endl;
  std::cout << "All examples completed successfully!" << std::endl;
  std::cout << "=================================================="
            << std::endl;

  return 0;
}