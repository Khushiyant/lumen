#include <iostream>
#include <lumen/graph.hpp>
#include <lumen/lumen.hpp>

// Example 1: Simple Linear Model (y = Wx + b)
void example_linear_model() {
  std::cout << "\n=== Example 1: Linear Model ===" << std::endl;

  lumen::Runtime rt;
  lumen::Graph graph;

  // Build graph
  auto *input = graph.add_input("input", {1, 10});    // Batch=1, Features=10
  auto *weight = graph.add_weight("weight", {10, 5}); // 10 inputs → 5 outputs
  auto *bias = graph.add_weight("bias", {5});

  // Operations: y = matmul(input, weight) + bias
  auto *matmul_out = graph.add_op("matmul", {input, weight});
  auto *add_out = graph.add_op("add", {matmul_out, bias});

  graph.mark_output(add_out);
  graph.print_summary();

  // Compile
  graph.optimize();
  auto *executable = graph.compile(&rt);

  // Execute
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

  // Input: 1x3x224x224 (Batch, Channels, Height, Width)
  auto *input = graph.add_input("input", {1, 3, 224, 224});

  // Conv2d: 3 input channels → 64 output channels, 3x3 kernel
  auto *weight = graph.add_weight("conv_weight", {64, 3, 3, 3});

  lumen::OpAttributes conv_attrs;
  conv_attrs.int_array_attrs["stride"] = {1, 1};
  conv_attrs.int_array_attrs["padding"] = {1, 1};

  auto *conv_out = graph.add_op("conv2d", {input, weight}, conv_attrs);
  auto *relu_out = graph.add_op("relu", {conv_out});

  graph.mark_output(relu_out);

  std::cout << "Before optimization:" << std::endl;
  graph.print_summary();

  // Optimize (should fuse conv2d + relu)
  graph.optimize();

  std::cout << "\nAfter optimization:" << std::endl;
  graph.print_summary();

  // Compile and verify
  auto *executable = graph.compile(&rt);

  std::cout << "Compilation successful!" << std::endl;
  std::cout << "Memory usage: "
            << (executable->get_memory_usage() / 1024.0 / 1024.0) << " MB"
            << std::endl;

  delete executable;
}

// Example 3: Multi-layer Network (demonstrates shape inference)
void example_multilayer() {
  std::cout << "\n=== Example 3: Multi-Layer Network ===" << std::endl;

  lumen::Runtime rt;
  lumen::Graph graph;

  // Input
  auto *input = graph.add_input("input", {4, 3, 32, 32}); // Batch=4, 32x32 RGB

  // Layer 1: Conv + BatchNorm + ReLU
  auto *conv1_w = graph.add_weight("conv1_w", {16, 3, 3, 3});
  lumen::OpAttributes conv1_attrs;
  conv1_attrs.int_array_attrs["stride"] = {1, 1};
  conv1_attrs.int_array_attrs["padding"] = {1, 1};

  auto *conv1 = graph.add_op("conv2d", {input, conv1_w}, conv1_attrs);
  auto *bn1 = graph.add_op("batchnorm", {conv1});
  auto *relu1 = graph.add_op("relu", {bn1});

  // Layer 2: MaxPool
  lumen::OpAttributes pool_attrs;
  pool_attrs.int_array_attrs["kernel_size"] = {2, 2};
  pool_attrs.int_array_attrs["stride"] = {2, 2};

  auto *pool1 = graph.add_op("maxpool2d", {relu1}, pool_attrs);

  // Layer 3: Conv + ReLU
  auto *conv2_w = graph.add_weight("conv2_w", {32, 16, 3, 3});
  lumen::OpAttributes conv2_attrs;
  conv2_attrs.int_array_attrs["stride"] = {1, 1};
  conv2_attrs.int_array_attrs["padding"] = {1, 1};

  auto *conv2 = graph.add_op("conv2d", {pool1, conv2_w}, conv2_attrs);
  auto *relu2 = graph.add_op("relu", {conv2});

  // Layer 4: Global Average Pool + Flatten
  auto *flatten = graph.add_op("flatten", {relu2});

  graph.mark_output(flatten);

  std::cout << "Network architecture:" << std::endl;
  graph.print_summary();

  // Optimize
  graph.optimize();

  std::cout << "\nOptimized network:" << std::endl;
  graph.print_summary();

  // Compile
  auto *executable = graph.compile(&rt);

  std::cout << "\nCompiled successfully!" << std::endl;
  std::cout << "Total memory: "
            << (executable->get_memory_usage() / 1024.0 / 1024.0) << " MB"
            << std::endl;

  delete executable;
}

// Example 4: Profiling
void example_profiling() {
  std::cout << "\n=== Example 4: Profiling ===" << std::endl;

  lumen::Runtime rt;
  lumen::Graph graph;

  auto *input = graph.add_input("input", {1, 512});
  auto *w1 = graph.add_weight("w1", {512, 256});
  auto *w2 = graph.add_weight("w2", {256, 128});
  auto *w3 = graph.add_weight("w3", {128, 10});

  auto *fc1 = graph.add_op("matmul", {input, w1});
  auto *relu1 = graph.add_op("relu", {fc1});
  auto *fc2 = graph.add_op("matmul", {relu1, w2});
  auto *relu2 = graph.add_op("relu", {fc2});
  auto *fc3 = graph.add_op("matmul", {relu2, w3});

  graph.mark_output(fc3);
  graph.optimize();

  auto *executable = graph.compile(&rt);

  // Create input
  auto *input_buffer = rt.alloc({1, 512});
  float *data = (float *)input_buffer->data();
  for (int i = 0; i < 512; i++)
    data[i] = 0.1f;

  // Profile
  std::cout << "\nProfiling execution..." << std::endl;
  auto profile_data = executable->profile({input_buffer});

  std::cout << "\nPer-operation timing:" << std::endl;
  for (const auto &p : profile_data) {
    std::cout << "  " << p.node_name << ": " << p.time_ms << " ms ("
              << p.backend << ")" << std::endl;
  }

  delete executable;
  delete input_buffer;
}

// Example 5: Error Handling
void example_error_handling() {
  std::cout << "\n=== Example 5: Error Handling ===" << std::endl;

  lumen::Runtime rt;
  lumen::Graph graph;

  try {
    // This should fail: matmul with incompatible shapes
    auto *a = graph.add_input("a", {10, 20});
    auto *b = graph.add_input("b", {15, 30}); // Wrong! Should be {20, X}

    auto *result = graph.add_op("matmul", {a, b});

    std::cout << "ERROR: Should have caught shape mismatch!" << std::endl;

  } catch (const lumen::ShapeInferenceError &e) {
    std::cout << "✓ Caught expected error: " << e.what() << std::endl;
  } catch (const lumen::GraphException &e) {
    std::cout << "✓ Caught graph error: " << e.what() << std::endl;
  }

  try {
    // This should fail: no outputs marked
    lumen::Graph graph2;
    auto *input = graph2.add_input("input", {1, 10});
    graph2.add_op("relu", {input});

    // Forgot to call graph2.mark_output()!
    auto *executable = graph2.compile(&rt);

    std::cout << "ERROR: Should have caught missing output!" << std::endl;

  } catch (const lumen::GraphCompilationError &e) {
    std::cout << "✓ Caught expected error: " << e.what() << std::endl;
  }
}

int main() {
  std::cout << "=================================================="
            << std::endl;
  std::cout << "Lumen Graph IR Examples" << std::endl;
  std::cout << "=================================================="
            << std::endl;

  example_linear_model();
  example_conv_relu();
  example_multilayer();
  example_profiling();
  example_error_handling();

  std::cout << "\n=================================================="
            << std::endl;
  std::cout << "All examples completed successfully!" << std::endl;
  std::cout << "=================================================="
            << std::endl;

  return 0;
}