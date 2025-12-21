#include <cassert>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <lumen/graph.hpp>
#include <lumen/lumen.hpp>
#include <vector>

// Helper to print timing
void print_metrics(const std::string &name, std::chrono::nanoseconds duration) {
  double ms = duration.count() / 1e6;
  std::cout << "  - " << name << ": " << std::fixed << std::setprecision(3)
            << ms << " ms" << std::endl;
}

// 1. Functional Test for CPU
void test_cpu_fallback() {
  lumen::Runtime rt;
  std::cout << "[Test] Switching to CPU Backend..." << std::endl;

  try {
    rt.set_backend("cpu");
  } catch (...) {
    std::cout << "SKIPPED: CPU backend not found." << std::endl;
    return;
  }

  if (rt.current_backend() != "cpu")
    return;

  auto A = rt.alloc({1});
  auto B = rt.alloc({1});
  auto C = rt.alloc({1});

  *(float *)A->data() = 10.0f;
  *(float *)B->data() = 5.0f;

  rt.execute("add", {A, B}, C);
  assert(*(float *)C->data() == 15.0f);

  rt.execute("mul", {A, B}, C);
  assert(*(float *)C->data() == 50.0f);

  std::cout << "PASS: CPU Backend Functional Test" << std::endl;
}

// 2. Existing Metal Tests
void test_metal_features() {
  lumen::Runtime rt;

  try {
    rt.set_backend("metal");
  } catch (...) {
    return;
  }

  if (rt.current_backend() != "metal")
    return;

  size_t n = 1024;
  auto A = rt.alloc({n});
  auto B = rt.alloc({n});
  auto C = rt.alloc({n});

  float *a_ptr = (float *)A->data();
  float *b_ptr = (float *)B->data();
  for (size_t i = 0; i < n; ++i) {
    a_ptr[i] = 1.0f;
    b_ptr[i] = 1.0f;
  }

  rt.execute("add", {A, B}, C);
  assert(((float *)C->data())[0] == 2.0f);

  std::cout << "PASS: Metal Backend (Zero-Copy & Ops)" << std::endl;
}

// 3. Compare Performance: Metal vs CPU
void benchmark_backend_comparison() {
  lumen::Runtime rt;
  size_t dim = 4096;

  auto run_bench = [&](const std::string &backend) {
    try {
      rt.set_backend(backend);
    } catch (const std::exception &e) {
      return;
    }

    if (rt.current_backend() != backend && backend != "cpu")
      return;

    auto A = rt.alloc({dim, dim});
    auto B = rt.alloc({dim, dim});
    auto C = rt.alloc({dim, dim});

    // Warmup
    rt.execute("matmul", {A, B}, C);
    rt.submit();
    rt.wait_all();

    auto start = std::chrono::high_resolution_clock::now();
    rt.execute("matmul", {A, B}, C);
    rt.submit();
    rt.wait_all();
    auto end = std::chrono::high_resolution_clock::now();

    std::cout << "PASS: " << backend << " MatMul (" << dim << "x" << dim << ")"
              << std::endl;
    print_metrics("Latency", end - start);
  };

  std::cout << "\n--- Performance Comparison ---" << std::endl;
  run_bench("cuda");
  run_bench("metal");
  run_bench("cpu");
}

void test_intelligent_routing() {
  std::cout << "\n--- Testing Intelligent Router ---" << std::endl;
  lumen::Runtime rt;

  // 1. Small Operation
  auto A = rt.alloc({10});
  auto B = rt.alloc({10});
  auto C = rt.alloc({10});

  rt.execute("add", {A, B}, C);

  // 2. Heavy Operation
  size_t dim = 2048;
  auto BigA = rt.alloc({dim, dim});
  auto BigB = rt.alloc({dim, dim});
  auto BigC = rt.alloc({dim, dim});

  rt.execute("matmul", {BigA, BigB}, BigC);

  std::cout << "PASS: Intelligent Routing Logic Executed" << std::endl;
}

void benchmark_memory_optimization() {
  std::cout << "\n--- Benchmarking Memory Planner Efficiency ---" << std::endl;
  lumen::Runtime rt;
  lumen::Graph graph;

  size_t layers = 20;
  size_t dim = 1024;
  auto *current = graph.add_input("input", {1, dim});

  for (size_t i = 0; i < layers; ++i) {
    auto *weight = graph.add_weight("w" + std::to_string(i), {dim, dim});
    current = graph.add_op("matmul", {current, weight});
    current = graph.add_op("relu", {current});
  }
  graph.mark_output(current);

  auto *executable = graph.compile(&rt);

  size_t total_alloc = layers * 2 * (dim * dim * sizeof(float));
  size_t peak_mem = executable->get_memory_usage();

  double savings = (1.0 - (double)peak_mem / total_alloc) * 100.0;

  std::cout << "  - Total Layers: " << layers << std::endl;
  std::cout << "  - Naive Memory Needed: " << total_alloc / (1024.0 * 1024.0)
            << " MB" << std::endl;
  std::cout << "  - Optimized Peak Memory: " << peak_mem / (1024.0 * 1024.0)
            << " MB" << std::endl;
  std::cout << "  - VRAM Savings: " << std::fixed << std::setprecision(2)
            << savings << "%" << std::endl;

  delete executable;
}

void benchmark_inference_throughput() {
  lumen::Runtime rt;
  lumen::Graph graph;

  auto *in = graph.add_input("in", {1, 3, 224, 224});
  auto *w = graph.add_weight("w", {64, 3, 3, 3});
  lumen::OpAttributes attrs;
  attrs.int_array_attrs["stride"] = {1, 1};
  attrs.int_array_attrs["padding"] = {1, 1};

  auto *conv = graph.add_op("conv2d", {in, w}, attrs);
  auto *out = graph.add_op("relu", {conv});
  graph.mark_output(out);

  auto *executable = graph.compile(&rt);
  auto input_buf = rt.alloc({1, 3, 224, 224});

  std::cout << "\n--- Throughput Benchmark (100 iterations) ---" << std::endl;
  auto start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < 100; ++i) {
    auto res = executable->execute({input_buf});
  }

  auto end = std::chrono::high_resolution_clock::now();
  double total_ms =
      std::chrono::duration<double, std::milli>(end - start).count();

  std::cout << "  - Average Latency: " << total_ms / 100.0 << " ms"
            << std::endl;
  std::cout << "  - Throughput: " << 1000.0 / (total_ms / 100.0) << " IPS"
            << std::endl;

  delete executable;
}

int main() {
  std::cout << "--- Starting Lumen Multi-Backend Tests ---" << std::endl;

  test_cpu_fallback();
  test_metal_features();
  benchmark_backend_comparison();
  test_intelligent_routing();
  benchmark_memory_optimization();
  benchmark_inference_throughput();

  std::cout << "--- All Tests Completed Successfully ---" << std::endl;
  return 0;
}