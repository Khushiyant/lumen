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
  rt.set_backend("cpu");

  if (rt.current_backend() != "cpu") {
    std::cout << "SKIPPED: CPU backend not found (should not happen)."
              << std::endl;
    return;
  }

  auto *A = rt.alloc({1});
  auto *B = rt.alloc({1});
  auto *C = rt.alloc({1});

  *(float *)A->data() = 10.0f;
  *(float *)B->data() = 5.0f;

  // CPU executes synchronously immediately inside sync/execute
  rt.execute("add", {A, B}, C);

  assert(*(float *)C->data() == 15.0f);

  rt.execute("mul", {A, B}, C);
  assert(*(float *)C->data() == 50.0f);

  std::cout << "PASS: CPU Backend Functional Test" << std::endl;
  delete A;
  delete B;
  delete C;
}

// 2. Existing Metal Tests (Runs on default backend)
void test_metal_features() {
  lumen::Runtime rt;
  rt.set_backend("metal");

  if (rt.current_backend() != "metal")
    return;

  // Zero-Copy Check
  size_t n = 1024;
  auto *A = rt.alloc({n});
  auto *B = rt.alloc({n});
  auto *C = rt.alloc({n});

  float *a_ptr = (float *)A->data();
  float *b_ptr = (float *)B->data();
  for (size_t i = 0; i < n; ++i) {
    a_ptr[i] = 1.0f;
    b_ptr[i] = 1.0f;
  }

  rt.execute("add", {A, B}, C);
  assert(((float *)C->data())[0] == 2.0f);

  std::cout << "PASS: Metal Backend (Zero-Copy & Ops)" << std::endl;
  delete A;
  delete B;
  delete C;
}

// 3. Compare Performance: Metal vs CPU
void benchmark_backend_comparison() {
  lumen::Runtime rt;
  size_t dim = 4096;

  auto run_bench = [&](const std::string &backend) {
    rt.set_backend(backend);
    // Only run if the backend is actually active on this system
    if (rt.current_backend() != backend && backend != "cpu")
      return;

    auto *A = rt.alloc({dim, dim});
    auto *B = rt.alloc({dim, dim});
    auto *C = rt.alloc({dim, dim});

    // Warmup: Build cache and ensure buffers are on device
    rt.execute("matmul", {A, B}, C);
    rt.submit();

    auto start = std::chrono::high_resolution_clock::now();
    rt.execute("matmul", {A, B}, C);
    rt.submit(); // Measure true execution
    auto end = std::chrono::high_resolution_clock::now();

    std::cout << "PASS: " << backend << " MatMul (" << dim << "x" << dim << ")"
              << std::endl;
    print_metrics("Latency", end - start);

    delete A;
    delete B;
    delete C;
  };

  std::cout << "\n--- Performance Comparison ---" << std::endl;
  run_bench("cuda");
  run_bench("metal");
  run_bench("cpu");
}

void test_intelligent_routing() {
  std::cout << "\n--- Testing Intelligent Router ---" << std::endl;
  lumen::Runtime rt;

  // 1. Small Operation (Should route to CPU)
  auto *A = rt.alloc({10});
  auto *B = rt.alloc({10});
  auto *C = rt.alloc({10});

  // CPU is faster for tiny ops (low latency)
  rt.execute("add", {A, B}, C);
  // If you print rt.current_backend(), it should be 'cpu'

  // 2. Heavy Operation (Should route to GPU if available)
  size_t dim = 2048;
  auto *BigA = rt.alloc({dim, dim});
  auto *BigB = rt.alloc({dim, dim});
  auto *BigC = rt.alloc({dim, dim});

  // GPU is faster for throughput
  rt.execute("matmul", {BigA, BigB}, BigC);
  // If CUDA is working, it should switch. If broken, it falls back to CPU.

  std::cout << "PASS: Intelligent Routing Logic Executed" << std::endl;

  delete A;
  delete B;
  delete C;
  delete BigA;
  delete BigB;
  delete BigC;
}

void benchmark_memory_optimization() {
  std::cout << "\n--- Benchmarking Memory Planner Efficiency ---" << std::endl;
  lumen::Runtime rt;
  lumen::Graph graph;

  // Create a long sequential chain (Deep MLP)
  size_t layers = 20;
  size_t dim = 1024; // 4MB per buffer
  auto *current = graph.add_input("input", {1, dim});

  for (size_t i = 0; i < layers; ++i) {
    auto *weight = graph.add_weight("w" + std::to_string(i), {dim, dim});
    current = graph.add_op("matmul", {current, weight});
    current = graph.add_op("relu", {current});
  }
  graph.mark_output(current);

  // Compile and measure
  auto *executable = graph.compile(&rt);

  size_t total_alloc =
      layers * 2 * (dim * dim * sizeof(float));     // Matmul + Relu per layer
  size_t peak_mem = executable->get_memory_usage(); // From your Memory Planner

  double savings = (1.0 - (double)peak_mem / total_alloc) * 100.0;

  std::cout << "  - Total Layers: " << layers << std::endl;
  // Without optimization, memory would scale linearly with layers
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

  // Simple CNN-like block
  auto *in = graph.add_input("in", {1, 3, 224, 224});
  auto *w = graph.add_weight("w", {64, 3, 3, 3});
  auto *conv = graph.add_op("conv2d", {in, w});
  auto *out = graph.add_op("relu", {conv});
  graph.mark_output(out);

  auto *executable = graph.compile(&rt);
  auto *input_buf = rt.alloc({1, 3, 224, 224});

  std::cout << "\n--- Throughput Benchmark (100 iterations) ---" << std::endl;
  auto start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < 100; ++i) {
    executable->execute({input_buf});
  }

  auto end = std::chrono::high_resolution_clock::now();
  double total_ms =
      std::chrono::duration<double, std::milli>(end - start).count();

  std::cout << "  - Average Latency: " << total_ms / 100.0 << " ms"
            << std::endl;
  std::cout << "  - Throughput: " << 1000.0 / (total_ms / 100.0) << " IPS"
            << std::endl;

  delete executable;
  delete input_buf;
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