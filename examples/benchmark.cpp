#include <lumen/lumen.hpp>
#include <iostream>
#include <chrono>
#include <vector>
#include <iomanip>
#include <cmath>

void run_matmul_bench(lumen::Runtime& rt, int dim) {
    size_t n = dim * dim;
    auto* A = rt.alloc(n * sizeof(float));
    auto* B = rt.alloc(n * sizeof(float));
    auto* C = rt.alloc(n * sizeof(float));

    // Warm-up to trigger JIT compilation/caching
    rt.execute("matmul", {A, B}, C);

    auto start = std::chrono::high_resolution_clock::now();
    int iterations = 10;
    for(int i = 0; i < iterations; ++i) {
        rt.execute("matmul", {A, B}, C);
    }
    auto end = std::chrono::high_resolution_clock::now();

    double avg_ms = std::chrono::duration<double, std::milli>(end - start).count() / iterations;
    
    // Formula for Floating Point Operations in MatMul: 2 * M * N * K
    double gflops = (2.0 * dim * dim * dim) / (avg_ms * 1e6);

    std::cout << "Matrix: " << dim << "x" << dim 
              << " | Avg Time: " << std::fixed << std::setprecision(3) << avg_ms << " ms"
              << " | Performance: " << gflops << " GFLOPS" << std::endl;

    delete A; delete B; delete C;
}

int main() {
    lumen::Runtime rt;
    std::cout << "\n--- Lumen MatMul Performance Benchmark ---\n";
    
    std::vector<int> sizes = {256, 512, 1024, 2048};
    for(int dim : sizes) {
        run_matmul_bench(rt, dim);
    }
    
    return 0;
}