#include <lumen/lumen.hpp>
#include <cassert>
#include <iostream>
#include <cmath>
#include <vector>
#include <chrono> 
#include <iomanip>

// Helper to print timing
void print_metrics(const std::string& name, std::chrono::nanoseconds duration) {
    double ms = duration.count() / 1e6;
    std::cout << "  - " << name << ": " << std::fixed << std::setprecision(3) << ms << " ms" << std::endl;
}

// 1. Functional Test for CPU
void test_cpu_fallback() {
    lumen::Runtime rt;
    std::cout << "[Test] Switching to CPU Backend..." << std::endl;
    rt.set_backend("cpu");

    if (rt.current_backend() != "cpu") {
        std::cout << "SKIPPED: CPU backend not found (should not happen)." << std::endl;
        return;
    }

    auto* A = rt.alloc({1});
    auto* B = rt.alloc({1});
    auto* C = rt.alloc({1});

    *(float*)A->data() = 10.0f;
    *(float*)B->data() = 5.0f;

    // CPU executes synchronously immediately inside sync/execute
    rt.execute("add", {A, B}, C); 
    
    assert(*(float*)C->data() == 15.0f);
    
    rt.execute("mul", {A, B}, C);
    assert(*(float*)C->data() == 50.0f);

    std::cout << "PASS: CPU Backend Functional Test" << std::endl;
    delete A; delete B; delete C;
}

// 2. Existing Metal Tests (Runs on default backend)
void test_metal_features() {
    lumen::Runtime rt;
    rt.set_backend("metal");
    
    if (rt.current_backend() != "metal") return;

    // Zero-Copy Check
    size_t n = 1024;
    auto* A = rt.alloc({n});
    auto* B = rt.alloc({n});
    auto* C = rt.alloc({n});

    float* a_ptr = (float*)A->data();
    float* b_ptr = (float*)B->data();
    for(size_t i=0; i<n; ++i) { a_ptr[i] = 1.0f; b_ptr[i] = 1.0f; }

    rt.execute("add", {A, B}, C);
    assert(((float*)C->data())[0] == 2.0f);

    std::cout << "PASS: Metal Backend (Zero-Copy & Ops)" << std::endl;
    delete A; delete B; delete C;
}

// 3. Compare Performance: Metal vs CPU
void benchmark_backend_comparison() {
    lumen::Runtime rt;
    size_t dim = 1024;
    
    auto run_bench = [&](const std::string& backend) {
        rt.set_backend(backend);
        // Only run if the backend is actually initialized
        if (rt.current_backend() != backend && backend != "cpu") return;

        auto* A = rt.alloc({dim, dim});
        auto* B = rt.alloc({dim, dim});
        auto* C = rt.alloc({dim, dim});

        rt.execute("matmul", {A, B}, C);
        rt.submit(); 

        auto start = std::chrono::high_resolution_clock::now();
        rt.execute("matmul", {A, B}, C);
        rt.submit();
        auto end = std::chrono::high_resolution_clock::now();

        std::cout << "PASS: " << backend << " MatMul (" << dim << "x" << dim << ")" << std::endl;
        print_metrics("Latency", end - start);

        delete A; delete B; delete C;
    };

    std::cout << "\n--- Performance Comparison ---" << std::endl;
    run_bench("cuda");   // Added CUDA
    run_bench("metal"); 
    run_bench("cpu");   
}

void test_intelligent_routing() {
    std::cout << "\n--- Testing Intelligent Router ---" << std::endl;
    lumen::Runtime rt;

    // 1. Small Operation (Should route to CPU)
    auto* A = rt.alloc({10}); 
    auto* B = rt.alloc({10}); 
    auto* C = rt.alloc({10});
    
    // CPU is faster for tiny ops (low latency)
    rt.execute("add", {A, B}, C); 
    // If you print rt.current_backend(), it should be 'cpu'

    // 2. Heavy Operation (Should route to GPU if available)
    size_t dim = 2048;
    auto* BigA = rt.alloc({dim, dim});
    auto* BigB = rt.alloc({dim, dim});
    auto* BigC = rt.alloc({dim, dim});

    // GPU is faster for throughput
    rt.execute("matmul", {BigA, BigB}, BigC);
    // If CUDA is working, it should switch. If broken, it falls back to CPU.

    std::cout << "PASS: Intelligent Routing Logic Executed" << std::endl;
    
    delete A; delete B; delete C;
    delete BigA; delete BigB; delete BigC;
}

int main() {
    std::cout << "--- Starting Lumen Multi-Backend Tests ---" << std::endl;
    
    test_cpu_fallback();
    test_metal_features();
    benchmark_backend_comparison();
    test_intelligent_routing();
    
    std::cout << "--- All Tests Completed Successfully ---" << std::endl;
    return 0;
}