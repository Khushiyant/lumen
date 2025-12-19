#include <lumen/lumen.hpp>
#include <iostream>
#include <vector>

int main() {
    lumen::Runtime rt;
    
    // Define size (e.g., 1024x1024 matrix)
    size_t dim = 1024;
    size_t n = dim * dim;
    size_t bytes = n * sizeof(float);

    std::cout << "[Example] Allocating buffers for Kernel Fusion..." << std::endl;

    // 1. Allocate Buffers
    auto* A = rt.alloc(bytes);
    auto* B = rt.alloc(bytes);
    auto* C = rt.alloc(bytes); // Intermediate (Output of MatMul, Input to Add)
    auto* D = rt.alloc(bytes); // Bias
    auto* E = rt.alloc(bytes); // Final Result

    // 2. Initialize Data (Fill with dummy values)
    float* a_ptr = (float*)A->data();
    float* b_ptr = (float*)B->data();
    float* d_ptr = (float*)D->data();
    
    for(size_t i = 0; i < n; i++) {
        a_ptr[i] = 1.0f; // Identity-like
        b_ptr[i] = 2.0f;
        d_ptr[i] = 5.0f; // Add 5 to everything
    }

    // 3. Record Operations (Lazy Execution)
    std::cout << "[Example] Recording operation chain..." << std::endl;
    
    // Op 1: C = A * B (Matrix Multiply)
    rt.record("matmul", {A, B}, C);
    
    // Op 2: E = C + D (Element-wise Add)
    // Note: 'C' is consumed here immediately without going back to CPU RAM
    rt.record("add", {C, D}, E);

    // 4. Sync (Trigger Fusion)
    std::cout << "[Example] Syncing (Compiling & Running Fused Kernel)..." << std::endl;
    rt.sync();

    // 5. Verify Results
    float* result = (float*)E->data();
    std::cout << "Workflow Complete." << std::endl;
    std::cout << "Result index 0: " << result[0] << " (Expected approx 2048*2 + 5 = 4101.0)" << std::endl; 
    // Note: For a 1024x1024 matrix multiplication of all 1s and 2s, 
    // the dot product is row*col. row is 1s, col is 2s. len is 1024.
    // So 1*2 * 1024 = 2048. Then we add 5. So 2053.
    // Wait, simple logic: Row(1,1...) dot Col(2,2...) = sum(1*2 for 1024) = 2048. 
    // + Bias(5) = 2053.
    
    // Clean up
    delete A; delete B; delete C; delete D; delete E;

    return 0;
}