#include <lumen/lumen.hpp>
#include <iostream>
#include <vector>

int main() {
    std::cout << "--- Lumen Intelligent Router Demo ---" << std::endl;
    lumen::Runtime rt;

    // 1. Small Operation Test (Should route to CPU)
    // The overhead of sending 100 elements to GPU is higher than just doing it on CPU.
    std::cout << "\n[Test 1] Small Matrix (10x10)..." << std::endl;
    {
        auto* A = rt.alloc({10, 10});
        auto* B = rt.alloc({10, 10});
        auto* C = rt.alloc({10, 10});

        // Initialize
        float* ptr = (float*)A->data();
        for(int i=0; i<100; i++) ptr[i] = 1.0f;

        // EXECUTE: Router analyzes size (100 elements) -> Chooses CPU
        rt.execute("matmul", {A, B}, C);
        
        std::cout << " -> Completed on Backend: " << rt.current_backend() << std::endl;
        
        delete A; delete B; delete C;
    }

    // 2. Large Operation Test (Should route to CUDA)
    // 4 Million elements justify the PCIe transfer cost.
    std::cout << "\n[Test 2] Large Matrix (2048x2048)..." << std::endl;
    {
        size_t dim = 2048;
        auto* A = rt.alloc({dim, dim});
        auto* B = rt.alloc({dim, dim});
        auto* C = rt.alloc({dim, dim});

        // EXECUTE: Router analyzes size (4M elements) -> Chooses CUDA
        rt.execute("matmul", {A, B}, C);
        
        std::cout << " -> Completed on Backend: " << rt.current_backend() << std::endl;

        delete A; delete B; delete C;
    }

    std::cout << "\n--- Demo Complete ---" << std::endl;
    return 0;
}