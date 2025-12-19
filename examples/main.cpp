#include <lumen/lumen.hpp>
#include <iostream>
#include <vector>
#include <iomanip>

void run_on_backend(lumen::Runtime& rt, const std::string& backend_name) {
    std::cout << "\n--- Running on Backend: " << backend_name << " ---" << std::endl;
    
    // 1. Explicitly Switch Backend
    rt.set_backend(backend_name);
    
    if (rt.current_backend() != backend_name) {
        std::cout << "Skipping: Backend not available." << std::endl;
        return;
    }

    // 2. Allocate Buffers (Allocated on the ACTIVE backend)
    // Note: In this Phase 1 implementation, buffers cannot yet be shared 
    // between backends. We must allocate new ones for the CPU.
    size_t dim = 1024;
    auto* A = rt.alloc({dim, dim});
    auto* B = rt.alloc({dim, dim});
    auto* C = rt.alloc({dim, dim});

    // 3. Initialize Data
    float* a_ptr = (float*)A->data();
    float* b_ptr = (float*)B->data();
    
    size_t n = dim * dim;
    for(size_t i = 0; i < n; i++) {
        a_ptr[i] = 1.0f;
        b_ptr[i] = 2.0f;
    }

    // 4. Record & Execute
    // Metal will fuse these; CPU will execute them sequentially in sync()
    std::cout << "[" << backend_name << "] Recording MatMul..." << std::endl;
    rt.record("matmul", {A, B}, C);

    std::cout << "[" << backend_name << "] Syncing..." << std::endl;
    rt.sync();

    // 5. Verify Result
    float* result = (float*)C->data();
    // 1.0 * 2.0 * 1024 elements = 2048.0
    std::cout << "[" << backend_name << "] Result[0]: " << result[0] 
              << " (Expected: 2048.0)" << std::endl;

    // Cleanup
    delete A; delete B; delete C;
}

int main() {
    lumen::Runtime rt;
    
    // 1. Run on Metal (Default on macOS)
    run_on_backend(rt, "metal");

    // 2. Run on CPU (Universal Fallback)
    run_on_backend(rt, "cpu");

    run_on_backend(rt, "cuda");

    return 0;
}