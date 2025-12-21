#include <iostream>
#include <lumen/lumen.hpp>
#include <vector>

int main() {
  std::cout << "--- Lumen Intelligent Router Demo ---" << std::endl;
  lumen::Runtime rt;

  // 1. Small Operation Test (Should route to CPU)
  std::cout << "\n[Test 1] Small Matrix (10x10)..." << std::endl;
  {
    auto A = rt.alloc({10, 10}); // Now shared_ptr
    auto B = rt.alloc({10, 10});
    auto C = rt.alloc({10, 10});

    // Initialize
    float *ptr = (float *)A->data();
    for (int i = 0; i < 100; i++)
      ptr[i] = 1.0f;

    // EXECUTE: Router analyzes size (100 elements) -> Chooses CPU
    rt.execute("matmul", {A, B}, C);

    std::cout << " -> Completed on Backend: " << rt.current_backend()
              << std::endl;

    // No delete calls needed for A, B, C
  }

  // 2. Large Operation Test (Should route to CUDA)
  std::cout << "\n[Test 2] Large Matrix (2048x2048)..." << std::endl;
  {
    size_t dim = 2048;
    auto A = rt.alloc({dim, dim});
    auto B = rt.alloc({dim, dim});
    auto C = rt.alloc({dim, dim});

    // EXECUTE: Router analyzes size (4M elements) -> Chooses CUDA
    rt.execute("matmul", {A, B}, C);

    std::cout << " -> Completed on Backend: " << rt.current_backend()
              << std::endl;
  }

  std::cout << "\n--- Demo Complete ---" << std::endl;
  return 0;
}