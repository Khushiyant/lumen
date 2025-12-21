#include <lumen/lumen.hpp>
#include <cassert>
#include <iostream>
#include <cmath>


void test_ops() {
    lumen::Runtime rt;
    // Now passing explicit shapes
    auto* A = rt.alloc({1});
    auto* B = rt.alloc({1});
    auto* C = rt.alloc({1});

    *(float*)A->data() = 20.0f;
    *(float*)B->data() = 4.0f;

    rt.execute("add", {A, B}, C);
    assert(*(float*)C->data() == 24.0f);

    rt.execute("mul", {A, B}, C);
    assert(*(float*)C->data() == 80.0f);

    std::cout << "Unit Tests Passed: Operations Verified." << std::endl;
    delete A; delete B; delete C; // Triggering RAII cleanup
}

void test_matrix_multiplication() {
    lumen::Runtime rt;
    
    size_t dim = 4;
    auto* A = rt.alloc({dim, dim});
    auto* B = rt.alloc({dim, dim});
    auto* C = rt.alloc({dim, dim});

    float* a_ptr = static_cast<float*>(A->data());
    float* b_ptr = static_cast<float*>(B->data());

    for(size_t i = 0; i < dim*dim; ++i) {
        a_ptr[i] = (i % (dim + 1) == 0) ? 1.0f : 0.0f; 
        b_ptr[i] = 5.0f;                               
    }

    rt.execute("matmul", {A, B}, C);

    float* c_ptr = static_cast<float*>(C->data());
    for(size_t i = 0; i < dim*dim; ++i) {
        assert(std::abs(c_ptr[i] - 5.0f) < 1e-5);
    }

    delete A; delete B; delete C;
    std::cout << "PASS: Matrix Multiplication (Identity Test)" << std::endl;
}

int main() {
    test_ops();
    test_matrix_multiplication();
    return 0;
}