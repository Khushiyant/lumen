#include <lumen/lumen.hpp>
#include <cassert>
#include <iostream>

void test_ops() {
    lumen::Runtime rt;
    auto* A = rt.alloc(sizeof(float));
    auto* B = rt.alloc(sizeof(float));
    auto* C = rt.alloc(sizeof(float));

    *(float*)A->data() = 20.0f;
    *(float*)B->data() = 4.0f;

    rt.execute("add", {A, B}, C);
    assert(*(float*)C->data() == 24.0f);

    rt.execute("mul", {A, B}, C);
    assert(*(float*)C->data() == 80.0f);

    std::cout << "Unit Tests Passed: Operations Verified." << std::endl;
    delete A; delete B; delete C;
}

void test_matrix_multiplication() {
    lumen::Runtime rt;
    
    // Create 4x4 matrices (16 elements)
    size_t dim = 4;
    size_t n = dim * dim;
    auto* A = rt.alloc(n * sizeof(float));
    auto* B = rt.alloc(n * sizeof(float));
    auto* C = rt.alloc(n * sizeof(float));

    float* a_ptr = static_cast<float*>(A->data());
    float* b_ptr = static_cast<float*>(B->data());

    // Identity Matrix A * Constant Matrix B
    for(size_t i = 0; i < n; ++i) {
        a_ptr[i] = (i % (dim + 1) == 0) ? 1.0f : 0.0f; // Identity
        b_ptr[i] = 5.0f;                               // All 5s
    }

    rt.execute("matmul", {A, B}, C);

    float* c_ptr = static_cast<float*>(C->data());
    // Result should be all 5s because Identity * B = B
    for(size_t i = 0; i < n; ++i) {
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