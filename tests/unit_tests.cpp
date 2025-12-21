#include <cassert>
#include <cmath>
#include <iostream>
#include <lumen/lumen.hpp>

// Forward declaration of the function in onnx_tests.cpp
void test_onnx_loading();

void test_ops() {
  lumen::Runtime rt;
  auto *A = rt.alloc({1});
  auto *B = rt.alloc({1});
  auto *C = rt.alloc({1});

  *(float *)A->data() = 20.0f;
  *(float *)B->data() = 4.0f;

  rt.execute("add", {A, B}, C);
  assert(*(float *)C->data() == 24.0f);

  rt.execute("mul", {A, B}, C);
  assert(*(float *)C->data() == 80.0f);

  std::cout << "PASS: Basic Operations (Add/Mul)" << std::endl;
  delete A;
  delete B;
  delete C;
}

void test_matrix_multiplication() {
  lumen::Runtime rt;
  size_t dim = 4;
  auto *A = rt.alloc({dim, dim});
  auto *B = rt.alloc({dim, dim});
  auto *C = rt.alloc({dim, dim});

  float *a_ptr = static_cast<float *>(A->data());
  float *b_ptr = static_cast<float *>(B->data());

  for (size_t i = 0; i < dim * dim; ++i) {
    a_ptr[i] = (i % (dim + 1) == 0) ? 1.0f : 0.0f; // Identity
    b_ptr[i] = 5.0f;
  }

  rt.execute("matmul", {A, B}, C);

  float *c_ptr = static_cast<float *>(C->data());
  for (size_t i = 0; i < dim * dim; ++i) {
    assert(std::abs(c_ptr[i] - 5.0f) < 1e-5);
  }

  delete A;
  delete B;
  delete C;
  std::cout << "PASS: Matrix Multiplication" << std::endl;
}

int main() {
  std::cout << "--- Starting Unit Tests ---" << std::endl;

  test_ops();
  test_matrix_multiplication();

  std::cout << "\n--- Starting ONNX Integration Test ---" << std::endl;
  // This will throw an exception if the model or importer fails
  try {
    test_onnx_loading();
    std::cout << "PASS: ONNX Loading & Inference" << std::endl;
  } catch (const std::exception &e) {
    std::cerr << "FAIL: ONNX Test threw exception: " << e.what() << std::endl;
    return 1;
  }

  std::cout << "--- All Unit Tests Passed ---" << std::endl;
  return 0;
}