#include <cassert>
#include <cmath>
#include <iostream>
#include <lumen/lumen.hpp>
#include <vector>

// Test helper to compare results with tolerance
bool arrays_equal(const float *a, const float *b, size_t n, float tol = 1e-4f) {
  for (size_t i = 0; i < n; ++i) {
    if (std::abs(a[i] - b[i]) > tol) {
      std::cerr << "Mismatch at index " << i << ": " << a[i] << " vs " << b[i]
                << std::endl;
      return false;
    }
  }
  return true;
}

// Test each operation on all available backends
void test_operation_parity(const std::string &op_name,
                           const std::vector<size_t> &input_shape1,
                           const std::vector<size_t> &input_shape2,
                           const std::vector<size_t> &output_shape,
                           lumen::OpAttributes attrs = {}) {
  std::cout << "\n[Testing " << op_name << "]" << std::endl;

  lumen::Runtime rt;
  std::vector<std::string> backends = {"cpu"};

#ifdef LUMEN_USE_CUDA
  backends.push_back("cuda");
#endif

#ifdef LUMEN_USE_METAL
  backends.push_back("metal");
#endif

  // Store reference result from CPU
  std::vector<float> reference_result;
  bool have_reference = false;

  for (const auto &backend : backends) {
    try {
      rt.set_backend(backend);
    } catch (...) {
      std::cout << "  [" << backend << "] Backend not available, skipping"
                << std::endl;
      continue;
    }

    if (rt.current_backend() != backend) {
      std::cout << "  [" << backend << "] Could not activate, skipping"
                << std::endl;
      continue;
    }

    // Allocate buffers
    auto *input1 = rt.alloc(input_shape1);
    auto *input2 = input_shape2.empty() ? nullptr : rt.alloc(input_shape2);
    auto *output = rt.alloc(output_shape);

    // Initialize inputs with test data
    float *in1_data = (float *)input1->data();
    for (size_t i = 0; i < input1->num_elements(); ++i) {
      in1_data[i] = (float)(i % 100) / 10.0f;
    }

    if (input2) {
      float *in2_data = (float *)input2->data();
      for (size_t i = 0; i < input2->num_elements(); ++i) {
        in2_data[i] = (float)((i + 5) % 100) / 10.0f;
      }
    }

    // Execute operation
    std::vector<lumen::Buffer *> inputs = {input1};
    if (input2)
      inputs.push_back(input2);

    try {
      rt.execute(op_name, inputs, output, attrs);
      rt.submit();
      rt.wait_all();

      // Get results
      float *result = (float *)output->data();

      if (!have_reference) {
        // First backend (CPU) becomes reference
        reference_result.assign(result, result + output->num_elements());
        have_reference = true;
        std::cout << "  [" << backend << "] ✓ Reference result captured"
                  << std::endl;
      } else {
        // Compare with reference
        if (arrays_equal(reference_result.data(), result,
                         output->num_elements())) {
          std::cout << "  [" << backend << "] ✓ Results match CPU reference"
                    << std::endl;
        } else {
          std::cout << "  [" << backend << "] ✗ Results differ from CPU!"
                    << std::endl;
        }
      }

    } catch (const std::exception &e) {
      std::cout << "  [" << backend << "] ✗ Failed: " << e.what() << std::endl;
    }

    // Cleanup
    delete input1;
    if (input2)
      delete input2;
    delete output;
  }
}

int main() {
  std::cout << "========================================" << std::endl;
  std::cout << "Operation Parity Test Suite" << std::endl;
  std::cout << "========================================" << std::endl;

  // Element-wise operations
  test_operation_parity("add", {4, 4}, {4, 4}, {4, 4});
  test_operation_parity("mul", {4, 4}, {4, 4}, {4, 4});
  test_operation_parity("relu", {4, 4}, {}, {4, 4});
  test_operation_parity("sigmoid", {4, 4}, {}, {4, 4});
  test_operation_parity("tanh", {4, 4}, {}, {4, 4});

  // Matrix operations
  test_operation_parity("matmul", {4, 8}, {8, 4}, {4, 4});

  // Normalization operations
  test_operation_parity("softmax", {4, 10}, {}, {4, 10});

  lumen::OpAttributes layer_norm_attrs;
  layer_norm_attrs.float_attrs["epsilon"] = 1e-5f;
  test_operation_parity("layer_norm", {2, 8}, {}, {2, 8}, layer_norm_attrs);

  // Convolution operations
  lumen::OpAttributes conv_attrs;
  conv_attrs.int_array_attrs["stride"] = {1, 1};
  conv_attrs.int_array_attrs["padding"] = {1, 1};
  test_operation_parity("conv2d", {1, 3, 8, 8}, {16, 3, 3, 3}, {1, 16, 8, 8},
                        conv_attrs);

  // Pooling operations
  lumen::OpAttributes pool_attrs;
  pool_attrs.int_array_attrs["kernel_size"] = {2, 2};
  pool_attrs.int_array_attrs["stride"] = {2, 2};
  test_operation_parity("maxpool2d", {1, 4, 8, 8}, {}, {1, 4, 4, 4},
                        pool_attrs);
  test_operation_parity("avgpool2d", {1, 4, 8, 8}, {}, {1, 4, 4, 4},
                        pool_attrs);

  test_operation_parity("global_average_pool", {2, 4, 8, 8}, {}, {2, 4});

  // Reduction operations
  test_operation_parity("reduce_mean", {4, 10}, {}, {4});
  test_operation_parity("reduce_sum", {4, 10}, {}, {4});

  std::cout << "\n========================================" << std::endl;
  std::cout << "Operation Parity Tests Complete" << std::endl;
  std::cout << "========================================" << std::endl;

  return 0;
}