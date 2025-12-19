#include <lumen/lumen.hpp>
#include <iostream>

int main() {
    lumen::Runtime rt;
    size_t dim = 1024;
    size_t n = dim * dim;

    // Allocate 3 buffers
    auto* input = rt.alloc(n * sizeof(float));
    auto* weights = rt.alloc(n * sizeof(float));
    auto* output = rt.alloc(n * sizeof(float));

    // Chain: Output = (Input * Weights) [MatMul]
    std::cout << "Step 1: Running Large Matrix Multiply..." << std::endl;
    rt.execute("matmul", {input, weights}, output);

    // Chain: Weights = (Output + Input) [Element-wise Add]
    // Note: Weights is repurposed instantly without copying data!
    std::cout << "Step 2: Chaining result into Element-wise Addition..." << std::endl;
    rt.execute("add", {output, input}, weights);

    std::cout << "Workflow Complete. Result is ready in 'weights' buffer." << std::endl;

    delete input; delete weights; delete output;
    return 0;
}