#include "lumen/lumen.hpp"
#include <iostream>
#include <vector>
#include <cstring>
#include <cmath>

// Check if we are on Apple to include the Accelerate framework
#ifdef __APPLE__
    #include <Accelerate/Accelerate.h>
#else
    // If we were on Linux, we would include <cblas.h> here
#endif

namespace lumen {

class CPUBackend : public Backend {
public:
    CPUBackend() {
        std::cout << "[Lumen] CPU Backend Initialized (Optimized)" << std::endl;
    }

    Buffer* create_buffer(const std::vector<size_t>& shape) override {
        size_t total_elements = 1;
        for (auto d : shape) total_elements *= d;
        
        // Use standard new for simplicity
        float* ptr = new float[total_elements];
        return new Buffer(shape, ptr, ptr, this);
    }

    void free_buffer(void* device_ptr) override {
        if (device_ptr) {
            delete[] static_cast<float*>(device_ptr);
        }
    }

    void execute(const std::string& op_name, const std::vector<Buffer*>& inputs, Buffer* output) override {
        std::vector<QueuedOp> q = {{op_name, inputs, output}};
        sync(q);
    }

    void sync(std::vector<QueuedOp>& queue) override {
        for (const auto& op : queue) {
            float* out = static_cast<float*>(op.output->data());
            
            if (op.op_name == "add") {
                float* A = static_cast<float*>(op.inputs[0]->data());
                float* B = static_cast<float*>(op.inputs[1]->data());
                size_t size = op.output->size_bytes() / sizeof(float);
                
                #ifdef __APPLE__
                    // vDSP_vadd is Apple's vectorized addition (SIMD)
                    vDSP_vadd(A, 1, B, 1, out, 1, size);
                #else
                    for (size_t i = 0; i < size; ++i) out[i] = A[i] + B[i];
                #endif
            } 
            else if (op.op_name == "mul") {
                float* A = static_cast<float*>(op.inputs[0]->data());
                float* B = static_cast<float*>(op.inputs[1]->data());
                size_t size = op.output->size_bytes() / sizeof(float);
                
                #ifdef __APPLE__
                    vDSP_vmul(A, 1, B, 1, out, 1, size);
                #else
                    for (size_t i = 0; i < size; ++i) out[i] = A[i] * B[i];
                #endif
            }
            else if (op.op_name == "matmul") {
                float* A = static_cast<float*>(op.inputs[0]->data());
                float* B = static_cast<float*>(op.inputs[1]->data());
                
                // Shapes: [M, K] x [K, N] -> [M, N]
                // Note: BLAS assumes Column-Major by default, but C++ is Row-Major.
                // We use CblasRowMajor to handle this correctly.
                int M = (int)op.inputs[0]->shape()[0];
                int K = (int)op.inputs[0]->shape()[1];
                int N = (int)op.inputs[1]->shape()[1];

                #ifdef __APPLE__
                    // cblas_sgemm: Single-precision General Matrix Multiply
                    // C = alpha * A * B + beta * C
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                                M, N, K, 
                                1.0f, A, K, B, N, 0.0f, out, N);
                #else
                    // Fallback for non-Apple (Naive loop from before)
                    std::memset(out, 0, M * N * sizeof(float));
                    for (size_t m = 0; m < M; ++m) {
                        for (size_t n = 0; n < N; ++n) {
                            float sum = 0.0f;
                            for (size_t k = 0; k < K; ++k) sum += A[m * K + k] * B[k * N + n];
                            out[m * N + n] = sum;
                        }
                    }
                #endif
            }
        }
    }
};

std::unique_ptr<Backend> create_cpu_backend() {
    return std::make_unique<CPUBackend>();
}

} // namespace lumen