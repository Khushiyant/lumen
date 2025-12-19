#include "lumen/lumen.hpp"
#include <iostream>
#include <vector>
#include <cstring>
#include <cmath>

#ifdef __APPLE__
    #ifndef ACCELERATE_NEW_LAPACK
        #define ACCELERATE_NEW_LAPACK
    #endif
    #ifndef ACCELERATE_LAPACK_ILP64
        #define ACCELERATE_LAPACK_ILP64
    #endif
    #include <Accelerate/Accelerate.h>
#elif defined(LUMEN_USE_OPENBLAS)
    #include <cblas.h>
#endif

namespace lumen {

class CPUBackend : public Backend {
public:
    CPUBackend() {
        #ifdef LUMEN_USE_OPENBLAS
            std::cout << "[Lumen] CPU Backend Initialized (OpenBLAS Optimized)" << std::endl;
        #elif defined(__APPLE__)
            std::cout << "[Lumen] CPU Backend Initialized (Apple Accelerate)" << std::endl;
        #else
            std::cout << "[Lumen] CPU Backend Initialized (Naive Fallback)" << std::endl;
        #endif
    }

    Buffer* create_buffer(const std::vector<size_t>& shape) override {
        size_t total_elements = 1;
        for (auto d : shape) total_elements *= d;
        float* ptr = new float[total_elements];
        // FIX: Added nullptr as the 5th argument (Runtime* rt)
        return new Buffer(shape, ptr, ptr, this, nullptr);
    }

    void free_buffer(void* device_ptr) override {
        if (device_ptr) delete[] static_cast<float*>(device_ptr);
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
                    vDSP_vadd(A, 1, B, 1, out, 1, size);
                #else
                    for (size_t i = 0; i < size; ++i) out[i] = A[i] + B[i];
                #endif
            } 
            else if (op.op_name == "matmul") {
                float* A = static_cast<float*>(op.inputs[0]->data());
                float* B = static_cast<float*>(op.inputs[1]->data());
                
                long M = (long)op.inputs[0]->shape()[0];
                long K = (long)op.inputs[0]->shape()[1];
                long N = (long)op.inputs[1]->shape()[1];

                #ifdef __APPLE__
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                                M, N, K, 1.0f, A, K, B, N, 0.0f, out, N);
                #elif defined(LUMEN_USE_OPENBLAS)
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                                (int)M, (int)N, (int)K, 
                                1.0f, A, (int)K, B, (int)N, 0.0f, out, (int)N);
                #else
                    std::memset(out, 0, M * N * sizeof(float));
                    for (int m = 0; m < M; ++m) {
                        for (int n = 0; n < N; ++n) {
                            float sum = 0.0f;
                            for (int k = 0; k < K; ++k) sum += A[m * K + k] * B[k * N + n];
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