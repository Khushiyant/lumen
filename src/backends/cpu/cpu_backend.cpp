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


    Buffer* create_buffer(const std::vector<size_t>& shape) override { // Added override
        size_t total_elements = 1;
        for (auto d : shape) total_elements *= d;
        float* ptr = new float[total_elements];

        std::vector<size_t> strides(shape.size());
        size_t s = 1;
        for (int i = (int)shape.size() - 1; i >= 0; --i) {
            strides[i] = s;
            s *= shape[i];
        }
        return new Buffer(shape, strides, ptr, ptr, this, 0);
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
        float* out = (float*)op.output->data();
        size_t n = op.output->size_bytes() / sizeof(float);
        if (op.op_name == "relu") {
            Buffer* in_buf = op.inputs[0];
            Buffer* out_buf = op.output;
            
            float* in_data = (float*)in_buf->data();
            float* out_data = (float*)out_buf->data();

            auto& shape = in_buf->shape();
            auto& in_strides = in_buf->strides();
            auto& out_strides = out_buf->strides();

            // For a 2D Tensor (generalized to N-D would use recursion or a flat iterator)
            for (size_t i = 0; i < shape[0]; ++i) {
                for (size_t j = 0; j < shape[1]; ++j) {
                    // Calculate exact memory positions using strides
                    size_t in_idx = i * in_strides[0] + j * in_strides[1];
                    size_t out_idx = i * out_strides[0] + j * out_strides[1];
                    
                    out_data[out_idx] = std::max(0.0f, in_data[in_idx]);
                }
            }
        }
        else if (op.op_name == "add") {
            float* a = (float*)op.inputs[0]->data();
            float* b = (float*)op.inputs[1]->data();
            for(size_t i=0; i<n; ++i) out[i] = a[i] + b[i];
        }
        else if (op.op_name == "mul") {
            float* a = (float*)op.inputs[0]->data();
            float* b = (float*)op.inputs[1]->data();
            for(size_t i=0; i<n; ++i) out[i] = a[i] * b[i];
        }
        else if (op.op_name == "matmul") {
    float* A = (float*)op.inputs[0]->data();
    float* B = (float*)op.inputs[1]->data();
    float* C = out;
    int M = (int)op.inputs[0]->shape()[0];
    int K = (int)op.inputs[0]->shape()[1];
    int N = (int)op.inputs[1]->shape()[1];

    #if defined(__APPLE__) || defined(LUMEN_USE_OPENBLAS)
        // Use optimized BLAS
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
    #else
        // Fallback to naive loop
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (int k = 0; k < K; ++k) {
                    sum += A[i * K + k] * B[k * N + j];
                }
                C[i * N + j] = sum;
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