#include "lumen/lumen.hpp"
#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>

namespace lumen {

// CUDA Kernels for Element-wise operations
__global__ void add_kernel(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

__global__ void mul_kernel(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] * b[i];
}

class CUDABackend : public Backend {
public:
    CUDABackend() {
        cudaError_t err = cudaSetDevice(0);
        if (err != cudaSuccess) {
            std::cerr << "[Lumen] Error: No CUDA device found!" << std::endl;
        }
        cublasCreate(&cublas_handle_);
        std::cout << "[Lumen] CUDA Backend Initialized" << std::endl;
    }

    ~CUDABackend() {
        cublasDestroy(cublas_handle_);
    }

    Buffer* create_buffer(const std::vector<size_t>& shape) override {
        size_t total_elements = 1;
        for (auto d : shape) total_elements *= d;
        size_t size = total_elements * sizeof(float);

        void* d_ptr = nullptr;
        cudaMalloc(&d_ptr, size);
        
        // Host pointer is nullptr because CUDA memory is discrete (not unified like Metal)
        // We will add migration logic in Phase 2.
        return new Buffer(shape, d_ptr, nullptr, this);
    }

    void free_buffer(void* device_ptr) override {
        if (device_ptr) cudaFree(device_ptr);
    }

    void execute(const std::string& op_name, const std::vector<Buffer*>& inputs, Buffer* output) override {
        std::vector<QueuedOp> q = {{op_name, inputs, output}};
        sync(q);
    }

    void sync(std::vector<QueuedOp>& queue) override {
        for (const auto& op : queue) {
            float* d_out = static_cast<float*>(op.output->device_handle());
            size_t n = op.output->size_bytes() / sizeof(float);

            if (op.op_name == "add" || op.op_name == "mul") {
                float* d_a = static_cast<float*>(op.inputs[0]->device_handle());
                float* d_b = static_cast<float*>(op.inputs[1]->device_handle());

                int threads = 256;
                int blocks = (n + threads - 1) / threads;

                if (op.op_name == "add") {
                    add_kernel<<<blocks, threads>>>(d_a, d_b, d_out, n);
                } else {
                    mul_kernel<<<blocks, threads>>>(d_a, d_b, d_out, n);
                }
            } 
            else if (op.op_name == "matmul") {
                // CuBLAS assumes Column-Major. We use standard trick: C = B^T * A^T
                // Or simply pass A and B but swap dimensions to match Row-Major expectation.
                // Here we perform simple Row-Major mapped Sgemm:
                float* d_a = static_cast<float*>(op.inputs[0]->device_handle());
                float* d_b = static_cast<float*>(op.inputs[1]->device_handle());
                
                int M = (int)op.inputs[0]->shape()[0];
                int K = (int)op.inputs[0]->shape()[1];
                int N = (int)op.inputs[1]->shape()[1];

                const float alpha = 1.0f;
                const float beta = 0.0f;

                // Note: cuBLAS is column-major, so we compute C^T = B^T * A^T
                // effectively swapping A and B and dimensions
                cublasSgemm(cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                            N, M, K,
                            &alpha,
                            d_b, N,  // B is first
                            d_a, K,  // A is second
                            &beta,
                            d_out, N);
            }
        }
        cudaDeviceSynchronize();
    }

private:
    cublasHandle_t cublas_handle_;
};

std::unique_ptr<Backend> create_cuda_backend() {
    return std::make_unique<CUDABackend>();
}

} // namespace lumen