#include "lumen/lumen.hpp"
#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdexcept> // Required for runtime_error

namespace lumen {

// Simple CUDA kernels
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
        // 1. Check for Device
        int deviceCount = 0;
        cudaError_t err = cudaGetDeviceCount(&deviceCount);
        if (err != cudaSuccess || deviceCount == 0) {
            throw std::runtime_error("No CUDA-capable device detected.");
        }

        // 2. Set Device
        err = cudaSetDevice(0);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to set CUDA device 0.");
        }

        // 3. Initialize cuBLAS
        cublasStatus_t stat = cublasCreate(&cublas_handle_);
        if (stat != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("Failed to initialize cuBLAS.");
        }

        std::cout << "[Lumen] CUDA Backend Initialized (Unified Memory)" << std::endl;
    }

    ~CUDABackend() {
        cublasDestroy(cublas_handle_);
    }

    Buffer* create_buffer(const std::vector<size_t>& shape) override {
        size_t total_elements = 1;
        for (auto d : shape) total_elements *= d;
        size_t size = total_elements * sizeof(float);

        void* ptr = nullptr;
        // FIX: Use Managed Memory so CPU and GPU can both access it (Zero-Copy)
        cudaError_t err = cudaMallocManaged(&ptr, size);
        
        if (err != cudaSuccess) {
            std::cerr << "[Lumen] CUDA Alloc Error: " << cudaGetErrorString(err) << std::endl;
            return nullptr;
        }

        // In Unified Memory, device_ptr == host_ptr
        return new Buffer(shape, ptr, ptr, this);
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

            // Synchronize to ensure CPU writes are visible to GPU
            // (Strictly speaking, cudaMallocManaged handles this on kernel launch, 
            // but explicit sync helps debugging)
            // cudaDeviceSynchronize(); 

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
                float* d_a = static_cast<float*>(op.inputs[0]->device_handle());
                float* d_b = static_cast<float*>(op.inputs[1]->device_handle());
                
                // cuBLAS expects column-major. We use C = B^T * A^T trick for row-major.
                // Or standard sgemm if we treat data as row-major and swap dimensions.
                int M = (int)op.inputs[0]->shape()[0];
                int K = (int)op.inputs[0]->shape()[1];
                int N = (int)op.inputs[1]->shape()[1];

                const float alpha = 1.0f;
                const float beta = 0.0f;

                cublasSgemm(cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                            N, M, K,
                            &alpha,
                            d_b, N,  
                            d_a, K,  
                            &beta,
                            d_out, N);
            }
        }
        // Essential: Wait for GPU to finish so CPU can read results immediately
        cudaDeviceSynchronize();
    }

private:
    cublasHandle_t cublas_handle_;
};

std::unique_ptr<Backend> create_cuda_backend() {
    return std::make_unique<CUDABackend>();
}

} // namespace lumen