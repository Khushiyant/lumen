#include "lumen/lumen.hpp"
#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdexcept>
#include <string>

namespace lumen {

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
        int deviceCount = 0;
        cudaGetDeviceCount(&deviceCount);
        if (deviceCount == 0) throw std::runtime_error("No CUDA devices found.");

        cudaSetDevice(0);
        cublasCreate(&cublas_handle_);
        std::cout << "[Lumen] CUDA Backend Initialized with Memory Pooling" << std::endl;
    }

    ~CUDABackend() {
        if (cublas_handle_) cublasDestroy(cublas_handle_);
    }

    Buffer* create_buffer(const std::vector<size_t>& shape) override {
        size_t total_elements = 1;
        for (auto d : shape) total_elements *= d;
        size_t size = total_elements * sizeof(float);

        // 1. Try pool first
        void* ptr = pool_.acquire(size);
        if (!ptr) {
            // 2. Allocate Unified Memory if pool empty
            cudaMallocManaged(&ptr, size);
        }

        std::vector<size_t> strides(shape.size());
        size_t s = 1;
        for (int i = (int)shape.size() - 1; i >= 0; --i) {
            strides[i] = s;
            s *= shape[i];
        }

        return new Buffer(shape, strides, ptr, ptr, this, 0);
    }

    // FIXED: Updated signature to match Backend base class
    void free_buffer(void* device_ptr, size_t size) override {
        if (device_ptr) {
            pool_.release(device_ptr, size);
        }
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

                if (op.op_name == "add") add_kernel<<<blocks, threads>>>(d_a, d_b, d_out, n);
                else mul_kernel<<<blocks, threads>>>(d_a, d_b, d_out, n);
            } 
            else if (op.op_name == "matmul") {
                float* d_a = static_cast<float*>(op.inputs[0]->device_handle());
                float* d_b = static_cast<float*>(op.inputs[1]->device_handle());
                int M = (int)op.inputs[0]->shape()[0], K = (int)op.inputs[0]->shape()[1], N = (int)op.inputs[1]->shape()[1];
                const float alpha = 1.0f, beta = 0.0f;
                cublasSgemm(cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_b, N, d_a, K, &beta, d_out, N);
            }
        }
        cudaDeviceSynchronize();
    }

private:
    cublasHandle_t cublas_handle_ = nullptr;
};

std::unique_ptr<Backend> create_cuda_backend() {
    return std::make_unique<CUDABackend>();
}

} // namespace lumen