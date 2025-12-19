#include <lumen/lumen.hpp>
#include <iostream>
#include <chrono>
#include <vector>

int main() {
    lumen::Runtime rt;

    // The GPU Shader string
    std::string shader = R"(
        #include <metal_stdlib>
        using namespace metal;
        kernel void multiply_by_two(device float* data [[buffer(0)]], uint id [[thread_position_in_grid]]) {
            data[id] = data[id] * 2.0;
        }
    )";

    // --- TEST 1: Small Task (Should trigger CPU) ---
    size_t smallSize = 500;
    lumen::Buffer* smallBuf = rt.alloc(smallSize * sizeof(float));
    
    std::cout << "\n[Test 1] Dispatching Small Workload..." << std::endl;
    auto start1 = std::chrono::high_resolution_clock::now();
    
    rt.run_task_smart(smallBuf, shader);
    
    auto end1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed1 = end1 - start1;
    std::cout << ">> CPU Execution time: " << elapsed1.count() << " ms" << std::endl;

    // --- TEST 2: Large Task (Should trigger GPU) ---
    size_t largeSize = 200000;
    lumen::Buffer* largeBuf = rt.alloc(largeSize * sizeof(float));

    std::cout << "\n[Test 2] Dispatching Large Workload..." << std::endl;
    auto start2 = std::chrono::high_resolution_clock::now();
    
    rt.run_task_smart(largeBuf, shader);
    
    auto end2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed2 = end2 - start2;
    std::cout << ">> GPU Execution time: " << elapsed2.count() << " ms" << std::endl;


    std::cout << "\n[Test 2 - Run 1] Dispatching Large Workload (Cold Start)..." << std::endl;
    auto s2 = std::chrono::high_resolution_clock::now();
    rt.run_task_smart(largeBuf, shader);
    auto e2 = std::chrono::high_resolution_clock::now();
    std::cout << ">> GPU Cold Run: " << std::chrono::duration<double, std::milli>(e2-s2).count() << " ms" << std::endl;

    std::cout << "\n[Test 2 - Run 2] Dispatching Large Workload (Cached)..." << std::endl;
    auto s3 = std::chrono::high_resolution_clock::now();
    rt.run_task_smart(largeBuf, shader); // This should be much faster now!
    auto e3 = std::chrono::high_resolution_clock::now();
    std::cout << ">> GPU Cached Run: " << std::chrono::duration<double, std::milli>(e3-s3).count() << " ms" << std::endl;

    // Cleanup
    delete smallBuf;
    delete largeBuf;

    return 0;
}