#include <metal_stdlib>
using namespace metal;

kernel void multiply_by_two(device float* data [[buffer(0)]], 
                           uint id [[thread_position_in_grid]]) {
    // Each GPU thread handles one element of the array
    data[id] = data[id] * 2.0;
}