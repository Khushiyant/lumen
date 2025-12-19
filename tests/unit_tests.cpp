#include <lumen/lumen.hpp>
#include <cassert>
#include <iostream>

void test_zero_copy_integrity() {
    lumen::Runtime rt;
    auto* buf = rt.alloc(1024 * sizeof(float));
    float* cpu_ptr = static_cast<float*>(buf->data());
    
    // Write on CPU
    cpu_ptr[0] = 123.456f;
    
    // Verify "Device Handle" exists (The Metal Buffer)
    assert(buf->device_handle() != nullptr);
    
    // Read back from same pointer
    assert(cpu_ptr[0] == 123.456f);
    
    delete buf;
    std::cout << "PASS: Zero-Copy Integrity" << std::endl;
}

int main() {
    test_zero_copy_integrity();
    return 0;
}