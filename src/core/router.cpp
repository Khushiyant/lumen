#include "lumen/lumen.hpp"
#include <iostream>

namespace lumen {

// --- FIX: Implement the method defined in the header ---
Backend* Router::select_backend(const std::string& op_name, 
                                const std::vector<size_t>& shape, 
                                const std::map<std::string, std::unique_ptr<Backend>>& backends) {
    
    // 1. If CUDA is available, check if the job is big enough
    if (backends.count("cuda")) {
        size_t total_elements = 1;
        for (auto d : shape) total_elements *= d;

        // HEURISTIC: Use GPU for > 10k elements or heavy MatMuls
        bool is_heavy = (total_elements > 10000) || (op_name == "matmul" && total_elements > 2500);
        
        if (is_heavy) {
            return backends.at("cuda").get();
        }
    }

    // 2. If Metal is available (macOS), prefer it for medium jobs
    if (backends.count("metal")) {
         return backends.at("metal").get();
    }

    // 3. Fallback: CPU
    // Small operations are faster on CPU (avoid PCIe latency)
    return backends.at("cpu").get();
}

} // namespace lumen