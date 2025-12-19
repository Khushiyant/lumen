#include "lumen/lumen.hpp"
#include <iostream>

namespace lumen {
Backend* Router::select_backend(const std::string& op_name, 
                                const std::vector<size_t>& shape, 
                                const std::map<std::string, std::unique_ptr<Backend>>& backends) {
    
    size_t total_elements = 1;
    for (auto d : shape) total_elements *= d;

    // Use GPU for heavy operations (> 10k elements or complex MatMuls)
    bool is_heavy = (total_elements > 10000) || (op_name == "matmul" && total_elements > 2500);

    if (backends.count("cuda") && is_heavy) {
        return backends.at("cuda").get();
    }

    // FIX: Only switch to Metal if the operation is heavy enough
    if (backends.count("metal") && is_heavy) {
         return backends.at("metal").get();
    }

    return backends.at("cpu").get();
}
}