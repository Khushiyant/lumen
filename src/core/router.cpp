#include "lumen/lumen.hpp"

namespace lumen {
Backend *Router::select_backend(
    const std::string &op_name, const std::vector<size_t> &shape,
    const std::map<std::string, std::unique_ptr<Backend>> &backends,
    const std::map<std::string, std::map<std::string, BackendMetrics>>
        &metrics) {

  size_t total_elements = 1;
  for (auto d : shape)
    total_elements *= d;

  std::string best_name = "cpu";
  double min_expected_ms = 1e9;

  for (auto const &[name, backend] : backends) {
    if (metrics.count(name) && metrics.at(name).count(op_name)) {
      auto m = metrics.at(name).at(op_name);
      // Prediction: Latency + (Size / Throughput)
      double expected =
          m.kernel_latency_ms + (total_elements / m.throughput_mops / 1000.0);

      if (expected < min_expected_ms) {
        min_expected_ms = expected;
        best_name = name;
      }
    }
  }

  return backends.at(best_name).get();
}
} // namespace lumen