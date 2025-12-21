#include "lumen/lumen.hpp"
#include "lumen/op_registry.hpp"
#include <iostream>
#include <sstream>
#include <cstring>

namespace lumen {

class CPUEvent : public Event {
public:
  void wait() override {}
  bool is_completed() override { return true; }
};

class CPUBackend : public Backend, public BackendWithRegistry {
public:
  CPUBackend() {
    register_standard_ops();
    register_backend_ops();

#ifdef LUMEN_USE_OPENBLAS
    std::cout << "[Lumen] CPU Backend Initialized (OpenBLAS) - "
              << supported_ops().size() << " ops" << std::endl;
#elif defined(__APPLE__)
    std::cout << "[Lumen] CPU Backend Initialized (Accelerate) - "
              << supported_ops().size() << " ops" << std::endl;
#else
    std::cout << "[Lumen] CPU Backend Initialized (Naive) - "
              << supported_ops().size() << " ops" << std::endl;
#endif
  }

  Buffer *create_buffer(const std::vector<size_t> &shape) override {
    size_t total_elements = 1;
    for (auto d : shape)
      total_elements *= d;
    size_t size = total_elements * sizeof(float);
    void *ptr = pool_.acquire(size);
    if (!ptr)
      ptr = new float[total_elements];
    std::vector<size_t> strides(shape.size());
    size_t s = 1;
    for (int i = (int)shape.size() - 1; i >= 0; --i) {
      strides[i] = s;
      s *= shape[i];
    }
    return new Buffer(shape, strides, ptr, ptr, this, 0);
  }

  void copy_h2d(void *host_ptr, void *device_ptr, size_t size) override {
    std::memcpy(device_ptr, host_ptr, size);
  }

  void copy_d2h(void *device_ptr, void *host_ptr, size_t size) override {
    std::memcpy(host_ptr, device_ptr, size);
  }

  void free_buffer(void *device_ptr, size_t size) override {
    if (device_ptr)
      pool_.release(device_ptr, size);
  }

  void execute(const std::string &op_name,
               const std::vector<std::shared_ptr<Buffer>> &inputs,
               std::shared_ptr<Buffer> output)
      override { 
    std::vector<QueuedOp> q = {
        {op_name, inputs, output, {}, ""}}; 
    sync(q);
  }

  std::shared_ptr<Event> sync(std::vector<QueuedOp> &queue) override {
    for (const auto &op : queue) {
      std::vector<std::string> sub_ops;
      std::string segment;
      std::stringstream ss(op.op_name);
      while (std::getline(ss, segment, '_')) {
        if (!sub_ops.empty()) {
          std::string potential_fused = sub_ops.back() + "_" + segment;
          if (supports_op(potential_fused)) {
            sub_ops.back() = potential_fused;
            continue;
          }
        }
        sub_ops.push_back(segment);
      }

      for (size_t i = 0; i < sub_ops.size(); ++i) {
        OpContext ctx;
        if (i == 0) {
          for (auto &in_sh : op.inputs)
            ctx.inputs.push_back(in_sh.get());
        } else {
          ctx.inputs = {op.output.get()};
        }
        ctx.output = op.output.get();
        ctx.attrs = op.attrs;

        execute_op(sub_ops[i], ctx);
      }
    }
    return std::make_shared<CPUEvent>();
  }

protected:
  void register_backend_ops() override {
    // CPU can add optimized versions here if needed
  }
};

std::unique_ptr<Backend> create_cpu_backend() {
  return std::make_unique<CPUBackend>();
}

} // namespace lumen