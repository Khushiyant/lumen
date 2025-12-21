#include "lumen/lumen.hpp"
#include <chrono>
#include <cstring>
#include <iostream>
#include <numeric>

namespace lumen {

// Forward declarations for backend creation functions
std::unique_ptr<Backend> create_cpu_backend();
#ifdef LUMEN_USE_METAL
std::unique_ptr<Backend> create_metal_backend();
#endif
#ifdef LUMEN_USE_CUDA
std::unique_ptr<Backend> create_cuda_backend();
#endif

// --- Buffer Implementation ---
Buffer::Buffer(const std::vector<size_t> &shape,
               const std::vector<size_t> &strides, void *device_ptr,
               void *host_ptr, Backend *creator, size_t offset)
    : shape_(shape), strides_(strides), device_ptr_(device_ptr),
      host_ptr_(host_ptr), offset_(offset), creator_(creator) {}

Buffer::~Buffer() {
  // If we have pending work on this buffer, wait for it before destruction
  if (last_write_event_) {
    last_write_event_->wait();
  }

  if (!is_view_ && runtime_context_) {
    runtime_context_->wait_all(); // Ensure work is done before freeing
  }
  if (!is_view_ && creator_ && device_ptr_) {
    // Pass size to the pool for categorization
    creator_->free_buffer(device_ptr_, this->size_bytes());
  }
}

Buffer *Buffer::view(const std::vector<size_t> &new_shape,
                     const std::vector<size_t> &new_strides,
                     size_t new_offset) {
  Buffer *v = new Buffer(new_shape, new_strides, device_ptr_, host_ptr_,
                         creator_, offset_ + new_offset);
  v->is_view_ = true;
  v->set_runtime(this->runtime_context_);
  // View shares the dependency of the parent
  v->set_last_event(this->last_write_event_);
  return v;
}

void *Buffer::data() {
  if (location_ == BufferLocation::DEVICE_ONLY && runtime_context_) {
    runtime_context_->wait_all(); // Sync before CPU access
  }
  return (char *)host_ptr_ + (offset_ * sizeof(float));
}

// --- Runtime Implementation ---

Runtime::Runtime() {
  // Always initialize CPU backend
  backends_["cpu"] = create_cpu_backend();

#ifdef LUMEN_USE_METAL
  try {
    backends_["metal"] = create_metal_backend();
  } catch (const std::exception& e) {
    std::cerr << "[Lumen] Warning: Metal initialization failed: " << e.what() << std::endl;
  }
#endif

#ifdef LUMEN_USE_CUDA
  try {
    backends_["cuda"] = create_cuda_backend();
  } catch (const std::exception& e) {
    std::cerr << "[Lumen] Warning: CUDA initialization failed: " << e.what() << std::endl;
  }
#endif

  run_startup_benchmarks();
  active_backend_name_ = "dynamic";
  active_backend_ = nullptr;
  router_ = std::make_unique<Router>();
}

Runtime::~Runtime() { wait_all(); }

void Runtime::run_startup_benchmarks() {
  for (auto &[name, backend] : backends_) {
    size_t dim = 256;
    auto start = std::chrono::high_resolution_clock::now();
    auto tA = std::shared_ptr<Buffer>(backend->create_buffer({dim, dim}));
    auto tB = std::shared_ptr<Buffer>(backend->create_buffer({dim, dim}));
    auto tC = std::shared_ptr<Buffer>(backend->create_buffer({dim, dim}));

    QueuedOp startup_op;
    startup_op.op_name = "matmul";
    startup_op.inputs = {tA, tB};
    startup_op.output = tC;
    startup_op.target_backend = name;
    std::vector<QueuedOp> startup_queue = {startup_op};
    auto ev = backend->sync(startup_queue);

    if (ev)
      ev->wait();

    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();

    metrics_[name]["matmul"] = {ms, (double)(dim * dim * dim) / (ms * 1e3)};

  }
}

size_t Buffer::num_elements() const {
  size_t total = 1;
  for (auto d : shape_)
    total *= d;
  return total;
}

size_t Buffer::size_bytes() const { return num_elements() * sizeof(float); }

void Runtime::execute(const std::string &op_name,
                      const std::vector<std::shared_ptr<Buffer>> &inputs,
                      std::shared_ptr<Buffer> output,
                      const OpAttributes &attrs) {
  Backend *target = active_backend_
                        ? active_backend_
                        : router_->select_backend(op_name, output->shape(),
                                                  backends_, metrics_);

  std::string target_name = active_backend_name_;
  if (!active_backend_) {
    for (auto &[n, p] : backends_)
      if (p.get() == target)
        target_name = n;
  }

  // --- CROSS-BACKEND SYNCHRONIZATION ---
  // If any input is still being processed by a different backend (or has a
  // pending event), we must wait for it to ensure data consistency.
  for (const auto &in : inputs) {
    if (in->get_last_event()) {
      if (!in->get_last_event()->is_completed()) {
        in->get_last_event()->wait();
      }
    }
  }

  QueuedOp op;
  op.op_name = op_name;
  op.inputs = inputs;
  op.output = output;
  op.attrs = attrs;
  op.target_backend = target_name;
  queue_.push_back(op);

  output->set_location(BufferLocation::DEVICE_ONLY);
}

std::vector<std::shared_ptr<Event>> Runtime::submit() {
  if (queue_.empty())
    return {};

  std::vector<QueuedOp> current_queue = std::move(queue_);
  queue_.clear();

  std::vector<std::shared_ptr<Event>> new_events;
  size_t i = 0;
  while (i < current_queue.size()) {
    std::string target = current_queue[i].target_backend;
    std::vector<QueuedOp> group;

    while (i < current_queue.size() &&
           current_queue[i].target_backend == target) {
      group.push_back(current_queue[i]);
      i++;
    }

    if (backends_.count(target)) {
      auto ev = backends_[target]->sync(group);
      if (ev) {
        new_events.push_back(ev);
        inflight_events_.push_back(ev);

        // Tag outputs with this event for future synchronization
        for (auto &op : group) {
          op.output->set_last_event(ev);
        }
      }
    }

    for (auto &op : group) {
      op.output->set_location(BufferLocation::BOTH_SYNCED);
    }
  }
  return new_events;
}

void Runtime::wait_all() {
  for (auto &ev : inflight_events_) {
    if (ev)
      ev->wait();
  }
  inflight_events_.clear();
}

// lumen/src/core/runtime.cpp

std::shared_ptr<Buffer> Runtime::alloc(const std::vector<size_t> &shape) {
  // Select the highest-performance backend available for allocation
  Backend *allocator = backends_.count("cuda") ? backends_["cuda"].get()
                                               : (backends_.count("metal")
                                                      ? backends_["metal"].get()
                                                      : backends_["cpu"].get());

  // Create the raw buffer from the backend
  Buffer *raw_buf = allocator->create_buffer(shape);

  // Wrap it in a shared_ptr with a standard deleter
  // The Buffer destructor will automatically handle
  // creator_->free_buffer(device_ptr_)
  auto buf = std::shared_ptr<Buffer>(raw_buf);

  buf->set_runtime(this);
  return buf;
}

void Runtime::set_backend(const std::string &name) {
  if (backends_.count(name)) {
    active_backend_ = backends_[name].get();
    active_backend_name_ = name;
  } else if (name == "dynamic") {
    active_backend_ = nullptr;
    active_backend_name_ = "dynamic";
  } else {
    throw std::runtime_error("Backend not found: " + name);
  }
}

std::string Runtime::current_backend() const {
  return active_backend_ ? active_backend_name_ : "dynamic";
}
} // namespace lumen