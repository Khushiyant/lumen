#pragma once
#include <chrono>
#include <cstddef>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace lumen {

// Pillar 1: Foundational Memory Pool
class MemoryPool {
public:
  struct Block {
    void *ptr;
    size_t size;
  };
  void *acquire(size_t size) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto it = free_blocks_.begin(); it != free_blocks_.end(); ++it) {
      if (it->size >= size && it->size <= size * 1.2) {
        void *ptr = it->ptr;
        free_blocks_.erase(it);
        return ptr;
      }
    }
    return nullptr;
  }
  void release(void *ptr, size_t size) {
    std::lock_guard<std::mutex> lock(mutex_);
    free_blocks_.push_back({ptr, size});
  }

private:
  std::list<Block> free_blocks_;
  std::mutex mutex_;
};

// Pillar 2: Asynchronous Events
class Event {
public:
  virtual ~Event() = default;
  virtual void wait() = 0;
  virtual bool is_completed() = 0;
};

class Buffer;
class Backend;
class Router;
class Runtime;

enum class BufferLocation { HOST_ONLY, DEVICE_ONLY, BOTH_SYNCED };

struct OpAttributes {
  std::map<std::string, int> int_attrs;
  std::map<std::string, float> float_attrs;
  std::map<std::string, std::vector<int>> int_array_attrs;
  std::map<std::string, std::string> string_attrs;
  std::map<std::string, bool> bool_attrs;

  // Convenience getters with defaults
  int get_int(const std::string &key, int default_val = 0) const {
    auto it = int_attrs.find(key);
    return it != int_attrs.end() ? it->second : default_val;
  }

  float get_float(const std::string &key, float default_val = 0.0f) const {
    auto it = float_attrs.find(key);
    return it != float_attrs.end() ? it->second : default_val;
  }

  std::vector<int> get_int_array(const std::string &key) const {
    auto it = int_array_attrs.find(key);
    return it != int_array_attrs.end() ? it->second : std::vector<int>{};
  }

  bool get_bool(const std::string &key, bool default_val = false) const {
    auto it = bool_attrs.find(key);
    return it != bool_attrs.end() ? it->second : default_val;
  }
};

struct QueuedOp {
  std::string op_name;
  std::vector<Buffer *> inputs;
  Buffer *output;
  OpAttributes attrs; 
  std::string target_backend;
};

struct BackendMetrics {
  double kernel_latency_ms;
  double throughput_mops;
};

class Backend {
public:
  virtual ~Backend() = default;
  virtual Buffer *create_buffer(const std::vector<size_t> &shape) = 0;
  virtual void free_buffer(void *device_ptr, size_t size) = 0;

  virtual void execute(const std::string &op_name,
                       const std::vector<Buffer *> &inputs, Buffer *output) = 0;

  virtual std::shared_ptr<Event> sync(std::vector<QueuedOp> &queue) = 0;

protected:
  MemoryPool pool_;
};

class Buffer {
public:
  Buffer(const std::vector<size_t> &shape, const std::vector<size_t> &strides,
         void *device_ptr, void *host_ptr, Backend *creator, size_t offset = 0);
  ~Buffer();
  void *data();
  void set_runtime(Runtime *rt) { runtime_context_ = rt; }
  const std::vector<size_t> &shape() const { return shape_; }
  const std::vector<size_t> &strides() const { return strides_; }
  size_t num_elements() const;
  size_t size_bytes() const;
  void *device_handle() const {
    return (char *)device_ptr_ + (offset_ * sizeof(float));
  }
  Backend *creator() const { return creator_; }
  Buffer *view(const std::vector<size_t> &new_shape,
               const std::vector<size_t> &new_strides, size_t new_offset = 0);
  void set_location(BufferLocation loc) { location_ = loc; }
  BufferLocation location() const { return location_; }
  void *device_handle_base() const { return device_ptr_; }

  size_t offset() const { return offset_; }
  size_t offset_bytes() const { return offset_ * sizeof(float); }
  void *device_ptr() const { return device_ptr_; }

private:
  std::vector<size_t> shape_;
  std::vector<size_t> strides_;
  void *device_ptr_;
  void *host_ptr_;
  size_t offset_;
  Backend *creator_;
  Runtime *runtime_context_ = nullptr;
  BufferLocation location_ = BufferLocation::BOTH_SYNCED;
  bool is_view_ = false;
};

class Router {
public:
  Backend *select_backend(
      const std::string &op_name, const std::vector<size_t> &shape,
      const std::map<std::string, std::unique_ptr<Backend>> &backends,
      const std::map<std::string, std::map<std::string, BackendMetrics>>
          &metrics);
};

class Runtime {
public:
  Runtime();
  ~Runtime();
  Buffer *alloc(const std::vector<size_t> &shape);

  void execute(const std::string &op_name, const std::vector<Buffer *> &inputs,
               Buffer *output, const OpAttributes &attrs = {});

  std::vector<std::shared_ptr<Event>> submit();
  void wait_all();
  void set_backend(const std::string &name);
  std::string current_backend() const;

private:
  void run_startup_benchmarks();
  std::map<std::string, std::unique_ptr<Backend>> backends_;
  std::map<std::string, std::map<std::string, BackendMetrics>> metrics_;
  std::vector<QueuedOp> queue_;
  std::vector<std::shared_ptr<Event>> inflight_events_;
  Backend *active_backend_;
  std::string active_backend_name_;
  std::unique_ptr<Router> router_;
};

} // namespace lumen