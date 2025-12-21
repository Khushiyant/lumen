// src/python/bindings.cpp - Complete Python Bindings for Lumen

#include "lumen/graph.hpp"
#include "lumen/lumen.hpp"
#include "lumen/onnx_importer.hpp"
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(lumen_py, m) {
  m.doc() = "Lumen: Intelligent Heterogeneous Deep Learning Runtime";

  // ============================================================================
  // CORE RUNTIME COMPONENTS
  // ============================================================================

  // Event class for async operations
  py::class_<lumen::Event, std::shared_ptr<lumen::Event>>(
      m, "Event", "Represents an asynchronous operation that can be waited on")
      .def("wait", &lumen::Event::wait, "Block until the operation completes")
      .def("is_completed", &lumen::Event::is_completed,
           "Check if the operation has completed without blocking");

  // OpAttributes for operation configuration
  py::class_<lumen::OpAttributes>(
      m, "OpAttributes",
      "Attributes for configuring operations (stride, padding, etc.)")
      .def(py::init<>())
      .def_readwrite("int_attrs", &lumen::OpAttributes::int_attrs,
                     "Dictionary of integer attributes")
      .def_readwrite("float_attrs", &lumen::OpAttributes::float_attrs,
                     "Dictionary of float attributes")
      .def_readwrite("int_array_attrs", &lumen::OpAttributes::int_array_attrs,
                     "Dictionary of integer array attributes")
      .def_readwrite("string_attrs", &lumen::OpAttributes::string_attrs,
                     "Dictionary of string attributes")
      .def_readwrite("bool_attrs", &lumen::OpAttributes::bool_attrs,
                     "Dictionary of boolean attributes")
      .def("get_int", &lumen::OpAttributes::get_int, py::arg("key"),
           py::arg("default_val") = 0,
           "Get integer attribute with default value")
      .def("get_float", &lumen::OpAttributes::get_float, py::arg("key"),
           py::arg("default_val") = 0.0f,
           "Get float attribute with default value")
      .def("get_bool", &lumen::OpAttributes::get_bool, py::arg("key"),
           py::arg("default_val") = false,
           "Get boolean attribute with default value")
      .def("get_int_array", &lumen::OpAttributes::get_int_array, py::arg("key"),
           "Get integer array attribute");

  // Buffer class with numpy integration
  py::class_<lumen::Buffer, std::shared_ptr<lumen::Buffer>>(
      m, "Buffer",
      "Multi-dimensional tensor buffer with zero-copy numpy interface")
      .def(
          "data",
          [](std::shared_ptr<lumen::Buffer> b) {
            // Zero-copy access to buffer data as numpy array
            float *ptr = (float *)b->data();
            std::vector<ssize_t> py_shape, py_strides;

            for (auto d : b->shape())
              py_shape.push_back((ssize_t)d);

            // Convert float-strides to byte-strides for NumPy
            for (auto s : b->strides())
              py_strides.push_back((ssize_t)s * sizeof(float));

            // Pass buffer as base to keep it alive while numpy array exists
            return py::array_t<float>(py_shape, py_strides, ptr, py::cast(b));
          },
          "Get zero-copy numpy array view of buffer data")
      .def_property_readonly("shape", &lumen::Buffer::shape,
                             "Get buffer shape as tuple")
      .def_property_readonly("strides", &lumen::Buffer::strides,
                             "Get buffer strides")
      .def_property_readonly("size", &lumen::Buffer::num_elements,
                             "Get total number of elements")
      .def_property_readonly("nbytes", &lumen::Buffer::size_bytes,
                             "Get size in bytes")
      .def("sync_to_host", &lumen::Buffer::sync_to_host,
           "Synchronize device data to host memory");

  // Runtime class - main entry point
  py::class_<lumen::Runtime>(m, "Runtime",
                             R"doc(
    Main runtime for executing operations on heterogeneous hardware.
    
    Example:
        >>> rt = lumen_py.Runtime()
        >>> rt.set_backend('metal')  # or 'cuda', 'cpu'
        >>> buf = rt.alloc([4, 4])
        >>> buf.data()[:] = np.eye(4)
    )doc")
      .def(py::init<>(), "Initialize runtime with all available backends")
      .def("alloc", &lumen::Runtime::alloc, py::arg("shape"),
           "Allocate a buffer with the given shape")
      .def("execute", &lumen::Runtime::execute, py::arg("op_name"),
           py::arg("inputs"), py::arg("output"),
           py::arg("attrs") = lumen::OpAttributes(),
           R"doc(
         Queue an operation for execution.
         
         Args:
             op_name: Operation name (e.g., 'add', 'matmul', 'conv2d')
             inputs: List of input buffers
             output: Output buffer
             attrs: Optional operation attributes
         )doc")
      .def("submit", &lumen::Runtime::submit,
           "Submit queued operations and return event handles")
      .def("wait_all", &lumen::Runtime::wait_all,
           "Wait for all pending operations to complete")
      .def("set_backend", &lumen::Runtime::set_backend, py::arg("name"),
           "Set active backend ('cpu', 'cuda', 'metal', or 'dynamic')")
      .def("current_backend", &lumen::Runtime::current_backend,
           "Get name of currently active backend");

  // ============================================================================
  // GRAPH IR COMPONENTS
  // ============================================================================

  // DataType enum
  py::enum_<lumen::DataType>(m, "DataType", "Tensor data types")
      .value("FLOAT32", lumen::DataType::FLOAT32)
      .value("FLOAT16", lumen::DataType::FLOAT16)
      .value("INT8", lumen::DataType::INT8)
      .value("INT32", lumen::DataType::INT32)
      .value("INT64", lumen::DataType::INT64)
      .value("BOOL", lumen::DataType::BOOL)
      .export_values();

  // TensorDescriptor class
  py::class_<lumen::TensorDescriptor>(
      m, "TensorDescriptor", "Describes a tensor in the computation graph")
      .def_property_readonly("name", &lumen::TensorDescriptor::name,
                             "Get tensor name")
      .def_property_readonly("shape", &lumen::TensorDescriptor::shape,
                             "Get tensor shape")
      .def_property_readonly("dtype", &lumen::TensorDescriptor::dtype,
                             "Get tensor data type")
      .def_property_readonly("num_elements",
                             &lumen::TensorDescriptor::num_elements,
                             "Get total number of elements")
      .def_property_readonly("size_bytes", &lumen::TensorDescriptor::size_bytes,
                             "Get size in bytes");

  // Graph class
  py::class_<lumen::Graph>(m, "Graph",
                           R"doc(
    Computation graph for defining and optimizing neural networks.
    
    Example:
        >>> graph = lumen_py.Graph()
        >>> x = graph.add_input('input', [1, 784])
        >>> w = graph.add_weight('weight', [784, 10])
        >>> y = graph.add_op('matmul', [x, w])
        >>> graph.mark_output(y)
        >>> graph.optimize()
    )doc")
      .def(py::init<>(), "Create empty computation graph")
      .def("add_input", &lumen::Graph::add_input, py::arg("name"),
           py::arg("shape"), py::arg("dtype") = lumen::DataType::FLOAT32,
           "Add input tensor to graph")
      .def(
          "add_weight",
          [](lumen::Graph &g, const std::string &name,
             const std::vector<size_t> &shape, py::array_t<float> data) {
            const float *data_ptr = nullptr;
            if (data.size() > 0) {
              data_ptr = data.data();
            }
            return g.add_weight(name, shape, data_ptr);
          },
          py::arg("name"), py::arg("shape"),
          py::arg("data") = py::array_t<float>(),
          "Add weight tensor to graph with optional initialization data")
      .def("add_op", &lumen::Graph::add_op, py::arg("op_type"),
           py::arg("inputs"), py::arg("attrs") = lumen::OpAttributes(),
           py::arg("name") = "", "Add operation node to graph")
      .def("mark_output", &lumen::Graph::mark_output, py::arg("tensor"),
           "Mark tensor as graph output")
      .def("optimize", &lumen::Graph::optimize,
           "Run optimization passes (fusion, dead code elimination)")
      .def("compile", &lumen::Graph::compile, py::arg("runtime"),
           py::return_value_policy::take_ownership,
           "Compile graph into executable form")
      .def("print_summary", &lumen::Graph::print_summary,
           "Print graph structure summary")
      .def("__repr__", &lumen::Graph::to_string);

  // ExecutableGraph class
  py::class_<lumen::ExecutableGraph>(
      m, "ExecutableGraph", "Compiled and optimized graph ready for execution")
      .def("execute",
           py::overload_cast<const std::vector<std::shared_ptr<lumen::Buffer>>
                                 &>(&lumen::ExecutableGraph::execute),
           py::arg("inputs"),
           "Execute graph with given inputs and return outputs")
      .def(
          "execute",
          py::overload_cast<const std::vector<std::shared_ptr<lumen::Buffer>> &,
                            const std::vector<std::shared_ptr<lumen::Buffer>>
                                &>(&lumen::ExecutableGraph::execute),
          py::arg("inputs"), py::arg("outputs"),
          "Execute graph with given inputs, writing to provided output buffers")
      .def("profile", &lumen::ExecutableGraph::profile, py::arg("inputs"),
           "Profile graph execution and return timing data")
      .def_property_readonly("memory_usage",
                             &lumen::ExecutableGraph::get_memory_usage,
                             "Get peak memory usage in bytes");

  // ProfilingData struct
  py::class_<lumen::ExecutableGraph::ProfilingData>(
      m, "ProfilingData", "Profiling information for a single operation")
      .def_readonly("node_name",
                    &lumen::ExecutableGraph::ProfilingData::node_name)
      .def_readonly("op_type", &lumen::ExecutableGraph::ProfilingData::op_type)
      .def_readonly("time_ms", &lumen::ExecutableGraph::ProfilingData::time_ms)
      .def_readonly("backend", &lumen::ExecutableGraph::ProfilingData::backend)
      .def("__repr__", [](const lumen::ExecutableGraph::ProfilingData &p) {
        return "<ProfilingData op=" + p.op_type +
               " time=" + std::to_string(p.time_ms) + "ms" +
               " backend=" + p.backend + ">";
      });

  // ============================================================================
  // ONNX IMPORTER
  // ============================================================================

  py::class_<lumen::ONNXImporter>(m, "ONNXImporter",
                                  "Import ONNX models into Lumen graphs")
      .def_static("import_model", &lumen::ONNXImporter::import_model,
                  py::arg("model_path"), "Import ONNX model from file path");

  // ============================================================================
  // HELPER FUNCTIONS
  // ============================================================================

  m.def(
      "create_conv2d_attrs",
      [](const std::vector<int> &stride, const std::vector<int> &padding,
         const std::vector<int> &dilation) {
        lumen::OpAttributes attrs;
        attrs.int_array_attrs["stride"] = stride;
        attrs.int_array_attrs["padding"] = padding;
        if (!dilation.empty()) {
          attrs.int_array_attrs["dilation"] = dilation;
        }
        return attrs;
      },
      py::arg("stride") = std::vector<int>{1, 1},
      py::arg("padding") = std::vector<int>{0, 0},
      py::arg("dilation") = std::vector<int>(),
      "Create attributes for conv2d operation");

  m.def(
      "create_pool2d_attrs",
      [](const std::vector<int> &kernel_size, const std::vector<int> &stride,
         const std::vector<int> &padding) {
        lumen::OpAttributes attrs;
        attrs.int_array_attrs["kernel_size"] = kernel_size;
        attrs.int_array_attrs["stride"] = stride;
        attrs.int_array_attrs["padding"] = padding;
        return attrs;
      },
      py::arg("kernel_size") = std::vector<int>{2, 2},
      py::arg("stride") = std::vector<int>(),
      py::arg("padding") = std::vector<int>{0, 0},
      "Create attributes for pooling operations");

  // ============================================================================
  // VERSION AND INFO
  // ============================================================================

  m.attr("__version__") = "0.1.0";

  m.def(
      "get_available_backends",
      []() {
        std::vector<std::string> backends = {"cpu"};
#ifdef LUMEN_USE_CUDA
        backends.push_back("cuda");
#endif
#ifdef LUMEN_USE_METAL
        backends.push_back("metal");
#endif
        return backends;
      },
      "Get list of available backends");

  m.def(
      "supports_cuda",
      []() {
#ifdef LUMEN_USE_CUDA
        return true;
#else
        return false;
#endif
      },
      "Check if CUDA support is compiled");

  m.def(
      "supports_metal",
      []() {
#ifdef LUMEN_USE_METAL
        return true;
#else
        return false;
#endif
      },
      "Check if Metal support is compiled");
}