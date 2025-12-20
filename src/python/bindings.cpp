#include "lumen/lumen.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(lumen_py, m) {
  m.doc() = "Lumen Intelligent Heterogeneous Runtime";

  py::class_<lumen::Buffer>(m, "Buffer")
      .def(
          "data",
          [](lumen::Buffer &b) {
            // Managed Access: data() triggers JIT sync if needed
            float *ptr = (float *)b.data();

            // Return as NumPy array for zero-copy access
            std::vector<ssize_t> py_shape;
            for (auto d : b.shape())
              py_shape.push_back((ssize_t)d);

            // keep_alive<0, 1> ensures the Buffer stays alive as long as this
            // numpy array exists
            return py::array_t<float>(py_shape, ptr);
          },
          py::keep_alive<0, 1>())
      .def_property_readonly("shape", &lumen::Buffer::shape);

  py::class_<lumen::Runtime>(m, "Runtime")
      .def(py::init<>())
      .def("alloc", &lumen::Runtime::alloc,
           py::return_value_policy::take_ownership,
           py::keep_alive<0, 1>()) // Buffer (0) keeps Runtime (1) alive
      .def("execute", &lumen::Runtime::execute)
      .def("submit", &lumen::Runtime::submit)
      .def("current_backend", &lumen::Runtime::current_backend);
}