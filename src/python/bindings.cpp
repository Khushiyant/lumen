// lumen/src/python/bindings.cpp

#include "lumen/lumen.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(lumen_py, m) {
  m.doc() = "Lumen Intelligent Heterogeneous Runtime";

  // 1. Bind the Event class
  py::class_<lumen::Event, std::shared_ptr<lumen::Event>>(m, "Event")
      .def("wait", &lumen::Event::wait)
      .def("is_completed", &lumen::Event::is_completed);

  // 2. Bind OpAttributes
  py::class_<lumen::OpAttributes>(m, "OpAttributes")
      .def(py::init<>())
      .def_readwrite("int_attrs", &lumen::OpAttributes::int_attrs)
      .def_readwrite("float_attrs", &lumen::OpAttributes::float_attrs)
      .def_readwrite("bool_attrs", &lumen::OpAttributes::bool_attrs)
      .def_readwrite("string_attrs", &lumen::OpAttributes::string_attrs);

  // 3. Bind the Buffer class with shared_ptr holder
  py::class_<lumen::Buffer, std::shared_ptr<lumen::Buffer>>(m, "Buffer")
      .def("data",
           [](std::shared_ptr<lumen::Buffer> b) {
             // Zero-copy access to buffer data
             float *ptr = (float *)b->data();
             std::vector<ssize_t> py_shape, py_strides;

             for (auto d : b->shape())
               py_shape.push_back((ssize_t)d);

             // Convert float-strides to byte-strides for NumPy
             for (auto s : b->strides())
               py_strides.push_back((ssize_t)s * sizeof(float));

             // Passing 'py::cast(b)' as the base ensures the Buffer object
             // stays alive as long as this NumPy array exists.
             return py::array_t<float>(py_shape, py_strides, ptr, py::cast(b));
           })
      .def_property_readonly("shape", &lumen::Buffer::shape);

  // 4. Bind the Runtime class
  py::class_<lumen::Runtime>(m, "Runtime")
      .def(py::init<>())
      .def("alloc", &lumen::Runtime::alloc) // Returns shared_ptr automatically
      .def("execute", &lumen::Runtime::execute, py::arg("op_name"),
           py::arg("inputs"), py::arg("output"),
           py::arg("attrs") = lumen::OpAttributes())
      .def("submit", &lumen::Runtime::submit)
      .def("set_backend", &lumen::Runtime::set_backend)
      .def("current_backend", &lumen::Runtime::current_backend);
}