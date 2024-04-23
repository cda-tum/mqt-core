#include "operations/Expression.hpp"
#include "python/pybind11.hpp"

#include <pybind11/operators.h>

namespace mqt {

void registerVariable(py::module& m) {
  py::class_<sym::Variable>(m, "Variable")
      .def(py::init<std::string>(), "name"_a = "")
      .def_property_readonly("name", &sym::Variable::getName)
      .def("__str__", &sym::Variable::getName)
      .def("__repr__", &sym::Variable::getName)
      .def(py::self == py::self)
      .def(py::self != py::self)
      .def(hash(py::self))
      .def(py::self < py::self)
      .def(py::self > py::self);
}
} // namespace mqt
