#include "operations/Expression.hpp"
#include "python/pybind11.hpp"

#include <pybind11/operators.h>

namespace mqt {

void registerVariable(py::module& m) {
  py::class_<sym::Variable>(m, "Variable", "A symbolic variable.")
      .def(py::init<std::string>(), "name"_a = "",
           "Create a variable with a given variable name. Variables are "
           "uniquely identified by their name, so if a variable with the same "
           "name already exists, the existing variable will be returned.")
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
