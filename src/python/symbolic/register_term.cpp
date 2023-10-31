#include "operations/Expression.hpp"
#include "pybind11/operators.h"
#include "python/pybind11.hpp"

#include <sstream>

namespace mqt {

void registerTerm(py::module& m) {
  py::class_<sym::Term<double>>(
      m, "Term",
      "A symbolic term which consists of a variable with a given coefficient.")
      .def(py::init<sym::Variable, double>(), "variable"_a,
           "coefficient"_a = 1.0,
           "Create a term with a given coefficient and variable.")
      .def_property_readonly("variable", &sym::Term<double>::getVar,
                             "Return the variable of this term.")
      .def_property_readonly("coefficient", &sym::Term<double>::getCoeff,
                             "Return the coefficient of this term.")
      .def("has_zero_coefficient", &sym::Term<double>::hasZeroCoeff,
           "Return true if the coefficient of this term is zero.")
      .def("add_coefficient", &sym::Term<double>::addCoeff, "coeff"_a,
           "Add `coeff` to the coefficient of this term.")
      .def("evaluate", &sym::Term<double>::evaluate, "assignment"_a,
           "Return the value of this term given by multiplying the coefficient "
           "of this term to the variable value dictated by the assignment.")
      .def(py::self * double())
      .def(double() * py::self)
      .def(py::self / double())
      .def(double() / py::self)
      .def(py::self == py::self)
      .def(py::self != py::self)
      .def(hash(py::self))
      .def("__str__",
           [](const sym::Term<double>& term) {
             std::stringstream ss;
             ss << term;
             return ss.str();
           })
      .def("__repr__", [](const sym::Term<double>& term) {
        std::stringstream ss;
        ss << term;
        return ss.str();
      });
}
} // namespace mqt
