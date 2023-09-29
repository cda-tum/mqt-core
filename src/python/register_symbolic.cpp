#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "operations/Expression.hpp"
#include <iostream>
#include <sstream>

namespace mqt {
  
namespace py = pybind11;
using namespace pybind11::literals;
  
void registerSymbolic(pybind11::module& m) {
  
    py::class_<sym::Variable>(m, "Variable", "A symbolic variable.")
      .def(py::init<std::string>(), "name"_a = "",
           "Create a variable with a given variable name. Variables are "
           "uniquely identified by their name, so if a variable with the same "
           "name already exists, the existing variable will be returned.")
      .def_property_readonly("name", &sym::Variable::getName)
      .def("__str__", &sym::Variable::getName)
      .def("__eq__", &sym::Variable::operator==)
      .def("__ne__", &sym::Variable::operator!=)
      .def("__lt__", &sym::Variable::operator<)
      .def("__gt__", &sym::Variable::operator>);

  py::class_<sym::Term<double>>(
      m, "Term",
      "A symbolic term which consists of a variable with a given coefficient.")
      .def(py::init<double, sym::Variable>(), "coefficient"_a, "variable"_a,
           "Create a term with a given coefficient and variable.")
      .def(py::init<sym::Variable>(), "variable"_a,
           "Create a term with a given variable and coefficient 1.")
      .def_property_readonly("variable", &sym::Term<double>::getVar,
                             "Return the variable of this term.")
      .def_property_readonly("coefficient", &sym::Term<double>::getCoeff,
                             "Return the coefficient of this term.")
      .def("has_zero_coefficient", &sym::Term<double>::hasZeroCoeff,
           "Return true if the coefficient of this term is zero.")
      .def("add_coefficient", &sym::Term<double>::addCoeff, "coeff"_a,
           "Add coeff a to the coefficient of this term.")
      .def("evaluate", &sym::Term<double>::evaluate, "assignment"_a,
           "Return the value of this term given by multiplying the coefficient "
           "of this term to the variable value dictated by the assignment.")
      .def("__mul__",
           [](const sym::Term<double>& lhs, double rhs) { return lhs * rhs; })
      .def("__rmul__",
           [](sym::Term<double> rhs, const double lhs) { return lhs * rhs; })
      .def("__truediv__",
           [](const sym::Term<double>& lhs, double rhs) { return lhs / rhs; })
      .def("__rtruediv__",
           [](sym::Term<double> rhs, const double lhs) { return lhs / rhs; });

  py::class_<sym::Expression<double, double>>(
      m, "Expression",
      "Class representing a symbolic sum of terms. The expression is of the "
      "form `constant + term_1 + term_2 + ... + term_n`.")
      .def(py::init<>(), "Create an empty expression.")
      .def(
          "__init__",
          [](sym::Expression<double, double>* expr,
             const std::vector<sym::Term<double>>& terms, double constant) {
            new (expr) sym::Expression<double, double>(terms, constant);
          },
          "terms"_a, "constant"_a = 0.0,
          "Create an expression with a given list of terms and a constant (0 "
          "by default).")
      .def(
          "__init__",
          [](sym::Expression<double, double>* expr,
             const sym::Term<double>& term, double constant) {
            new (expr) sym::Expression<double, double>(
                std::vector<sym::Term<double>>{term}, constant);
          },
          "term"_a, "constant"_a = 0.0,
          "Create an expression with a given term and a constant (0 by "
          "default).")
      .def(py::init<double>(), "constant"_a,
           "Create a constant expression involving no symbolic terms.")
      .def_property("constant", &sym::Expression<double, double>::getConst,
                    &sym::Expression<double, double>::setConst)
      .def(
          "__iter__",
          [](const sym::Expression<double, double>& expr) {
            return py::make_iterator(expr.begin(), expr.end());
          },
          py::keep_alive<0, 1>())
      .def("__getitem__",
           [](const sym::Expression<double, double>& expr, std::size_t idx) {
             if (idx >= expr.numTerms()) {
               throw py::index_error();
             }
             return expr.getTerms()[idx];
           })

      .def("is_zero", &sym::Expression<double, double>::isZero,
           "Return true if this expression is zero, i.e., all terms have "
           "coefficient 0 and the constant is 0 as well.")
      .def("is_constant", &sym::Expression<double, double>::isConstant,
           "Return true if this expression is constant, i.e., all terms have "
           "coefficient 0 or no terms are involved.")
      .def("num_terms", &sym::Expression<double, double>::numTerms,
           "Return the number of terms in this expression.")
      .def("__len__", &sym::Expression<double, double>::numTerms)
      .def_property_readonly("terms",
                             &sym::Expression<double, double>::getTerms)
      .def("evaluate", &sym::Expression<double, double>::evaluate,
           "assignment"_a,
           "Return the value of this expression given by summing the values of "
           "all instantiated terms and the constant given by the assignment.")
      // addition operators
      .def("__add__",
           [](const sym::Expression<double, double>& lhs,
              const sym::Expression<double, double>& rhs) { return lhs + rhs; })
      .def("__add__", [](const sym::Expression<double, double>& lhs,
                         const sym::Term<double>& rhs) { return lhs + rhs; })
      .def("__add__", [](const sym::Expression<double, double>& lhs,
                         const double rhs) { return lhs + rhs; })
      .def("__radd__", [](const sym::Expression<double, double>& rhs,
                          const sym::Term<double>& lhs) { return lhs + rhs; })
      .def("__radd__", [](const sym::Expression<double, double>& rhs,
                          const double lhs) { return rhs + lhs; })
      // subtraction operators
      .def("__sub__",
           [](const sym::Expression<double, double>& lhs,
              const sym::Expression<double, double>& rhs) { return lhs - rhs; })
      .def("__sub__", [](const sym::Expression<double, double>& lhs,
                         const sym::Term<double>& rhs) { return lhs - rhs; })
      .def("__sub__", [](const sym::Expression<double, double>& lhs,
                         const double rhs) { return lhs - rhs; })
      .def("__rsub__", [](const sym::Expression<double, double>& rhs,
                          const sym::Term<double>& lhs) { return lhs - rhs; })
      .def("__rsub__", [](const sym::Expression<double, double>& rhs,
                          const double lhs) { return lhs - rhs; })
      // multiplication operators
      .def("__mul__", [](const sym::Expression<double, double>& lhs,
                         double rhs) { return lhs * rhs; })
      .def("__rmul__", [](const sym::Expression<double, double>& rhs,
                          double lhs) { return rhs * lhs; })
      // division operators
      .def("__truediv__", [](const sym::Expression<double, double>& lhs,
                             double rhs) { return lhs / rhs; })
      .def("__rtruediv__", [](const sym::Expression<double, double>& rhs,
                              double lhs) { return rhs / lhs; })

      .def(
          "__eq__",
          [](const sym::Expression<double, double>& lhs,
             const sym::Expression<double, double>& rhs) { return lhs == rhs; })
      .def("__str__", [](const sym::Expression<double, double>& expr) {
        std::stringstream ss;
        ss << expr;
        return ss.str();
      });  
}
}  // namespace mqt
