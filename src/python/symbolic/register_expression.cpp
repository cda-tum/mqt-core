#include "operations/Expression.hpp"
#include "pybind11/operators.h"
#include "python/pybind11.hpp"

#include <sstream>

namespace mqt {

void registerExpression(py::module& m) {
  py::class_<sym::Expression<double, double>>(
      m, "Expression",
      "Class representing a symbolic sum of terms. The expression is of the "
      "form `constant + term_1 + term_2 + ... + term_n`.")
      .def(py::init([](const std::vector<sym::Term<double>>& terms,
                       double constant) {
             return sym::Expression<double, double>(terms, constant);
           }),
           "terms"_a, "constant"_a = 0.0,
           "Create an expression with a given list of terms and a constant (0 "
           "by default).")
      .def(py::init([](const sym::Term<double>& term, double constant) {
             return sym::Expression<double, double>(
                 std::vector<sym::Term<double>>{term}, constant);
           }),
           "term"_a, "constant"_a = 0.0,
           "Create an expression with a given term and a constant (0 by "
           "default).")
      .def(py::init<double>(), "constant"_a = 0.0,
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
           [](const sym::Expression<double, double>& expr,
              const std::size_t idx) {
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
      .def(py::self + py::self)
      .def(py::self + double())
      .def("__add__", [](const sym::Expression<double, double>& lhs,
                         const sym::Term<double>& rhs) { return lhs + rhs; })
      .def("__radd__", [](const sym::Expression<double, double>& rhs,
                          const sym::Term<double>& lhs) { return lhs + rhs; })
      .def("__radd__", [](const sym::Expression<double, double>& rhs,
                          const double lhs) { return rhs + lhs; })
      // subtraction operators
      .def(py::self - py::self)
      .def(py::self - double())
      .def(double() - py::self)
      .def("__sub__", [](const sym::Expression<double, double>& lhs,
                         const sym::Term<double>& rhs) { return lhs - rhs; })
      .def("__rsub__", [](const sym::Expression<double, double>& rhs,
                          const sym::Term<double>& lhs) { return lhs - rhs; })
      // multiplication operators
      .def(py::self * double())
      .def(double() * py::self)
      // division operators
      .def(py::self / double())
      .def("__rtruediv__", [](const sym::Expression<double, double>& rhs,
                              double lhs) { return rhs / lhs; })
      // comparison operators
      .def(py::self == py::self)
      .def("__str__",
           [](const sym::Expression<double, double>& expr) {
             std::stringstream ss;
             ss << expr;
             return ss.str();
           })
      .def("__repr__", [](const sym::Expression<double, double>& expr) {
        std::stringstream ss;
        ss << expr;
        return ss.str();
      });
}
} // namespace mqt
