#include "Definitions.hpp"
#include "Permutation.hpp"
#include "operations/Control.hpp"
#include "python/pybind11.hpp"

#include <pybind11/operators.h>
#include <sstream>

namespace mqt {

void registerPermutation(py::module& m) {
  py::class_<qc::Permutation>(m, "Permutation")
      .def(py::init([](const py::dict& p) {
             qc::Permutation perm;
             for (const auto& [key, value] : p) {
               perm[key.cast<qc::Qubit>()] = value.cast<qc::Qubit>();
             }
             return perm;
           }),
           "perm"_a, "Create a permutation from a dictionary.")
      .def("apply",
           py::overload_cast<const qc::Controls&>(&qc::Permutation::apply,
                                                  py::const_),
           "controls"_a)
      .def("apply",
           py::overload_cast<const qc::Targets&>(&qc::Permutation::apply,
                                                 py::const_),
           "targets"_a)
      .def("__getitem__",
           [](const qc::Permutation& p, const qc::Qubit q) { return p.at(q); })
      .def("__setitem__", [](qc::Permutation& p, const qc::Qubit q,
                             const qc::Qubit r) { p.at(q) = r; })
      .def("__delitem__",
           [](qc::Permutation& p, const qc::Qubit q) { p.erase(q); })
      .def("__len__", &qc::Permutation::size)
      .def(
          "__iter__",
          [](const qc::Permutation& p) {
            return py::make_iterator(p.begin(), p.end());
          },
          py::keep_alive<0, 1>())
      .def(py::self == py::self)
      .def(py::self != py::self)
      .def(hash(py::self))
      .def("__str__",
           [](const qc::Permutation& p) {
             std::stringstream ss;
             ss << "{";
             for (const auto& [k, v] : p) {
               ss << k << ": " << v << ", ";
             }
             ss << "}";
             return ss.str();
           })
      .def("__repr__", [](const qc::Permutation& p) {
        std::stringstream ss;
        ss << "Permutation({";
        for (const auto& [k, v] : p) {
          ss << k << ": " << v << ", ";
        }
        ss << "})";
        return ss.str();
      });
  py::implicitly_convertible<py::dict, qc::Permutation>();
}

} // namespace mqt
