#include "dd/StateGenerator.hpp"
#include "dd/States.hpp"
#include "python/pybind11.hpp"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace mqt {

py::array_t<std::complex<dd::fp>> getNumPyVectorFromDD(const dd::vEdge& e) {
  assert(!e.isTerminal() && "Root must not be a terminal");
  auto* data = new std::complex<dd::fp>[2ULL << e.p->v];

  dd::getVector(e, 1, 0ULL, data);

  py::capsule owner(data, [](void* p) noexcept {
    delete[] static_cast<std::complex<dd::fp>*>(p);
  });

  return py::array_t<std::complex<dd::fp>>(
      {2ULL << e.p->v}, {sizeof(std::complex<dd::fp>)}, data, owner);
}

void registerDDModule(py::module_& m) {
  py::class_<dd::StateGenerator>(m, "StateGenerator",
                                 "A class for working with DDs and states")
      .def(py::init<std::size_t>(), "seed"_a = 0U)
      .def(
          "get_random_state_from_structured_dd",
          [](dd::StateGenerator& stateGenerator, const std::size_t levels,
             const std::vector<std::size_t>& nodesPerLevel) {
            const auto state =
                stateGenerator.generateRandomVectorDD(levels, nodesPerLevel);

            return getNumPyVectorFromDD(state);
          },
          "levels"_a, "nodes_per_level"_a);
}

} // namespace mqt
