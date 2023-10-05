#include "algorithms/Entanglement.hpp"

namespace qc {
Entanglement::Entanglement(const std::size_t nq) : QuantumComputation(nq, nq) {
  name = "entanglement_" + std::to_string(nq);
  const auto top = static_cast<Qubit>(nq - 1);

  h(top);
  for (std::size_t i = 1; i < nq; i++) {
    cx(top, static_cast<Qubit>(top - i));
  }
}
} // namespace qc
