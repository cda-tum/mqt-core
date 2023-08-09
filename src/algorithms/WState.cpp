#include "algorithms/WState.hpp"

namespace qc {
void fGate(QuantumComputation& qc, const Qubit i, const Qubit j, const Qubit k,
           const Qubit n) {
  const auto theta = std::acos(std::sqrt(1.0 / static_cast<double>(k - n + 1)));
  qc.ry(j, -theta);
  qc.z(j, qc::Control{i});
  qc.ry(j, theta);
}

WState::WState(const std::size_t nq) : QuantumComputation(nq) {
  if (nq == 0) {
    return;
  }
  auto nQubits = static_cast<Qubit>(nq);

  name = "wstate_" + std::to_string(nq);
  const auto top = nQubits - 1;

  x(top);

  for (Qubit m = 1; m < nq; m++) {
    fGate(*this, nQubits - m, nQubits - m - 1, nQubits, m);
  }

  for (Qubit k = nQubits - 1; k > 0; k--) {
    x(k, qc::Control{k - 1});
  }
}
} // namespace qc
