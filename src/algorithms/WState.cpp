#include "algorithms/WState.hpp"

namespace qc {
void fGate(QuantumComputation& qc, const Qubit i, const Qubit j, const Qubit k,
           const Qubit n) {
  const auto theta = std::acos(std::sqrt(1.0 / static_cast<double>(k - n + 1)));
  qc.ry(j, -theta);
  qc.z(j, qc::Control{i});
  qc.ry(j, theta);
}

WState::WState(const Qubit nq) : QuantumComputation(nq) {
  name = "wstate_" + std::to_string(nq);

  x(nq - 1);

  for (Qubit m = 1; m < nq; m++) {
    fGate(*this, nq - m, nq - m - 1, nq, m);
  }

  for (Qubit k = nq - 1; k > 0; k--) {
    x(k, qc::Control{k - 1});
  }
}
} // namespace qc
