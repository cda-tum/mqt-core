#include "dd/FunctionalityConstruction.hpp"
#include "dd/Package.hpp"

namespace dd {
/**
    Checks for partial equivalence between the two circuits c1 and c2
    that have no ancilla qubits.
    Assumption: the input and output permutations are the same.

    @param circuit1 First circuit
    @param circuit2 Second circuit
    @return true if the two circuits c1 and c2 are partially equivalent.
    **/
template <class Config>
bool zeroAncillaePartialEquivalenceCheck(
    qc::QuantumComputation c1, qc::QuantumComputation c2,
    const std::unique_ptr<dd::Package<Config>>& dd) {
  if (c1.getNqubits() != c2.getNqubits() ||
      c1.getGarbage() != c2.getGarbage()) {
    throw std::runtime_error(
        "The circuits need to have the same number of qubits and the same "
        "permutation of input and output qubits.");
  }
  c2.invert();
  for (auto& gate : c1) {
    c2.emplace_back(gate);
  }

  const auto u = buildFunctionality(&c2, *dd, false, false);

  return dd->isCloseToIdentity(u, 1.0E-10, c1.getGarbage(), false);
}
// get next garbage qubit after n
inline Qubit getNextGarbage(Qubit n, const std::vector<bool>& garbage) {
  while (n < static_cast<Qubit>(garbage.size()) && !garbage.at(n)) {
    n++;
  }
  return n;
}
/**
    Checks for partial equivalence between the two circuits c1 and c2.
    Assumption: the data qubits are all at the beginning of the input qubits and
    the input and output permutations are the same.

    @param circuit1 First circuit
    @param circuit2 Second circuit
    @return true if the two circuits c1 and c2 are partially equivalent.
    **/
template <class Config>
bool partialEquivalenceCheck(qc::QuantumComputation c1,
                             qc::QuantumComputation c2,
                             const std::unique_ptr<dd::Package<Config>>& dd) {

  const auto d1 = c1.getNqubitsWithoutAncillae();
  const auto d2 = c2.getNqubitsWithoutAncillae();
  const auto m1 = c1.getNmeasuredQubits();
  const auto m2 = c2.getNmeasuredQubits();
  if (m1 != m2 || d1 != d2) {
    return false;
  }
  const auto n1 = static_cast<Qubit>(c1.getNqubits());
  const auto n2 = static_cast<Qubit>(c2.getNqubits());
  if (d1 == n1 && d2 == n2) {
    // no ancilla qubits
    return zeroAncillaePartialEquivalenceCheck(c1, c2, dd);
  }
  // add swaps in order to put the measured (= not garbage) qubits in the end
  const auto garbage1 = c1.getGarbage();

  auto nextGarbage = getNextGarbage(0, garbage1);
  // find the first garbage qubit at the end
  for (std::int64_t i = std::min(n1, n2) - 1;
       i >= static_cast<std::int64_t>(m1); i--) {
    if (!garbage1.at(static_cast<Qubit>(i))) {
      // swap it to the beginning
      c1.swap(static_cast<Qubit>(i), nextGarbage);
      c2.swap(static_cast<Qubit>(i), nextGarbage);
      ++nextGarbage;
      nextGarbage = getNextGarbage(nextGarbage, garbage1);
    }
  }

  // partialEquivalenceCheck with dd

  const auto u1 = buildFunctionality(&c1, *dd, false, false);
  const auto u2 = buildFunctionality(&c2, *dd, false, false);

  return dd->partialEquivalenceCheck(u1, u2, static_cast<Qubit>(d1),
                                     static_cast<Qubit>(m1));
}

std::pair<qc::QuantumComputation, qc::QuantumComputation>
generateRandomBenchmark(size_t n, Qubit d, Qubit m);
} // namespace dd
