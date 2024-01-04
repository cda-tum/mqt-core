#include "dd/FunctionalityConstruction.hpp"
#include "dd/Package.hpp"

namespace dd {

// get next garbage qubit after n
inline Qubit getNextGarbage(Qubit n, const std::vector<bool>& garbage) {
  while (n < garbage.size() && !garbage.at(n)) {
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
                             std::unique_ptr<dd::Package<Config>>& dd) {

  auto d1 = c1.getNqubitsWithoutAncillae();
  auto d2 = c2.getNqubitsWithoutAncillae();
  auto m1 = c1.getNmeasuredQubits();
  auto m2 = c2.getNmeasuredQubits();
  if (m1 != m2 || d1 != d2) {
    return false;
  }

  // add swaps in order to put the measured (= not garbage) qubits in the end
  auto garbage = c1.getGarbage();
  auto n = static_cast<Qubit>(garbage.size());
  auto nextGarbage = getNextGarbage(0, garbage);
  // find the first garbage qubit at the end
  for (Qubit i = n - 1; i >= static_cast<Qubit>(m1); i--) {
    if (!garbage.at(i)) {
      // swap it to the beginning
      c1.swap(i, nextGarbage);
      c2.swap(i, nextGarbage);
      ++nextGarbage;
      nextGarbage = getNextGarbage(nextGarbage, garbage);
    }
  }

  // partialEquivalenceCheck with dd

  auto u1 = buildFunctionality(&c1, dd, false, false);
  auto u2 = buildFunctionality(&c2, dd, false, false);
  if (d1 == n) {
    // no ancilla qubits
    return dd->zeroAncillaPartialEquivalenceCheck(u1, u2,
                                                  static_cast<Qubit>(m1));
  }
  return dd->partialEquivalenceCheck(u1, u2, static_cast<Qubit>(d1),
                                     static_cast<Qubit>(m1));
}
} // namespace dd
