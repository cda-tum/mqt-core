#include "dd/FunctionalityConstruction.hpp"
#include "dd/Package.hpp"

namespace dd {

template <class Config>
mEdge partialEquivalenceCheckDDSubroutine(mEdge u, const Qubit m, const Qubit k,
                                          const Qubit extra,
                                          Package<Config>& dd) {
  // add extra ancillary qubits
  if (extra > 0) {
    if (u.p->v + 1U + extra > dd.qubits()) {
      dd.resize(u.p->v + 1U + extra);
    }
    u = dd.kronecker(dd.makeIdent(extra), u);
  }
  if (u.isTerminal()) {
    return u;
  }
  const Qubit n = u.p->v + 1;
  const Qubit d = n - k;
  u = dd.setColumnsToZero(u, d);
  const auto u2 = dd.shiftAllRows(u, m, d);
  return dd.multiply(dd.conjugateTranspose(u), u2);
}

/**
 Checks for partial equivalence between the two circuits u1 and u2,
  where the first d qubits of the circuits are the data qubits and
  the first m qubits are the measured qubits.
 @param u1 DD representation of first circuit
 @param u2 DD representation of second circuit
 @param d Number of data qubits
 @param m Number of measured qubits
 @return true if the two circuits u1 and u2 are partially equivalent.
 **/
template <class Config>
bool partialEquivalenceCheckDD(mEdge u1, mEdge u2, const Qubit d, const Qubit m,
                               Package<Config>& dd) {
  if (m == 0) {
    return true;
  }
  if (u1.isTerminal() && u2.isTerminal()) {
    return u1 == u2;
  }
  if (u1.isZeroTerminal() || u2.isZeroTerminal()) {
    return false;
  }
  // add qubits such that u1 and u2 have the same dimension
  if (u1.isTerminal()) {
    auto w = u1.w;
    u1 = dd.makeIdent(u2.p->v + 1);
    u1.w = w;
  } else if (u2.isTerminal()) {
    auto w = u2.w;
    u2 = dd.makeIdent(u1.p->v + 1);
    u2.w = w;
  } else if (u1.p->v < u2.p->v) {
    u1 = dd.kronecker(dd.makeIdent(u2.p->v - u1.p->v), u1);
  } else if (u1.p->v > u2.p->v) {
    u2 = dd.kronecker(dd.makeIdent(u1.p->v - u2.p->v), u2);
  }

  const Qubit n = u1.p->v + 1;
  Qubit k = n - d;
  Qubit extra{0};
  if (m > k) {
    extra = m - k;
  }
  k = k + extra;

  const auto u1Prime = partialEquivalenceCheckDDSubroutine(u1, m, k, extra, dd);
  const auto u2Prime = partialEquivalenceCheckDDSubroutine(u2, m, k, extra, dd);

  return u1Prime == u2Prime;
}

/**
    Checks for partial equivalence between the two circuits c1 and c2
    that have no ancilla qubits.
    Assumption: the input and output permutations are the same.

    @param circuit1 First circuit
    @param circuit2 Second circuit
    @return true if the two circuits c1 and c2 are partially equivalent.
    **/
template <class Config>
bool zeroAncillaePartialEquivalenceCheck(qc::QuantumComputation c1,
                                         qc::QuantumComputation c2,
                                         Package<Config>& dd) {
  if (c1.getNqubits() != c2.getNqubits() ||
      c1.getGarbage() != c2.getGarbage()) {
    throw std::invalid_argument(
        "The circuits need to have the same number of qubits and the same "
        "permutation of input and output qubits.");
  }
  c2.invert();
  for (auto& gate : c1) {
    c2.emplace_back(gate);
  }

  const auto u = buildFunctionality(&c2, dd, false, false);

  return dd.isCloseToIdentity(u, 1.0E-10, c1.getGarbage(), false);
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
                             qc::QuantumComputation c2, Package<Config>& dd) {

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

  const auto u1 = buildFunctionality(&c1, dd, false, false);
  const auto u2 = buildFunctionality(&c2, dd, false, false);

  return partialEquivalenceCheckDD(u1, u2, static_cast<Qubit>(d1),
                                   static_cast<Qubit>(m1), dd);
}

std::pair<qc::QuantumComputation, qc::QuantumComputation>
generateRandomBenchmark(size_t n, qc::Qubit d, qc::Qubit m);
} // namespace dd
