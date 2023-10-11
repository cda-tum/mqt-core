#include "algorithms/QPE.hpp"

namespace qc {
QPE::QPE(const std::size_t nq, const bool exact, const bool iter)
    : precision(nq), iterative(iter) {
  if (exact) {
    // if an exact solution is wanted, generate a random n-bit number and
    // convert it to an appropriate phase
    const std::uint64_t max = 1ULL << nq;
    auto distribution =
        std::uniform_int_distribution<std::uint64_t>(0, max - 1);
    std::uint64_t theta = 0;
    while (theta == 0) {
      theta = distribution(mt);
    }
    lambda = 0.;
    for (std::size_t i = 0; i < nq; ++i) {
      if ((theta & (1ULL << (nq - i - 1))) != 0) {
        lambda += 1. / static_cast<double>(1ULL << i);
      }
    }
  } else {
    // if an inexact solution is wanted, generate a random n+1-bit number (that
    // has its last bit set) and convert it to an appropriate phase
    const std::uint64_t max = 1ULL << (nq + 1);
    auto distribution =
        std::uniform_int_distribution<std::uint64_t>(0, max - 1);
    std::uint64_t theta = 0;
    while ((theta & 1) == 0) {
      theta = distribution(mt);
    }
    lambda = 0.;
    for (std::size_t i = 0; i <= nq; ++i) {
      if ((theta & (1ULL << (nq - i))) != 0) {
        lambda += 1. / static_cast<double>(1ULL << i);
      }
    }
  }
  createCircuit();
}

QPE::QPE(const fp l, const std::size_t prec, const bool iter)
    : lambda(l), precision(prec), iterative(iter) {
  createCircuit();
}

std::ostream& QPE::printStatistics(std::ostream& os) const {
  os << "QPE Statistics:\n";
  os << "\tn: " << nqubits + 1 << "\n";
  os << "\tm: " << getNindividualOps() << "\n";
  os << "\tlambda: " << lambda << "π"
     << "\n";
  os << "\tprecision: " << precision << "\n";
  os << "\titerative: " << iterative << "\n";
  os << "--------------"
     << "\n";
  return os;
}

void QPE::createCircuit() {
  addQubitRegister(1, "psi");

  if (iterative) {
    addQubitRegister(1, "q");
  } else {
    addQubitRegister(precision, "q");
  }

  addClassicalRegister(precision, "c");

  // prepare eigenvalue
  x(0);

  if (iterative) {
    for (std::size_t i = 0; i < precision; i++) {
      // Hadamard
      h(1);

      // normalize angle
      const auto angle = std::remainder(
          static_cast<double>(1ULL << (precision - 1 - i)) * lambda, 2.0);

      // controlled phase rotation
      cp(angle * PI, 1, 0);

      // hybrid quantum-classical inverse QFT
      for (std::size_t j = 0; j < i; j++) {
        auto iQFTLambda = -PI / static_cast<double>(1ULL << (i - j));
        classicControlled(P, 1, {j, 1U}, 1U, {iQFTLambda});
      }
      h(1);

      // measure result
      measure(1, i);

      // reset qubit if not finished
      if (i < precision - 1) {
        reset(1);
      }
    }
  } else {
    // Hadamard Layer
    for (std::size_t i = 1; i <= precision; i++) {
      h(static_cast<Qubit>(i));
    }

    for (std::size_t i = 0; i < precision; i++) {
      // normalize angle
      const auto angle = std::remainder(
          static_cast<double>(1ULL << (precision - 1 - i)) * lambda, 2.0);

      // controlled phase rotation
      cp(angle * PI, static_cast<Qubit>(1 + i), 0);

      // inverse QFT
      for (std::size_t j = 1; j < 1 + i; j++) {
        const auto iQFTLambda = -PI / static_cast<double>(2ULL << (i - j));
        if (j == i) {
          csdg(static_cast<Qubit>(i), static_cast<Qubit>(1 + i));
        } else if (j == (i - 1)) {
          ctdg(static_cast<Qubit>(i - 1), static_cast<Qubit>(1 + i));
        } else {
          cp(iQFTLambda, static_cast<Qubit>(j), static_cast<Qubit>(1 + i));
        }
      }
      h(static_cast<Qubit>(1 + i));
    }

    // measure results
    for (std::size_t i = 0; i < nqubits - 1; i++) {
      measure(static_cast<Qubit>(i + 1), i);
    }
  }
}
} // namespace qc
