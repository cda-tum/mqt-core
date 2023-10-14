#include "algorithms/QFT.hpp"

namespace qc {
QFT::QFT(const std::size_t nq, const bool includeMeas, const bool dyn)
    : precision(nq), includeMeasurements(includeMeas), dynamic(dyn) {
  name = "qft_" + std::to_string(nq);
  if (precision == 0) {
    return;
  }

  if (dynamic) {
    addQubitRegister(1);
  } else {
    addQubitRegister(precision);
  }
  addClassicalRegister(precision);
  createCircuit();
}

std::ostream& QFT::printStatistics(std::ostream& os) const {
  os << "QFT (" << precision << ") Statistics:\n";
  os << "\tn: " << nqubits << "\n";
  os << "\tm: " << getNindividualOps() << "\n";
  os << "\tdynamic: " << dynamic << "\n";
  os << "--------------"
     << "\n";
  return os;
}
void QFT::createCircuit() {
  if (dynamic) {
    for (std::size_t i = 0; i < precision; i++) {
      // apply classically controlled phase rotations
      for (std::size_t j = 1; j <= i; ++j) {
        const auto d = static_cast<Qubit>(precision - j);
        if (j == i) {
          classicControlled(S, 0, {d, 1U}, 1U);
        } else if (j == i - 1) {
          classicControlled(T, 0, {d, 1U}, 1U);
        } else {
          const auto powerOfTwo = std::pow(2., i - j + 1);
          const auto lambda = PI / powerOfTwo;
          classicControlled(P, 0, {d, 1U}, 1U, std::vector{lambda});
        }
      }

      // apply Hadamard
      h(0);

      // measure result
      measure(0, precision - 1 - i);

      // reset qubit if not finished
      if (i < precision - 1) {
        reset(0);
      }
    }
  } else {
    // apply quantum Fourier transform
    for (std::size_t i = 0; i < precision; ++i) {
      const auto q = static_cast<Qubit>(i);

      // apply controlled rotations
      for (std::size_t j = i; j > 0; --j) {
        const auto d = static_cast<Qubit>(q - j);
        if (j == 1) {
          cs(q, d);
        } else if (j == 2) {
          ct(q, d);
        } else {
          const auto powerOfTwo = std::pow(2., j);
          const auto lambda = PI / powerOfTwo;
          cp(lambda, q, d);
        }
      }

      // apply Hadamard
      h(q);
    }

    if (includeMeasurements) {
      // measure qubits in reverse order
      for (std::size_t i = 0; i < precision; ++i) {
        measure(static_cast<Qubit>(i), precision - 1 - i);
      }
    } else {
      for (Qubit i = 0; i < static_cast<Qubit>(precision / 2); ++i) {
        swap(i, static_cast<Qubit>(precision - 1 - i));
      }
      for (std::size_t i = 0; i < precision; ++i) {
        outputPermutation[static_cast<Qubit>(i)] =
            static_cast<Qubit>(precision - 1 - i);
      }
    }
  }
}
} // namespace qc
