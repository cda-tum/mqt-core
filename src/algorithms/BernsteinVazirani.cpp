#include "algorithms/BernsteinVazirani.hpp"

namespace qc {
BernsteinVazirani::BernsteinVazirani(const BitString& hiddenString,
                                     const bool dyn)
    : s(hiddenString), dynamic(dyn) {
  std::size_t msb = 0;
  for (std::size_t i = 0; i < s.size(); ++i) {
    if (s.test(i)) {
      msb = i;
    }
  }
  bitwidth = msb + 1;
  createCircuit();
}

BernsteinVazirani::BernsteinVazirani(const std::size_t nq, const bool dyn)
    : bitwidth(nq), dynamic(dyn) {
  auto distribution = std::bernoulli_distribution();
  for (std::size_t i = 0; i < nq; ++i) {
    if (distribution(mt)) {
      s.set(i);
    }
  }
  createCircuit();
}

BernsteinVazirani::BernsteinVazirani(const BitString& hiddenString,
                                     const std::size_t nq, const bool dyn)
    : s(hiddenString), bitwidth(nq), dynamic(dyn) {
  createCircuit();
}

std::ostream& BernsteinVazirani::printStatistics(std::ostream& os) const {
  os << "BernsteinVazirani (" << bitwidth << ") Statistics:\n";
  os << "\tn: " << bitwidth + 1 << "\n";
  os << "\tm: " << getNindividualOps() << "\n";
  os << "\ts: " << expected << "\n";
  os << "\tdynamic: " << dynamic << "\n";
  os << "--------------"
     << "\n";
  return os;
}

void BernsteinVazirani::createCircuit() {
  name = "bv_" + s.to_string();

  expected = s.to_string();
  std::reverse(expected.begin(), expected.end());
  while (expected.length() > bitwidth) {
    expected.pop_back();
  }
  std::reverse(expected.begin(), expected.end());

  addQubitRegister(1, "flag");

  if (dynamic) {
    addQubitRegister(1, "q");
  } else {
    addQubitRegister(bitwidth, "q");
  }

  addClassicalRegister(bitwidth, "c");

  // prepare flag qubit
  x(0);

  if (dynamic) {
    for (std::size_t i = 0; i < bitwidth; ++i) {
      // initial Hadamard
      h(1);

      // apply controlled-Z gate according to secret bitstring
      if (s.test(i)) {
        cz(1, 0);
      }

      // final Hadamard
      h(1);

      // measure result
      measure(1, i);

      // reset qubit if not finished
      if (i < bitwidth - 1) {
        reset(1);
      }
    }
  } else {
    // initial Hadamard transformation
    for (std::size_t i = 1; i <= bitwidth; ++i) {
      h(static_cast<Qubit>(i));
    }

    // apply controlled-Z gates according to secret bitstring
    for (std::size_t i = 1; i <= bitwidth; ++i) {
      if (s.test(i - 1)) {
        cz(static_cast<Qubit>(i), 0);
      }
    }

    // final Hadamard transformation
    for (std::size_t i = 1; i <= bitwidth; ++i) {
      h(static_cast<Qubit>(i));
    }

    // measure results
    for (std::size_t i = 1; i <= bitwidth; i++) {
      measure(static_cast<Qubit>(i), i - 1);
    }
  }
}
} // namespace qc
