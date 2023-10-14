#include "algorithms/RandomCliffordCircuit.hpp"

namespace qc {

RandomCliffordCircuit::RandomCliffordCircuit(const std::size_t nq,
                                             const std::size_t d,
                                             const std::size_t s)
    : depth(d), seed(s) {
  addQubitRegister(nq);
  addClassicalRegister(nq);

  std::mt19937_64 generator;
  if (seed == 0) {
    // this is probably overkill but better safe than sorry
    std::array<std::mt19937_64::result_type, std::mt19937_64::state_size>
        randomData{};
    std::random_device rd;
    std::generate(std::begin(randomData), std::end(randomData), std::ref(rd));
    std::seed_seq seeds(std::begin(randomData), std::end(randomData));
    generator.seed(seeds);
  } else {
    generator.seed(seed);
  }
  std::uniform_int_distribution<std::uint16_t> distribution(0, 11520);
  cliffordGenerator = [&]() { return distribution(generator); };

  for (std::size_t l = 0; l < depth; ++l) {
    if (nqubits == 1) {
      append1QClifford(cliffordGenerator(), 0);
    } else if (nqubits == 2) {
      append2QClifford(cliffordGenerator(), 0, 1);
    } else {
      if (l % 2 != 0) {
        for (std::size_t i = 1; i < nqubits - 1; i += 2) {
          append2QClifford(cliffordGenerator(), static_cast<Qubit>(i),
                           static_cast<Qubit>(i + 1));
        }
      } else {
        for (std::size_t i = 0; i < nqubits - 1; i += 2) {
          append2QClifford(cliffordGenerator(), static_cast<Qubit>(i),
                           static_cast<Qubit>(i + 1));
        }
      }
    }
  }
}

std::ostream& RandomCliffordCircuit::printStatistics(std::ostream& os) const {
  os << "Random Clifford circuit statistics:\n";
  os << "\tn: " << nqubits << "\n";
  os << "\tm: " << getNindividualOps() << "\n";
  os << "\tdepth: " << depth << "\n";
  os << "\tseed: " << seed << "\n";
  os << "--------------"
     << "\n";
  return os;
}

void RandomCliffordCircuit::append1QClifford(const std::uint16_t idx,
                                             const Qubit target) {
  const auto id = static_cast<std::uint8_t>(idx % 24);
  auto qc = QuantumComputation(nqubits);
  // Hadamard
  if ((id / 12 % 2) != 0) {
    qc.h(target);
  }

  // Rotation
  if (id / 4 % 3 == 1) {
    qc.h(target);
    qc.s(target);
  } else if (id / 4 % 3 == 2) {
    qc.sdg(target);
    qc.h(target);
  }

  // Pauli
  if (id % 4 == 1) {
    qc.z(target);
  } else if (id % 4 == 2) {
    qc.x(target);
  } else if (id % 4 == 3) {
    qc.y(target);
  }
  emplace_back<CompoundOperation>(qc.asCompoundOperation());
}

void RandomCliffordCircuit::append2QClifford(const std::uint16_t idx,
                                             const Qubit control,
                                             const Qubit target) {
  auto id = static_cast<std::uint16_t>(idx % 11520);
  const auto pauliIdx = static_cast<std::uint8_t>(id % 16);
  id /= 16;

  auto qc = QuantumComputation(nqubits);
  if (id < 36) {
    // single-qubit Cliffords
    if ((id / 9 % 2) != 0) {
      qc.h(control);
    }
    if ((id / 18 % 2) != 0) {
      qc.h(target);
    }

    if (id % 3 == 1) {
      qc.h(control);
      qc.s(control);
    } else if (id % 3 == 2) {
      qc.sdg(control);
      qc.h(control);
    }
    if (id / 3 % 3 == 1) {
      qc.h(target);
      qc.s(target);
    } else if (id / 3 % 3 == 2) {
      qc.sdg(target);
      qc.h(target);
    }
  } else if (id < 360) {
    // Cliffords with a single CNOT
    id -= 36;

    if ((id / 81 % 2) != 0) {
      qc.h(control);
    }
    if ((id / 162 % 2) != 0) {
      qc.h(target);
    }

    if (id % 3 == 1) {
      qc.h(control);
      qc.s(control);
    } else if (id % 3 == 2) {
      qc.sdg(control);
      qc.h(control);
    }
    if (id / 3 % 3 == 1) {
      qc.h(target);
      qc.s(target);
    } else if (id / 3 % 3 == 2) {
      qc.sdg(target);
      qc.h(target);
    }

    qc.cx(control, target);

    if (id / 9 % 3 == 1) {
      qc.h(control);
      qc.s(control);
    } else if (id / 9 % 3 == 2) {
      qc.sdg(control);
      qc.h(control);
    }
    if (id / 27 % 3 == 1) {
      qc.h(target);
      qc.s(target);
    } else if (id / 27 % 3 == 2) {
      qc.sdg(target);
      qc.h(target);
    }
  } else if (id < 684) {
    // Cliffords with two CNOTs
    id -= 360;

    if ((id / 81 % 2) != 0) {
      qc.h(control);
    }
    if ((id / 162 % 2) != 0) {
      qc.h(target);
    }

    if (id % 3 == 1) {
      qc.h(control);
      qc.s(control);
    } else if (id % 3 == 2) {
      qc.sdg(control);
      qc.h(control);
    }
    if (id / 3 % 3 == 1) {
      qc.h(target);
      qc.s(target);
    } else if (id / 3 % 3 == 2) {
      qc.sdg(target);
      qc.h(target);
    }

    qc.cx(control, target);
    qc.cx(target, control);

    if (id / 9 % 3 == 1) {
      qc.h(control);
      qc.s(control);
    } else if (id / 9 % 3 == 2) {
      qc.sdg(control);
      qc.h(control);
    }
  } else {
    // Cliffords with a SWAP
    id -= 684;

    if ((id / 9 % 2) != 0) {
      qc.h(control);
    }
    if ((id / 18 % 2) != 0) {
      qc.h(target);
    }

    if (id % 3 == 1) {
      qc.h(control);
      qc.s(control);
    } else if (id % 3 == 2) {
      qc.sdg(control);
      qc.h(control);
    }

    qc.cx(control, target);
    qc.cx(target, control);
    qc.cx(control, target);
  }

  // random Pauli on control qubit
  if (pauliIdx % 4 == 1) {
    qc.z(control);
  } else if (pauliIdx % 4 == 2) {
    qc.x(control);
  } else if (pauliIdx % 4 == 3) {
    qc.y(control);
  }

  // random Pauli on target qubit
  if (pauliIdx / 4 == 1) {
    qc.z(target);
  } else if (pauliIdx / 4 == 2) {
    qc.x(target);
  } else if (pauliIdx / 4 == 3) {
    qc.y(target);
  }

  emplace_back<CompoundOperation>(qc.asCompoundOperation());
}
} // namespace qc
