/*
 * This file is part of MQT QFR library which is released under the MIT license.
 * See file README.md or go to https://www.cda.cit.tum.de/research/quantum/ for more information.
 */

#include "algorithms/RandomCliffordCircuit.hpp"

namespace qc {

    RandomCliffordCircuit::RandomCliffordCircuit(const std::size_t nq, const std::size_t d, const std::size_t s):
        depth(d), seed(s) {
        addQubitRegister(nq);
        addClassicalRegister(nq);

        for (std::size_t i = 0; i < nqubits; ++i) {
            initialLayout.insert({i, i});
            outputPermutation.insert({i, i});
        }

        std::mt19937_64 generator;
        if (seed == 0) {
            // this is probably overkill but better safe than sorry
            std::array<std::mt19937_64::result_type, std::mt19937_64::state_size> randomData{};
            std::random_device                                                    rd;
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
                        append2QClifford(cliffordGenerator(), static_cast<Qubit>(i), static_cast<Qubit>(i + 1));
                    }
                } else {
                    for (std::size_t i = 0; i < nqubits - 1; i += 2) {
                        append2QClifford(cliffordGenerator(), static_cast<Qubit>(i), static_cast<Qubit>(i + 1));
                    }
                }
            }
        }
    }

    std::ostream& RandomCliffordCircuit::printStatistics(std::ostream& os) const {
        os << "Random Clifford circuit statistics:\n";
        os << "\tn: " << nqubits << std::endl;
        os << "\tm: " << getNindividualOps() << std::endl;
        os << "\tdepth: " << depth << std::endl;
        os << "\tseed: " << seed << std::endl;
        os << "--------------" << std::endl;
        return os;
    }

    void RandomCliffordCircuit::append1QClifford(const std::uint16_t idx, const Qubit target) {
        const auto id = static_cast<std::uint8_t>(idx % 24);
        emplace_back<CompoundOperation>(nqubits);
        auto* comp = dynamic_cast<CompoundOperation*>(ops.back().get());
        // Hadamard
        if ((id / 12 % 2) != 0) {
            comp->emplace_back<StandardOperation>(nqubits, target, H);
        }

        // Rotation
        if (id / 4 % 3 == 1) {
            comp->emplace_back<StandardOperation>(nqubits, target, H);
            comp->emplace_back<StandardOperation>(nqubits, target, S);
        } else if (id / 4 % 3 == 2) {
            comp->emplace_back<StandardOperation>(nqubits, target, Sdag);
            comp->emplace_back<StandardOperation>(nqubits, target, H);
        }

        // Pauli
        if (id % 4 == 1) {
            comp->emplace_back<StandardOperation>(nqubits, target, Z);
        } else if (id % 4 == 2) {
            comp->emplace_back<StandardOperation>(nqubits, target, X);
        } else if (id % 4 == 3) {
            comp->emplace_back<StandardOperation>(nqubits, target, Y);
        }
    }

    void RandomCliffordCircuit::append2QClifford(const std::uint16_t idx, const Qubit control, const Qubit target) {
        auto       id       = static_cast<std::uint16_t>(idx % 11520);
        const auto pauliIdx = static_cast<std::uint8_t>(id % 16);
        id /= 16;

        emplace_back<CompoundOperation>(nqubits);
        auto* comp = dynamic_cast<CompoundOperation*>(ops.back().get());
        if (id < 36) {
            // single-qubit Cliffords
            if ((id / 9 % 2) != 0) {
                comp->emplace_back<StandardOperation>(nqubits, control, H);
            }
            if ((id / 18 % 2) != 0) {
                comp->emplace_back<StandardOperation>(nqubits, target, H);
            }

            if (id % 3 == 1) {
                comp->emplace_back<StandardOperation>(nqubits, control, H);
                comp->emplace_back<StandardOperation>(nqubits, control, S);
            } else if (id % 3 == 2) {
                comp->emplace_back<StandardOperation>(nqubits, control, Sdag);
                comp->emplace_back<StandardOperation>(nqubits, control, H);
            }
            if (id / 3 % 3 == 1) {
                comp->emplace_back<StandardOperation>(nqubits, target, H);
                comp->emplace_back<StandardOperation>(nqubits, target, S);
            } else if (id / 3 % 3 == 2) {
                comp->emplace_back<StandardOperation>(nqubits, target, Sdag);
                comp->emplace_back<StandardOperation>(nqubits, target, H);
            }
        } else if (id < 360) {
            // Cliffords with a single CNOT
            id -= 36;

            if ((id / 81 % 2) != 0) {
                comp->emplace_back<StandardOperation>(nqubits, control, H);
            }
            if ((id / 162 % 2) != 0) {
                comp->emplace_back<StandardOperation>(nqubits, target, H);
            }

            if (id % 3 == 1) {
                comp->emplace_back<StandardOperation>(nqubits, control, H);
                comp->emplace_back<StandardOperation>(nqubits, control, S);
            } else if (id % 3 == 2) {
                comp->emplace_back<StandardOperation>(nqubits, control, Sdag);
                comp->emplace_back<StandardOperation>(nqubits, control, H);
            }
            if (id / 3 % 3 == 1) {
                comp->emplace_back<StandardOperation>(nqubits, target, H);
                comp->emplace_back<StandardOperation>(nqubits, target, S);
            } else if (id / 3 % 3 == 2) {
                comp->emplace_back<StandardOperation>(nqubits, target, Sdag);
                comp->emplace_back<StandardOperation>(nqubits, target, H);
            }

            comp->emplace_back<StandardOperation>(nqubits, qc::Control{control}, target, X);

            if (id / 9 % 3 == 1) {
                comp->emplace_back<StandardOperation>(nqubits, control, H);
                comp->emplace_back<StandardOperation>(nqubits, control, S);
            } else if (id / 9 % 3 == 2) {
                comp->emplace_back<StandardOperation>(nqubits, control, Sdag);
                comp->emplace_back<StandardOperation>(nqubits, control, H);
            }
            if (id / 27 % 3 == 1) {
                comp->emplace_back<StandardOperation>(nqubits, target, H);
                comp->emplace_back<StandardOperation>(nqubits, target, S);
            } else if (id / 27 % 3 == 2) {
                comp->emplace_back<StandardOperation>(nqubits, target, Sdag);
                comp->emplace_back<StandardOperation>(nqubits, target, H);
            }
        } else if (id < 684) {
            // Cliffords with two CNOTs
            id -= 360;

            if ((id / 81 % 2) != 0) {
                comp->emplace_back<StandardOperation>(nqubits, control, H);
            }
            if ((id / 162 % 2) != 0) {
                comp->emplace_back<StandardOperation>(nqubits, target, H);
            }

            if (id % 3 == 1) {
                comp->emplace_back<StandardOperation>(nqubits, control, H);
                comp->emplace_back<StandardOperation>(nqubits, control, S);
            } else if (id % 3 == 2) {
                comp->emplace_back<StandardOperation>(nqubits, control, Sdag);
                comp->emplace_back<StandardOperation>(nqubits, control, H);
            }
            if (id / 3 % 3 == 1) {
                comp->emplace_back<StandardOperation>(nqubits, target, H);
                comp->emplace_back<StandardOperation>(nqubits, target, S);
            } else if (id / 3 % 3 == 2) {
                comp->emplace_back<StandardOperation>(nqubits, target, Sdag);
                comp->emplace_back<StandardOperation>(nqubits, target, H);
            }

            comp->emplace_back<StandardOperation>(nqubits, qc::Control{control}, target, X);
            comp->emplace_back<StandardOperation>(nqubits, qc::Control{target}, control, X);

            if (id / 9 % 3 == 1) {
                comp->emplace_back<StandardOperation>(nqubits, control, H);
                comp->emplace_back<StandardOperation>(nqubits, control, S);
            } else if (id / 9 % 3 == 2) {
                comp->emplace_back<StandardOperation>(nqubits, control, Sdag);
                comp->emplace_back<StandardOperation>(nqubits, control, H);
            }
            if (id / 27 % 3 == 1) {
                comp->emplace_back<StandardOperation>(nqubits, target, H);
                comp->emplace_back<StandardOperation>(nqubits, target, S);
            } else if (id / 27 % 3 == 2) {
                comp->emplace_back<StandardOperation>(nqubits, target, Sdag);
                comp->emplace_back<StandardOperation>(nqubits, target, H);
            }
        } else {
            // Cliffords with a SWAP
            id -= 684;

            if ((id / 9 % 2) != 0) {
                comp->emplace_back<StandardOperation>(nqubits, control, H);
            }
            if ((id / 18 % 2) != 0) {
                comp->emplace_back<StandardOperation>(nqubits, target, H);
            }

            if (id % 3 == 1) {
                comp->emplace_back<StandardOperation>(nqubits, control, H);
                comp->emplace_back<StandardOperation>(nqubits, control, S);
            } else if (id % 3 == 2) {
                comp->emplace_back<StandardOperation>(nqubits, control, Sdag);
                comp->emplace_back<StandardOperation>(nqubits, control, H);
            }
            if (id / 3 % 3 == 1) {
                comp->emplace_back<StandardOperation>(nqubits, target, H);
                comp->emplace_back<StandardOperation>(nqubits, target, S);
            } else if (id / 3 % 3 == 2) {
                comp->emplace_back<StandardOperation>(nqubits, target, Sdag);
                comp->emplace_back<StandardOperation>(nqubits, target, H);
            }

            comp->emplace_back<StandardOperation>(nqubits, qc::Control{control}, target, X);
            comp->emplace_back<StandardOperation>(nqubits, qc::Control{target}, control, X);
            comp->emplace_back<StandardOperation>(nqubits, qc::Control{control}, target, X);
        }

        // random Pauli on control qubit
        if (pauliIdx % 4 == 1) {
            comp->emplace_back<StandardOperation>(nqubits, control, Z);
        } else if (pauliIdx % 4 == 2) {
            comp->emplace_back<StandardOperation>(nqubits, control, X);
        } else if (pauliIdx % 4 == 3) {
            comp->emplace_back<StandardOperation>(nqubits, control, Y);
        }

        // random Pauli on target qubit
        if (pauliIdx / 4 == 1) {
            comp->emplace_back<StandardOperation>(nqubits, target, Z);
        } else if (pauliIdx / 4 == 2) {
            comp->emplace_back<StandardOperation>(nqubits, target, X);
        } else if (pauliIdx / 4 == 3) {
            comp->emplace_back<StandardOperation>(nqubits, target, Y);
        }
    }
} // namespace qc
