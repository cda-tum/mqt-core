/*
 * This file is part of JKQ QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#include "algorithms/RandomCliffordCircuit.hpp"

namespace qc {

    RandomCliffordCircuit::RandomCliffordCircuit(dd::QubitCount nq, std::size_t depth, std::size_t seed):
        depth(depth), seed(seed) {
        addQubitRegister(nq);
        addClassicalRegister(nq);

        for (dd::QubitCount i = 0; i < nqubits; ++i) {
            initialLayout.insert({i, i});
            outputPermutation.insert({i, i});
        }

        std::mt19937_64 generator;
        if (seed == 0) {
            // this is probably overkill but better safe than sorry
            std::array<std::mt19937_64::result_type, std::mt19937_64::state_size> random_data{};
            std::random_device                                                    rd;
            std::generate(std::begin(random_data), std::end(random_data), std::ref(rd));
            std::seed_seq seeds(std::begin(random_data), std::end(random_data));
            generator.seed(seeds);
        } else {
            generator.seed(seed);
        }
        std::uniform_int_distribution<std::uint_fast16_t> distribution(0, 11520);
        cliffordGenerator = [&]() { return distribution(generator); };

        for (std::size_t l = 0; l < depth; ++l) {
            if (nqubits == 1) {
                append1QClifford(cliffordGenerator(), 0);
            } else if (nqubits == 2) {
                append2QClifford(cliffordGenerator(), 0, 1);
            } else {
                if (l % 2) {
                    for (dd::Qubit i = 1; i < static_cast<dd::Qubit>(nqubits - 1); i += 2) {
                        append2QClifford(cliffordGenerator(), i, static_cast<dd::Qubit>(i + 1));
                    }
                } else {
                    for (dd::Qubit i = 0; i < static_cast<dd::Qubit>(nqubits - 1); i += 2) {
                        append2QClifford(cliffordGenerator(), i, static_cast<dd::Qubit>(i + 1));
                    }
                }
            }
        }
    }

    std::ostream& RandomCliffordCircuit::printStatistics(std::ostream& os) const {
        os << "Random Clifford circuit statistics:\n";
        os << "\tn: " << static_cast<std::size_t>(nqubits) << std::endl;
        os << "\tm: " << getNindividualOps() << std::endl;
        os << "\tdepth: " << depth << std::endl;
        os << "\tseed: " << seed << std::endl;
        os << "--------------" << std::endl;
        return os;
    }

    void RandomCliffordCircuit::append1QClifford(std::uint_fast16_t idx, dd::Qubit target) {
        std::uint_fast8_t id = idx % 24;
        emplace_back<CompoundOperation>(nqubits);
        auto comp = dynamic_cast<CompoundOperation*>(ops.back().get());
        // Hadamard
        if (id / 12 % 2) {
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

    void RandomCliffordCircuit::append2QClifford(std::uint_fast16_t idx, dd::Qubit control, dd::Qubit target) {
        std::uint_fast16_t id       = idx % 11520;
        std::uint_fast8_t  pauliIdx = id % 16;
        id /= 16;

        emplace_back<CompoundOperation>(nqubits);
        auto comp = dynamic_cast<CompoundOperation*>(ops.back().get());
        if (id < 36) {
            if (id / 9 % 2)
                comp->emplace_back<StandardOperation>(nqubits, control, H);
            if (id / 18 % 2)
                comp->emplace_back<StandardOperation>(nqubits, target, H);

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
            id -= 36;

            if (id / 81 % 2)
                comp->emplace_back<StandardOperation>(nqubits, control, H);
            if (id / 162 % 2)
                comp->emplace_back<StandardOperation>(nqubits, target, H);

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

            comp->emplace_back<StandardOperation>(nqubits, dd::Control{control}, target, X);

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
            id -= 360;

            if (id / 81 % 2)
                comp->emplace_back<StandardOperation>(nqubits, control, H);
            if (id / 162 % 2)
                comp->emplace_back<StandardOperation>(nqubits, target, H);

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

            comp->emplace_back<StandardOperation>(nqubits, dd::Control{control}, target, X);
            comp->emplace_back<StandardOperation>(nqubits, dd::Control{target}, control, X);

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
            id -= 684;

            if (id / 9 % 2)
                comp->emplace_back<StandardOperation>(nqubits, control, H);
            if (id / 18 % 2)
                comp->emplace_back<StandardOperation>(nqubits, target, H);

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

            comp->emplace_back<StandardOperation>(nqubits, dd::Control{control}, target, X);
            comp->emplace_back<StandardOperation>(nqubits, dd::Control{target}, control, X);
            comp->emplace_back<StandardOperation>(nqubits, dd::Control{control}, target, X);
        }

        if (pauliIdx % 4 == 1) {
            comp->emplace_back<StandardOperation>(nqubits, control, Z);
        } else if (pauliIdx % 4 == 2) {
            comp->emplace_back<StandardOperation>(nqubits, control, X);
        } else if (pauliIdx % 4 == 3) {
            comp->emplace_back<StandardOperation>(nqubits, control, Y);
        }

        if (pauliIdx / 4 == 1) {
            comp->emplace_back<StandardOperation>(nqubits, target, Z);
        } else if (pauliIdx / 4 == 2) {
            comp->emplace_back<StandardOperation>(nqubits, target, X);
        } else if (pauliIdx / 4 == 3) {
            comp->emplace_back<StandardOperation>(nqubits, target, Y);
        }
    }
} // namespace qc
