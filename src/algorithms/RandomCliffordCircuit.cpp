/*
 * This file is part of IIC-JKU QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#include "algorithms/RandomCliffordCircuit.hpp"

namespace qc {


	RandomCliffordCircuit::RandomCliffordCircuit(unsigned short nq, unsigned int depth, unsigned int seed) {
		this->depth = depth;
		this->seed = seed;

		addQubitRegister(nq);
		addClassicalRegister(nq);

		for (unsigned short i = 0; i < nqubits; ++i) {
			initialLayout.insert({ i, i});
			outputPermutation.insert({ i, i});
		}

		std::mt19937_64 generator;
		if(seed == 0) {
			// this is probably overkill but better safe than sorry
			std::array<std::mt19937_64::result_type , std::mt19937_64::state_size> random_data{};
			std::random_device rd;
			std::generate(std::begin(random_data), std::end(random_data), std::ref(rd));
			std::seed_seq seeds(std::begin(random_data), std::end(random_data));
			generator.seed(seeds);
		} else {
			generator.seed(seed);
		}
		std::uniform_int_distribution<unsigned short> distribution(0, 11520);
		cliffordGenerator = [&]() { return distribution(generator); };

		for (unsigned int l=0; l<depth; ++l) {
			if (nqubits == 1) {
				append1QClifford(cliffordGenerator(), 0);
			} else if (nqubits == 2) {
				append2QClifford(cliffordGenerator(), 0, 1);
			} else {
				if (l%2) {
					for (int i=1; i<nqubits-1; i+=2) {
						append2QClifford(cliffordGenerator(), i, i+1);
					}
				} else {
					for (int i=0; i<nqubits-1; i+=2) {
						append2QClifford(cliffordGenerator(), i, i+1);
					}
				}
			}
		}
	}

	std::ostream& RandomCliffordCircuit::printStatistics(std::ostream& os) {
		os << "Random Clifford circuit statistics:\n";
		os << "\tn: " << nqubits << std::endl;
		os << "\tm: " << getNindividualOps() << std::endl;
		os << "\tdepth: " << depth << std::endl;
		os << "\tseed: " << seed << std::endl;
		os << "--------------" << std::endl;
		return os;
	}

	void RandomCliffordCircuit::append1QClifford(unsigned int idx, unsigned short target) {
		unsigned short id = idx % 24;
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

	void RandomCliffordCircuit::append2QClifford(unsigned int idx, unsigned short control, unsigned short target) {
		unsigned short id = idx % 11520;
		unsigned short pauliIdx = id % 16;
		id /= 16;

		emplace_back<CompoundOperation>(nqubits);
		auto comp = dynamic_cast<CompoundOperation*>(ops.back().get());
		if (id < 36) {
			if (id / 9 % 2)
				comp->emplace_back<StandardOperation>(nqubits, control, H);
			if (id / 18 % 2)
				comp->emplace_back<StandardOperation>(nqubits, target, H);

			if (id % 3 == 1){
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

			if (id % 3 == 1){
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

			comp->emplace_back<StandardOperation>(nqubits, Control(control), target, X);

			if (id / 9 % 3 == 1){
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

			if (id % 3 == 1){
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

			comp->emplace_back<StandardOperation>(nqubits, Control(control), target, X);
			comp->emplace_back<StandardOperation>(nqubits, Control(target), control, X);

			if (id / 9 % 3 == 1){
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

			if (id % 3 == 1){
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

			comp->emplace_back<StandardOperation>(nqubits, Control(control), target, X);
			comp->emplace_back<StandardOperation>(nqubits, Control(target), control, X);
			comp->emplace_back<StandardOperation>(nqubits, Control(control), target, X);
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
}
