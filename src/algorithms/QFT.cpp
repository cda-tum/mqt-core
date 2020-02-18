/*
 * This file is part of IIC-JKU QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#include "algorithms/QFT.hpp"


namespace qc {
	QFT::QFT(unsigned short nq, bool performSwaps) {
        nqubits = nq;
        this->performSwaps = performSwaps;
        for (unsigned short i = 0; i < nqubits; ++i) {
        	initialLayout.insert({i, i});
        	outputPermutation.insert({i, i});
        }
        qregs.insert({"q", {0, nqubits}});
        cregs.insert({"c", {0, nqubits}});

        for (unsigned short i = 0; i < nqubits; ++i) {
            emplace_back<StandardOperation>(nqubits, i, H);
            for (int j = 1; j < nqubits-i; ++j) {
                long double powerOfTwo = std::pow(2.L, j);
                long double lambda = qc::PI / powerOfTwo;
                if(j == 1) {
                    emplace_back<StandardOperation>(nqubits, Control(i+1), i, S);
                } else if (j == 2) {
                    emplace_back<StandardOperation>(nqubits, Control(i+2), i, T);
                } else {
                    emplace_back<StandardOperation>(nqubits, Control(i+j), i, RZ, lambda);
                }
            }
        }

        if (performSwaps) {
            for (unsigned short i = 0; i < nqubits/2; ++i) {
                emplace_back<StandardOperation>(nqubits, std::vector<Control>{}, i, (unsigned short)(nqubits-1-i), SWAP);
            }
        }
    }

	dd::Edge QFT::buildFunctionality(std::unique_ptr<dd::Package>& dd) {
		auto e = QuantumComputation::buildFunctionality(dd);
		if (!performSwaps) {
			for (unsigned short i = 0; i < nqubits; ++i) {
				outputPermutation.at(i) = nqubits - 1 - i;
			}
		}
		return e;
	}

	dd::Edge QFT::simulate(const dd::Edge& in, std::unique_ptr<dd::Package>& dd) {
		auto e = QuantumComputation::simulate(in, dd);
		if (!performSwaps) {
			for (unsigned short i = 0; i < nqubits; ++i) {
				outputPermutation.at(i) = nqubits - 1 - i;
			}
		}
		return e;
	}

	std::ostream& QFT::printStatistics(std::ostream& os) {
        os << "QFT (" << nqubits << ") Statistics:\n";
        os << "\tn: " << nqubits << std::endl;
        os << "\tm: " << getNindividualOps() << std::endl;
        os << "\tswaps: " << (performSwaps? "yes" : "no") << std::endl;
        os << "--------------" << std::endl;
        return os;
    }
}
