/*
 * This file is part of IIC-JKU QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#include "algorithms/Grover.hpp"


namespace qc {
    /***
     * Private Methods
     ***/
	void Grover::setup(QuantumComputation& qc) {
        qc.emplace_back<StandardOperation>(nqubits+nancillae, nqubits, X);
        for (unsigned short i = 0; i < nqubits; ++i)
            qc.emplace_back<StandardOperation>(nqubits+nancillae, i, H);
    }

    void Grover::oracle(QuantumComputation& qc) {
        const std::bitset<64> xBits(x);
        std::vector<Control> controls{};
        for (unsigned short i = 0; i < nqubits; ++i) {
            controls.emplace_back(i, xBits[i]? Control::pos: Control::neg);
        }
        unsigned short target = nqubits;
        qc.emplace_back<StandardOperation>(nqubits+nancillae, controls, target, qc::Z);
    }

    void Grover::diffusion(QuantumComputation& qc) {
        //std::vector<unsigned short> targets{};
        for (unsigned short i = 0; i < nqubits; ++i) {
            //targets.push_back(i);
            qc.emplace_back<StandardOperation>(nqubits+nancillae, i, H);
        }
        for (unsigned short i = 0; i < nqubits; ++i) {
            qc.emplace_back<StandardOperation>(nqubits+nancillae, i, X);
        }

        //qc.emplace_back<StandardOperation>(nqubits+nancillae, targets, H);
        //qc.emplace_back<StandardOperation>(nqubits+nancillae, targets, X);

        auto target = static_cast<unsigned short>(std::max(nqubits-1, 0));
        qc.emplace_back<StandardOperation>(nqubits+nancillae, target, H);
        std::vector<Control> controls{};
        for (unsigned short j = 0; j < nqubits-1; ++j) {
            controls.emplace_back(j);
        }
        qc.emplace_back<StandardOperation>(nqubits+nancillae, controls, target);
        qc.emplace_back<StandardOperation>(nqubits+nancillae, target, H);

        for (auto i = static_cast<short>(nqubits-1); i >= 0; --i) {
            qc.emplace_back<StandardOperation>(nqubits+nancillae, i, X);
        }
        for (auto i = static_cast<short>(nqubits-1); i >= 0; --i) {
            qc.emplace_back<StandardOperation>(nqubits+nancillae, i, H);
        }

        //qc.emplace_back<StandardOperation>(nqubits+nancillae, targets, X);
        //qc.emplace_back<StandardOperation>(nqubits+nancillae, targets, H);
    }

    void Grover::full_grover(QuantumComputation& qc) {
        // Generate circuit
        setup(qc);

        for (unsigned long long j = 0; j < iterations; ++j) {
            oracle(qc);
            diffusion(qc);
        }
	    // properly uncompute ancillary qubit
	    qc.emplace_back<StandardOperation>(nqubits+nancillae, nqubits, X);
    }

    /***
     * Public Methods
     ***/
    Grover::Grover(unsigned short nq, unsigned int seed) : seed(seed) {
        name = "grover_" + std::to_string(nq);

        addQubitRegister(nq);
        addAncillaryRegister(1);
        addClassicalRegister(nq+1);

	    for (unsigned short i = 0; i <= nqubits; ++i) {
		    initialLayout.insert({ i, i});
		    outputPermutation.insert({ i, i});
	    }
	    line.fill(LINE_DEFAULT);

        std::mt19937_64 generator(this->seed);
        std::uniform_int_distribution<unsigned long long> distribution(0, static_cast<unsigned long long>(std::pow((long double)2, std::max(static_cast<unsigned short>(0),nqubits)) - 1));
        oracleGenerator = [&]() { return distribution(generator); };
        x = oracleGenerator();

        if (nqubits <= 3) {
            iterations = 1;
        } else if (nqubits%2 == 0) {
            iterations = (unsigned long long)std::round(PI_4 * std::pow(2.L, (nqubits+1)/2.L-1) * std::sqrt(2));
        } else {
            iterations = (unsigned long long)std::round(PI_4 * std::pow(2.L, (nqubits)/2.L));
        }

        full_grover(*this);

    }

    dd::Edge Grover::buildFunctionality(std::unique_ptr<dd::Package>& dd) {
        dd->setMode(dd::Matrix);

        QuantumComputation groverIteration(nqubits+1);
        oracle(groverIteration);
        diffusion(groverIteration);

        dd::Edge iteration = groverIteration.buildFunctionality(dd);

        dd::Edge e = iteration;
        dd->incRef(e);

        for (unsigned long long i = 0; i < iterations-1; ++i) {
            dd::Edge f = dd->multiply(iteration, e);
            dd->decRef(e);
            e = f;
            dd->incRef(e);
            dd->garbageCollect();
        }

        QuantumComputation qc(nqubits+nancillae);
        setup(qc);
        auto g = qc.buildFunctionality(dd);
        dd::Edge f = dd->multiply(e, g);
        dd->decRef(e);
        dd->decRef(g);
        dd->incRef(f);
        e = f;

        // properly uncompute ancillary qubit
	    f = dd->multiply(StandardOperation(nqubits+nancillae, nqubits, X).getDD(dd, line), e);
	    dd->decRef(e);
	    dd->incRef(f);
	    e = f;

        auto q = removeQubit(nqubits);
        addAncillaryQubit(q.first, q.second);
        e = reduceAncillae(e, dd);

        dd->decRef(iteration);
        dd->garbageCollect(true);
        return e;
    }

    dd::Edge Grover::simulate(const dd::Edge& in, std::unique_ptr<dd::Package>& dd) {
        //TODO: Enhance this simulation routine // delegate to simulator

        // initial state too small
        dd::Edge initialState = in;
        if (in.p->v == nqubits-1) {
        	initialState = dd->extend(in, 1);
        }
        return QuantumComputation::simulate(initialState, dd);
    }

    std::ostream& Grover::printStatistics(std::ostream& os) {
        os << "Grover (" << nqubits << ") Statistics:\n";
        os << "\tn: " << nqubits+1 << std::endl;
        os << "\tm: " << getNindividualOps() << std::endl;
        os << "\tseed: " << seed << std::endl;
        os << "\tx: " << x << std::endl;
        os << "\ti: " << iterations << std::endl;
        os << "--------------" << std::endl;
        return os;
    }
}
