//
// Created by Lukas Burgholzer on 09.12.19.
//

#include "Grover.hpp"


namespace qc {
    /***
     * Private Methods
     ***/
	void Grover::setup(QuantumComputation& qc) {
        qc.emplace_back<StandardOperation>(nqubits, nqubits-1, X);
        for (unsigned short i = 0; i < nqubits; ++i)
            qc.emplace_back<StandardOperation>(nqubits, i, H);
    }

    void Grover::oracle(QuantumComputation& qc) {
        const std::bitset<64> xBits(x);
        std::vector<Control> controls{};
        for (unsigned short i = 0; i < nqubits-1; ++i) {
            controls.emplace_back(i, xBits[i]? Control::pos: Control::neg);
        }
        unsigned short target = nqubits-1;
        qc.emplace_back<StandardOperation>(nqubits, controls, target);
    }

    void Grover::diffusion(QuantumComputation& qc) {
        //std::vector<unsigned short> targets{};
        for (unsigned short i = 0; i < nqubits-1; ++i) {
            //targets.push_back(i);
            qc.emplace_back<StandardOperation>(nqubits, i, H);
        }
        for (unsigned short i = 0; i < nqubits-1; ++i) {
            qc.emplace_back<StandardOperation>(nqubits, i, X);
        }

        //qc.emplace_back<StandardOperation>(nqubits, targets, H);
        //qc.emplace_back<StandardOperation>(nqubits, targets, X);

        unsigned short target = std::max(nqubits-2, 0);
        qc.emplace_back<StandardOperation>(nqubits, target, H);
        std::vector<Control> controls{};
        for (unsigned short j = 0; j < nqubits-2; ++j) {
            controls.emplace_back(j);
        }
        qc.emplace_back<StandardOperation>(nqubits, controls, target);
        qc.emplace_back<StandardOperation>(nqubits, target, H);

        for (int i = nqubits-2; i >= 0; --i) {
            qc.emplace_back<StandardOperation>(nqubits, i, X);
        }
        for (int i = nqubits-2; i >= 0; --i) {
            qc.emplace_back<StandardOperation>(nqubits, i, H);
        }

        //qc.emplace_back<StandardOperation>(nqubits, targets, X);
        //qc.emplace_back<StandardOperation>(nqubits, targets, H);
    }

    void Grover::full_grover(QuantumComputation& qc) {
        // Generate circuit
        setup(qc);

        for (unsigned long long j = 0; j < iterations; ++j) {
            oracle(qc);
            diffusion(qc);
        }
    }

    /***
     * Public Methods
     ***/
    Grover::Grover(unsigned short nq, unsigned int seed, bool includeSetup) {
        nqubits = nq+1;
        this->seed = seed;
        this->includeSetup = includeSetup;
        for (unsigned short i = 0; i < nqubits; ++i) {
            inputPermutation.insert({i, i});
            outputPermutation.insert({i, i});
        }
	    qregs.insert({"q", {0, nq}});
        qregs.insert({"anc", {nq, 1}});
	    cregs.insert({"c", {0, nq}});
	    cregs.insert({"c_anc", {nq, 1}});

        std::mt19937_64 generator(this->seed);
        std::uniform_int_distribution<unsigned long long> distribution(0, (unsigned long long) (std::pow((long double)2, std::max(0,nqubits-1)) - 1));
        oracleGenerator = bind(distribution, ref(generator));
        x = oracleGenerator();

        if (nqubits <= 3) {
            iterations = 1;
        } else if (nqubits%2 == 0) {
            iterations = (unsigned long long)std::round(PI_4 * std::pow(2.L, (nqubits)/2.L-1) * std::sqrt(2));
        } else {
            iterations = (unsigned long long)std::round(PI_4 * std::pow(2.L, (nqubits-1)/2.L));
        }
        if (includeSetup) {
            full_grover(*this);
        } else {
            for (unsigned long long i = 0; i < iterations; ++i) {
                oracle(*this);
                diffusion(*this);
            }
        }
    }

    dd::Edge Grover::buildFunctionality(std::unique_ptr<dd::Package>& dd) {
        dd->useMatrixNormalization(true);

        QuantumComputation groverIteration(nqubits);
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
        if(includeSetup) {
            QuantumComputation qc(nqubits);
            this->setup(qc);
            auto g = qc.buildFunctionality(dd);
            dd::Edge f = dd->multiply(e, g);
            dd->decRef(e);
            dd->decRef(g);
            dd->incRef(f);
            e = f;
        }
        dd->decRef(iteration);
        dd->garbageCollect(true);
        dd->useMatrixNormalization(false);
        return e;
    }

    dd::Edge Grover::simulate(const dd::Edge& in, std::unique_ptr<dd::Package>& dd) {
        //TODO: Enhance this simulation routine // delegate to simulator
        return QuantumComputation::simulate(in, dd);
    }

    std::ostream& Grover::printStatistics(std::ostream& os) {
        os << "Grover (" << nqubits-1 << ") Statistics:\n";
        os << "\tn: " << nqubits << std::endl;
        os << "\tm: " << getNindividualOps() << std::endl;
        os << "\tseed: " << seed << std::endl;
        os << "\tx: " << x << std::endl;
        os << "\ti: " << iterations << std::endl;
        os << "\tinit: " << (includeSetup? "yes" : "no") << std::endl;
        os << "--------------" << std::endl;
        return os;
    }
}
