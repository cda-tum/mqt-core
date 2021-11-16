/*
 * This file is part of JKQ QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#include "algorithms/Grover.hpp"

namespace qc {
    /***
     * Private Methods
     ***/
    void Grover::setup(QuantumComputation& qc) const {
        qc.x(static_cast<dd::Qubit>(nDataQubits));
        for (dd::QubitCount i = 0; i < nDataQubits; ++i)
            qc.h(static_cast<dd::Qubit>(i));
    }

    void Grover::oracle(QuantumComputation& qc) const {
        dd::Controls controls{};
        for (dd::QubitCount i = 0; i < nDataQubits; ++i) {
            controls.emplace(dd::Control{static_cast<dd::Qubit>(i), targetValue.test(i) ? dd::Control::Type::pos : dd::Control::Type::neg});
        }
        qc.z(static_cast<dd::Qubit>(nDataQubits), controls);
    }

    void Grover::diffusion(QuantumComputation& qc) const {
        for (dd::QubitCount i = 0; i < nDataQubits; ++i) {
            qc.h(static_cast<dd::Qubit>(i));
        }
        for (dd::QubitCount i = 0; i < nDataQubits; ++i) {
            qc.x(static_cast<dd::Qubit>(i));
        }

        qc.h(0);
        dd::Controls controls{};
        for (dd::Qubit j = 1; j < nDataQubits; ++j) {
            controls.emplace(dd::Control{j});
        }
        qc.x(0, controls);
        qc.h(0);

        for (auto i = static_cast<dd::Qubit>(nDataQubits - 1); i >= 0; --i) {
            qc.x(i);
        }
        for (auto i = static_cast<dd::Qubit>(nDataQubits - 1); i >= 0; --i) {
            qc.h(i);
        }
    }

    void Grover::full_grover(QuantumComputation& qc) const {
        // create initial superposition
        setup(qc);

        // apply Grover iterations
        for (std::size_t j = 0; j < iterations; ++j) {
            oracle(qc);
            diffusion(qc);
        }

        // measure the resulting state
        for (dd::QubitCount i = 0; i < nDataQubits; ++i)
            qc.measure(static_cast<dd::Qubit>(i), i);
    }

    /***
     * Public Methods
     ***/
    Grover::Grover(dd::QubitCount nq, std::size_t seed):
        seed(seed), nDataQubits(nq) {
        name = "grover_" + std::to_string(nq);

        addQubitRegister(nDataQubits, "q");
        addQubitRegister(1, "flag");
        addClassicalRegister(nDataQubits);

        mt.seed(seed);

        std::bernoulli_distribution distribution{};
        for (std::size_t i = 0; i < nDataQubits; i++) {
            if (distribution(mt)) {
                targetValue.set(i);
            }
        }

        expected = targetValue.to_string();
        std::reverse(expected.begin(), expected.end());
        while (expected.length() > static_cast<std::size_t>(nqubits - 1))
            expected.pop_back();
        std::reverse(expected.begin(), expected.end());

        if (nDataQubits <= 3) {
            iterations = 1;
        } else if (nDataQubits % 2 == 0) {
            iterations = static_cast<std::size_t>(std::round(dd::PI_4 * std::pow(2.L, (nDataQubits + 1.) / 2.L - 1.) * std::sqrt(2)));
        } else {
            iterations = static_cast<std::size_t>(std::round(dd::PI_4 * std::pow(2.L, (nDataQubits) / 2.L)));
        }

        full_grover(*this);
    }

    MatrixDD Grover::buildFunctionality(std::unique_ptr<dd::Package>& dd) const {
        QuantumComputation groverIteration(nqubits);
        oracle(groverIteration);
        diffusion(groverIteration);

        auto iteration = groverIteration.buildFunctionality(dd);

        auto e = iteration;
        dd->incRef(e);

        for (std::size_t i = 0; i < iterations - 1; ++i) {
            auto f = dd->multiply(iteration, e);
            dd->incRef(f);
            dd->decRef(e);
            e = f;
            dd->garbageCollect();
        }

        QuantumComputation qc(nqubits);
        setup(qc);
        auto g = qc.buildFunctionality(dd);
        auto f = dd->multiply(e, g);
        dd->incRef(f);
        dd->decRef(e);
        dd->decRef(g);
        e = f;

        dd->decRef(iteration);
        return e;
    }

    std::ostream& Grover::printStatistics(std::ostream& os) const {
        os << "Grover (" << static_cast<std::size_t>(nqubits - 1) << ") Statistics:\n";
        os << "\tn: " << static_cast<std::size_t>(nqubits) << std::endl;
        os << "\tm: " << getNindividualOps() << std::endl;
        os << "\tseed: " << seed << std::endl;
        os << "\tx: " << expected << std::endl;
        os << "\ti: " << iterations << std::endl;
        os << "--------------" << std::endl;
        return os;
    }

    MatrixDD Grover::buildFunctionalityRecursive(std::unique_ptr<dd::Package>& dd) const {
        QuantumComputation groverIteration(nqubits);
        oracle(groverIteration);
        diffusion(groverIteration);

        auto            iter = groverIteration.buildFunctionalityRecursive(dd);
        auto            e    = iter;
        std::bitset<64> iterBits(iterations);
        auto            msb = static_cast<std::size_t>(std::floor(std::log2(iterations)));
        auto            f   = iter;
        dd->incRef(f);
        bool zero = !iterBits[0];
        for (std::size_t j = 1; j <= msb; ++j) {
            auto tmp = dd->multiply(f, f);
            dd->incRef(tmp);
            dd->decRef(f);
            f = tmp;
            if (iterBits[j]) {
                if (zero) {
                    dd->incRef(f);
                    dd->decRef(e);
                    e    = f;
                    zero = false;
                } else {
                    auto g = dd->multiply(e, f);
                    dd->incRef(g);
                    dd->decRef(e);
                    e = g;
                    dd->garbageCollect();
                }
            }
        }
        dd->decRef(f);

        // apply state preparation setup
        qc::QuantumComputation statePrep(nqubits);
        setup(statePrep);
        auto s   = statePrep.buildFunctionality(dd);
        auto tmp = dd->multiply(e, s);
        dd->incRef(tmp);
        dd->decRef(s);
        dd->decRef(e);
        e = tmp;

        return e;
    }
} // namespace qc
