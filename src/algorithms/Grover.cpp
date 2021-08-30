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
        qc.x(nqubits);
        for (dd::QubitCount i = 0; i < nqubits; ++i)
            qc.h(i);
    }

    void Grover::oracle(QuantumComputation& qc) const {
        const std::bitset<64> xBits(targetValue);
        dd::Controls          controls{};
        for (dd::QubitCount i = 0; i < nqubits; ++i) {
            controls.emplace(dd::Control{static_cast<dd::Qubit>(i), xBits[i] ? dd::Control::Type::pos : dd::Control::Type::neg});
        }
        auto target = static_cast<dd::Qubit>(nqubits);
        qc.z(target, controls);
    }

    void Grover::diffusion(QuantumComputation& qc) const {
        //std::vector<unsigned short> targets{};
        for (dd::QubitCount i = 0; i < nqubits; ++i) {
            //targets.push_back(i);
            qc.h(i);
        }
        for (dd::QubitCount i = 0; i < nqubits; ++i) {
            qc.x(i);
        }

        auto target = static_cast<dd::Qubit>(std::max(static_cast<dd::QubitCount>(nqubits - 1), static_cast<dd::QubitCount>(0)));
        qc.h(target);
        dd::Controls controls{};
        for (dd::Qubit j = 0; j < nqubits - 1; ++j) {
            controls.emplace(dd::Control{j});
        }
        qc.x(target, controls);
        qc.h(target);

        for (auto i = static_cast<dd::Qubit>(nqubits - 1); i >= 0; --i) {
            qc.x(i);
        }
        for (auto i = static_cast<dd::Qubit>(nqubits - 1); i >= 0; --i) {
            qc.h(i);
        }
    }

    void Grover::full_grover(QuantumComputation& qc) const {
        // Generate circuit
        setup(qc);

        for (std::size_t j = 0; j < iterations; ++j) {
            oracle(qc);
            diffusion(qc);
        }
        // properly uncompute ancillary qubit
        qc.emplace_back<StandardOperation>(nqubits + nancillae, nqubits, X);
    }

    /***
     * Public Methods
     ***/
    Grover::Grover(dd::QubitCount nq, std::size_t seed):
        seed(seed) {
        name = "grover_" + std::to_string(nq);

        addQubitRegister(nq);
        addAncillaryRegister(1);
        addClassicalRegister(nq + 1);

        std::mt19937_64                            generator(seed);
        std::uniform_int_distribution<std::size_t> distribution(0, static_cast<std::size_t>(std::pow(2.L, std::max(static_cast<dd::QubitCount>(0), nqubits)) - 1.));
        targetValue = distribution(generator);

        if (nqubits <= 3) {
            iterations = 1;
        } else if (nqubits % 2 == 0) {
            iterations = static_cast<std::size_t>(std::round(dd::PI_4 * std::pow(2.L, (nqubits + 1.) / 2.L - 1.) * std::sqrt(2)));
        } else {
            iterations = static_cast<std::size_t>(std::round(dd::PI_4 * std::pow(2.L, (nqubits) / 2.L)));
        }

        full_grover(*this);
        setLogicalQubitGarbage(static_cast<dd::Qubit>(nqubits));
    }

    MatrixDD Grover::buildFunctionality(std::unique_ptr<dd::Package>& dd) const {
        QuantumComputation groverIteration(nqubits + 1);
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

        QuantumComputation qc(nqubits + nancillae);
        setup(qc);
        auto g = qc.buildFunctionality(dd);
        auto f = dd->multiply(e, g);
        dd->incRef(f);
        dd->decRef(e);
        dd->decRef(g);
        e = f;

        // properly handle ancillary qubit
        e = dd->reduceAncillae(e, ancillary);
        e = dd->reduceGarbage(e, garbage);

        dd->decRef(iteration);
        return e;
    }

    std::ostream& Grover::printStatistics(std::ostream& os) const {
        os << "Grover (" << static_cast<std::size_t>(nqubits) << ") Statistics:\n";
        os << "\tn: " << static_cast<std::size_t>(nqubits + 1) << std::endl;
        os << "\tm: " << getNindividualOps() << std::endl;
        os << "\tseed: " << seed << std::endl;
        os << "\tx: " << targetValue << std::endl;
        os << "\ti: " << iterations << std::endl;
        os << "--------------" << std::endl;
        return os;
    }

    MatrixDD Grover::buildFunctionalityRecursive(std::unique_ptr<dd::Package>& dd) const {
        QuantumComputation groverIteration(nqubits + 1);
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
        qc::QuantumComputation statePrep(nqubits + 1);
        setup(statePrep);
        auto s   = statePrep.buildFunctionality(dd);
        auto tmp = dd->multiply(e, s);
        dd->incRef(tmp);
        dd->decRef(s);
        dd->decRef(e);
        e = tmp;

        // properly handle ancillary qubit
        e = dd->reduceAncillae(e, ancillary);
        e = dd->reduceGarbage(e, garbage);

        return e;
    }
} // namespace qc
