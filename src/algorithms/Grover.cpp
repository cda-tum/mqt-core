/*
 * This file is part of MQT QFR library which is released under the MIT license.
 * See file README.md or go to https://www.cda.cit.tum.de/research/quantum/ for more information.
 */

#include "algorithms/Grover.hpp"

namespace qc {
    /***
     * Private Methods
     ***/
    void Grover::setup(QuantumComputation& qc) const {
        qc.x(static_cast<Qubit>(nDataQubits));
        for (std::size_t i = 0; i < nDataQubits; ++i) {
            qc.h(static_cast<Qubit>(i));
        }
    }

    void Grover::oracle(QuantumComputation& qc) const {
        Controls controls{};
        for (std::size_t i = 0; i < nDataQubits; ++i) {
            controls.emplace(Control{static_cast<Qubit>(i), targetValue.test(i) ? Control::Type::Pos : Control::Type::Neg});
        }
        qc.z(static_cast<Qubit>(nDataQubits), controls);
    }

    void Grover::diffusion(QuantumComputation& qc) const {
        for (std::size_t i = 0; i < nDataQubits; ++i) {
            qc.h(static_cast<Qubit>(i));
        }
        for (std::size_t i = 0; i < nDataQubits; ++i) {
            qc.x(static_cast<Qubit>(i));
        }

        qc.h(0);
        Controls controls{};
        for (Qubit j = 1; j < nDataQubits; ++j) {
            controls.emplace(Control{j});
        }
        qc.x(0, controls);
        qc.h(0);

        for (auto i = static_cast<std::make_signed_t<Qubit>>(nDataQubits - 1); i >= 0; --i) {
            qc.x(static_cast<Qubit>(i));
        }
        for (auto i = static_cast<std::make_signed_t<Qubit>>(nDataQubits - 1); i >= 0; --i) {
            qc.h(static_cast<Qubit>(i));
        }
    }

    void Grover::fullGrover(QuantumComputation& qc) const {
        // create initial superposition
        setup(qc);

        // apply Grover iterations
        for (std::size_t j = 0; j < iterations; ++j) {
            oracle(qc);
            diffusion(qc);
        }

        // measure the resulting state
        for (std::size_t i = 0; i < nDataQubits; ++i) {
            qc.measure(static_cast<Qubit>(i), i);
        }
    }

    /***
     * Public Methods
     ***/
    Grover::Grover(std::size_t nq, std::size_t s):
        seed(s), nDataQubits(nq) {
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
        while (expected.length() > nqubits - 1) {
            expected.pop_back();
        }
        std::reverse(expected.begin(), expected.end());

        if (nDataQubits <= 2) {
            iterations = 1;
        } else if (nDataQubits % 2 == 1) {
            iterations = static_cast<std::size_t>(std::round(PI_4 * std::pow(2., static_cast<double>(nDataQubits + 1) / 2. - 1.) * std::sqrt(2)));
        } else {
            iterations = static_cast<std::size_t>(std::round(PI_4 * std::pow(2., static_cast<double>(nDataQubits) / 2.)));
        }

        fullGrover(*this);
    }

    std::ostream& Grover::printStatistics(std::ostream& os) const {
        os << "Grover (" << nqubits - 1 << ") Statistics:\n";
        os << "\tn: " << nqubits << std::endl;
        os << "\tm: " << getNindividualOps() << std::endl;
        os << "\tseed: " << seed << std::endl;
        os << "\tx: " << expected << std::endl;
        os << "\ti: " << iterations << std::endl;
        os << "--------------" << std::endl;
        return os;
    }
} // namespace qc
