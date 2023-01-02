/*
* This file is part of MQT QFR library which is released under the MIT license.
* See file README.md or go to https://www.cda.cit.tum.de/research/quantum/ for more information.
*/

#pragma once

#include "Ecc.hpp"
#include "QuantumComputation.hpp"

//Reference to this ecc in https://arxiv.org/pdf/1608.05053.pdf
namespace ecc {
    class Q9Surface: public Ecc {
    public:
        Q9Surface(std::shared_ptr<qc::QuantumComputation> qc, std::size_t measureFq):
            Ecc({ID::Q9Surface, 9, 8, "Q9Surface", {{4, "qeccX"}, {4, "qeccZ"}}}, std::move(qc), measureFq) {}

    protected:
        void measureAndCorrect() override;

        void writeDecoding() override;

        void mapGate(const qc::Operation& gate) override;

    private:
        //{a,{b,c}} == qubit a is checked by b and c
        std::array<std::set<std::size_t>, 9>                    qubitCorrectionX   = {{{1}, {3}, {3}, {1, 4}, {3, 4}, {3, 6}, {4}, {4}, {6}}};
        std::array<std::set<std::size_t>, 9>                    qubitCorrectionZ   = {{{2}, {0, 2}, {0}, {2}, {2, 5}, {5}, {7}, {5, 7}, {5}}};
        static constexpr std::array<Qubit, 4>                   X_ANCILLA_QUBITS   = {1, 3, 4, 6};
        static constexpr std::array<Qubit, 4>                   Z_ANCILLA_QUBITS   = {0, 2, 5, 7};
        std::set<Qubit>                                         uncorrectedXQubits = {2, 6};
        std::set<Qubit>                                         uncorrectedZQubits = {0, 8};
        static constexpr QubitCount                             ANCILLA_WIDTH      = 4;
        static constexpr std::array<Qubit, 3>                   LOGICAL_X          = {2, 4, 6};
        static constexpr std::array<Qubit, 3>                   LOGICAL_Z          = {0, 4, 8};
        static constexpr std::array<std::pair<Qubit, Qubit>, 4> SWAP_INDICES       = {{{0, 6}, {3, 7}, {2, 8}, {1, 5}}};
    };
} // namespace ecc
