/*
* This file is part of MQT QFR library which is released under the MIT license.
* See file README.md or go to https://www.cda.cit.tum.de/research/quantum/ for more information.
*/

#pragma once

#include "Ecc.hpp"
#include "QuantumComputation.hpp"
namespace ecc {
    class Q3Shor: public Ecc {
    public:
        Q3Shor(std::shared_ptr<qc::QuantumComputation> qc, std::size_t measureFq):
            Ecc(
                    {ID::Q3Shor, N_REDUNDANT_QUBITS, 2, "Q3Shor", {{2, "qecc"}}}, std::move(qc), measureFq) {}

    protected:
        void writeEncoding() override;

        void measureAndCorrect() override;

        void writeDecoding() override;

        void mapGate(const qc::Operation& gate) override;

        static constexpr std::size_t N_REDUNDANT_QUBITS = 3;
    };
} // namespace ecc
