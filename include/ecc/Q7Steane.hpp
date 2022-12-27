/*
* This file is part of MQT QFR library which is released under the MIT license.
* See file README.md or go to https://www.cda.cit.tum.de/research/quantum/ for more information.
*/

#pragma once

#include "Ecc.hpp"
#include "QuantumComputation.hpp"
namespace ecc {
    class Q7Steane: public Ecc {
    public:
        Q7Steane(std::shared_ptr<qc::QuantumComputation> qc, std::size_t measureFq):
            Ecc(
                    {ID::Q7Steane, 7, 3, "Q7Steane", {{3, "qecc"}}}, std::move(qc), measureFq) {}

    protected:
        void writeEncoding() override;

        void measureAndCorrect() override;
        void measureAndCorrectSingle(bool xSyndrome);

        void writeDecoding() override;

        void mapGate(const qc::Operation& gate) override;
    };
} // namespace ecc
