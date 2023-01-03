/*
* This file is part of MQT QFR library which is released under the MIT license.
* See file README.md or go to https://www.cda.cit.tum.de/research/quantum/ for more information.
*/

#pragma once

#include "Ecc.hpp"
#include "QuantumComputation.hpp"
namespace ecc {
    class Q5Laflamme: public Ecc {
    public:
        Q5Laflamme(std::shared_ptr<qc::QuantumComputation> qc, std::size_t measureFq):
            Ecc({ID::Q5Laflamme, 5, 4, "Q5Laflamme", {{4, "qecc"}, {1, "encode"}}}, std::move(qc), measureFq) {}

    protected:
        void writeEncoding() override;

        void measureAndCorrect() override;

        void writeDecoding() override;

        void mapGate(const qc::Operation& gate) override;

        static constexpr std::array<std::array<qc::OpType, 5>, 4> STABILIZER_MATRIX = {{
                {qc::X, qc::Z, qc::Z, qc::X, qc::I}, //c0
                {qc::I, qc::X, qc::Z, qc::Z, qc::X}, //c1
                {qc::X, qc::I, qc::X, qc::Z, qc::Z}, //c2
                {qc::Z, qc::X, qc::I, qc::X, qc::Z}  //c3
        }};

        static constexpr std::array<Qubit, 8> DECODING_CORRECTION_VALUES = {1, 2, 4, 7, 8, 11, 13, 14}; //values with odd amount of '1' bits
    };
} // namespace ecc
