/*
* This file is part of MQT QFR library which is released under the MIT license.
* See file README.md or go to https://www.cda.cit.tum.de/research/quantum/ for more information.
*/

#pragma once

#include "Ecc.hpp"
#include "QuantumComputation.hpp"

class Q18Surface: public Ecc {
public:
    Q18Surface(std::shared_ptr<qc::QuantumComputation> qc, std::size_t measureFq):
        Ecc({ID::Q18Surface, 36, 0, "Q18Surface", {{ancillaWidth, "qeccX"}, {ancillaWidth, "qeccZ"}}}, std::move(qc), measureFq) {}

    constexpr static std::array<dd::Qubit, 18> dataQubits     = {1, 3, 5, 6, 8, 10, 13, 15, 17, 18, 20, 22, 25, 27, 29, 30, 32, 34};
    constexpr static std::array<dd::Qubit, 18> ancillaIndices = {0, 2, 4, 7, 9, 11, 12, 14, 16, 19, 21, 23, 24, 26, 28, 31, 33, 35};
    constexpr static dd::Qubit                 xInformation   = 14;
    constexpr static std::array<dd::Qubit, 3> logicalX = {5,10,15};
    constexpr static std::array<dd::Qubit, 3> logicalZ = {20,25,30};
    constexpr static dd::QubitCount ancillaWidth = 8;

    //{a,{b,c}} == qubit a is checked by b and c
    std::map<std::size_t, std::vector<std::size_t>> qubitCorrectionX = {
            {1, {0, 2}},
            {3, {2, 4}},
            {5, {4}},
            {6, {0, 12}},
            {8, {2}},
            {10, {4, 16}},
            {13, {12}},
            {15, {16}},
            {18, {12, 24}},
            {20, {26}},
            {22, {16, 28}},
            {25, {24, 26}},
            {27, {26, 28}},
            {29, {28}},
            {30, {24}}
    };

    std::map<std::size_t, std::vector<std::size_t>> qubitCorrectionZ = {
            {5, {11}},
            {6, {7}},
            {8, {7, 9}},
            {10, {9, 11}},
            {13, {7, 19}},
            {15, {9}},
            {17, {11, 23}},
            {20, {19}},
            {22, {23}},
            {25, {19, 31}},
            {27, {33}},
            {29, {23, 35}},
            {30, {31}},
            {32, {31, 33}},
            {34, {33, 35}}};

    static constexpr std::array<std::size_t, 8> xChecks = {0, 2, 4, 12, 16, 24, 26, 28};
    static constexpr std::array<std::size_t, 8> zChecks = {7, 9, 11, 19, 23, 31, 33, 35};

protected:
    void measureAndCorrect() override;

    void writeDecoding() override;

    void mapGate(const qc::Operation& gate) override;
};
