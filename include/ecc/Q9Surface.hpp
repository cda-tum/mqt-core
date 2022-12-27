/*
* This file is part of MQT QFR library which is released under the MIT license.
* See file README.md or go to https://www.cda.cit.tum.de/research/quantum/ for more information.
*/

#pragma once

#include "Ecc.hpp"
#include "QuantumComputation.hpp"

//Reference to this ecc in https://arxiv.org/pdf/1608.05053.pdf

class Q9Surface: public Ecc {
public:
    Q9Surface(std::shared_ptr<qc::QuantumComputation> qc, std::size_t measureFq):
        Ecc({ID::Q9Surface, 9, 8, "Q9Surface", {{4, "qeccX"}, {4, "qeccZ"}}}, std::move(qc), measureFq) {}

protected:
    void measureAndCorrect() override;

    void writeDecoding() override;

    void mapGate(const qc::Operation& gate) override;

    // Set parameter for verifying the ecc
    [[maybe_unused]] const size_t insertErrorAfterNGates = 55;

    //{a,{b,c}} == qubit a is checked by b and c
    std::array<std::set<std::size_t>, 9>                            qubitCorrectionX   = {{{1}, {3}, {3}, {1, 4}, {3, 4}, {3, 6}, {4}, {4}, {6}}};
    std::array<std::set<std::size_t>, 9>                            qubitCorrectionZ   = {{{2}, {0, 2}, {0}, {2}, {2, 5}, {5}, {7}, {5, 7}, {5}}};
    static constexpr std::array<dd::Qubit, 4>                       xAncillaQubits     = {1, 3, 4, 6};
    static constexpr std::array<dd::Qubit, 4>                       zAncillaQubits     = {0, 2, 5, 7};
    std::set<dd::Qubit>                                             uncorrectedXQubits = {2, 6};
    std::set<dd::Qubit>                                             uncorrectedZQubits = {0, 8};
    static constexpr dd::QubitCount                                 ancillaWidth       = 4;
    static constexpr std::array<dd::Qubit, 3>                       logicalX           = {2, 4, 6};
    static constexpr std::array<dd::Qubit, 3>                       logicalZ           = {0, 4, 8};
    static constexpr std::array<std::pair<dd::Qubit, dd::Qubit>, 4> swapIndices        = {{{0, 6}, {3, 7}, {2, 8}, {1, 5}}};
};
