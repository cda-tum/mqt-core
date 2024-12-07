/*
 * Copyright (c) 2024 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "dd/GateMatrixDefinitions.hpp"

#include "dd/DDDefinitions.hpp"
#include "ir/operations/OpType.hpp"

#include <complex>
#include <unordered_map>
#include <vector>

namespace dd {

const std::unordered_map<qc::OpType,
                         GateMatrix (*const)(const std::vector<fp>&)>
    MATS_GENERATORS{{
        // clang-format off
        {qc::I,   [](const std::vector<fp>&) -> GateMatrix { return {1, 0, 0, 1}; }},
        {qc::H,   [](const std::vector<fp>&) -> GateMatrix { return {SQRT2_2, SQRT2_2, SQRT2_2, -SQRT2_2}; }},
        {qc::X,   [](const std::vector<fp>&) -> GateMatrix { return {0, 1, 1, 0}; }},
        {qc::Y,   [](const std::vector<fp>&) -> GateMatrix { return {0, {0, -1}, {0, 1}, 0}; }},
        {qc::Z,   [](const std::vector<fp>&) -> GateMatrix { return {1, 0, 0, -1}; }},
        {qc::S,   [](const std::vector<fp>&) -> GateMatrix { return {1, 0, 0, {0, 1}}; }},
        {qc::Sdg, [](const std::vector<fp>&) -> GateMatrix { return {1, 0, 0, {0, -1}}; }},
        {qc::T,   [](const std::vector<fp>&) -> GateMatrix { return {1, 0, 0, {SQRT2_2, SQRT2_2}}; }},
        {qc::Tdg, [](const std::vector<fp>&) -> GateMatrix { return {1, 0, 0, {SQRT2_2, -SQRT2_2}}; }},
        {qc::SX,  [](const std::vector<fp>&) -> GateMatrix { return {std::complex{0.5, 0.5}, std::complex{0.5, -0.5}, std::complex{0.5, -0.5}, std::complex{0.5, 0.5}}; }},
        {qc::SXdg,[](const std::vector<fp>&) -> GateMatrix { return {std::complex{0.5, -0.5}, std::complex{0.5, 0.5}, std::complex{0.5, 0.5}, std::complex{0.5, -0.5}}; }},
        {qc::V,   [](const std::vector<fp>&) -> GateMatrix { return {SQRT2_2, {0., -SQRT2_2}, {0., -SQRT2_2}, SQRT2_2}; }},
        {qc::Vdg, [](const std::vector<fp>&) -> GateMatrix { return {SQRT2_2, {0., SQRT2_2}, {0., SQRT2_2}, SQRT2_2}; }},
        {qc::U,   [](const std::vector<fp>& params){ return uMat(params[0], params[1], params[2]); }},
        {qc::U2,  [](const std::vector<fp>& params){ return u2Mat(params[0], params[1]); }},
        {qc::P,   [](const std::vector<fp>& params){ return pMat(params[0]); }},
        {qc::RX,  [](const std::vector<fp>& params){ return rxMat(params[0]); }},
        {qc::RY,  [](const std::vector<fp>& params){ return ryMat(params[0]); }},
        {qc::RZ,  [](const std::vector<fp>& params){ return rzMat(params[0]); }},
        // clang-format on
    }};

const std::unordered_map<qc::OpType,
                         TwoQubitGateMatrix (*const)(const std::vector<fp>&)>
    TWO_GATE_MATS_GENERATORS{
        // clang-format off
        {qc::SWAP,      [](const std::vector<fp>&) -> TwoQubitGateMatrix {return {{ {1, 0, 0, 0}, {0, 0, 1, 0}, {0, 1, 0, 0}, {0, 0, 0, 1}}}; }},
        {qc::iSWAP,     [](const std::vector<fp>&) -> TwoQubitGateMatrix {return {{ {1, 0, 0, 0}, {0, 0, {0, 1}, 0}, {0, {0, 1}, 0, 0}, {0, 0, 0, 1}}}; }},
        {qc::iSWAPdg,   [](const std::vector<fp>&) -> TwoQubitGateMatrix {return {{ {1, 0, 0, 0}, {0, 0, {0, -1},0}, {0, {0, -1}, 0, 0}, {0, 0, 0, 1}}}; }},
        {qc::ECR,       [](const std::vector<fp>&) -> TwoQubitGateMatrix {return {{ {0, 0, SQRT2_2, {0, SQRT2_2}}, {0, 0, {0, SQRT2_2}, SQRT2_2}, {SQRT2_2, {0, -SQRT2_2}, 0, 0}, {std::complex{0., -SQRT2_2}, SQRT2_2, 0, 0}}}; }},
        {qc::DCX,       [](const std::vector<fp>&) -> TwoQubitGateMatrix {return {{{1, 0, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}, {0, 1, 0, 0}}}; }},
        {qc::Peres,     [](const std::vector<fp>&) -> TwoQubitGateMatrix {return {{{0, 0, 0, 1}, {0, 0, 1, 0}, {1, 0, 0, 0}, {0, 1, 0, 0}}}; }},
        {qc::Peresdg,   [](const std::vector<fp>&) -> TwoQubitGateMatrix {return {{ {0, 0, 1, 0}, {0, 0, 0, 1}, {0, 1, 0, 0}, {1, 0, 0, 0}}}; }},
        {qc::RXX,       [](const std::vector<fp>& params){ return rxxMat(params[0]); }},
        {qc::RYY,       [](const std::vector<fp>& params){ return ryyMat(params[0]); }},
        {qc::RZZ,       [](const std::vector<fp>& params){ return rzzMat(params[0]); }},
        {qc::RZX,       [](const std::vector<fp>& params){ return rzxMat(params[0]); }},
        {qc::XXminusYY, [](const std::vector<fp>& params){ return xxMinusYYMat(params[0], params[1]); }},
        {qc::XXplusYY,  [](const std::vector<fp>& params){ return xxPlusYYMat(params[0], params[1]); }},
        // clang-format on
    };

} // namespace dd
