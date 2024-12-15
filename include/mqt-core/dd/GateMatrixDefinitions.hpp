/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include "dd/DDDefinitions.hpp"
#include "ir/operations/OpType.hpp"

#include <vector>

namespace dd {

// Gate matrices
constexpr GateMatrix MEAS_ZERO_MAT{1, 0, 0, 0};
constexpr GateMatrix MEAS_ONE_MAT{0, 0, 0, 1};

GateMatrix opToSingleQubitGateMatrix(qc::OpType t,
                                     const std::vector<fp>& params = {});

TwoQubitGateMatrix opToTwoQubitGateMatrix(qc::OpType t,
                                          const std::vector<fp>& params = {});

} // namespace dd
