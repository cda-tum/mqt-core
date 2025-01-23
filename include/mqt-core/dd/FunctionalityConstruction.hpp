/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include "dd/Package_fwd.hpp"
#include "ir/Permutation.hpp"
#include "ir/QuantumComputation.hpp"

#include <cstddef>
#include <stack>

namespace dd {
using namespace qc;

template <class Config>
MatrixDD buildFunctionality(const QuantumComputation& qc, Package<Config>& dd);

template <class Config>
MatrixDD buildFunctionalityRecursive(const QuantumComputation& qc,
                                     Package<Config>& dd);

template <class Config>
bool buildFunctionalityRecursive(const QuantumComputation& qc,
                                 std::size_t depth, std::size_t opIdx,
                                 std::stack<MatrixDD>& s,
                                 Permutation& permutation, Package<Config>& dd);

} // namespace dd
