/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include "Definitions.hpp"
#include "ir/QuantumComputation.hpp"

#include <cstddef>

namespace qc {
auto appendGroverInitialization(QuantumComputation& qc) -> void;
auto appendGroverOracle(QuantumComputation& qc, const BitString& targetValue)
    -> void;
auto appendGroverDiffusion(QuantumComputation& qc) -> void;

[[nodiscard]] auto computeNumberOfIterations(Qubit nq) -> std::size_t;

[[nodiscard]] auto createGrover(Qubit nq, const BitString& targetValue)
    -> QuantumComputation;
[[nodiscard]] auto createGrover(Qubit nq, std::size_t seed = 0)
    -> QuantumComputation;
} // namespace qc
