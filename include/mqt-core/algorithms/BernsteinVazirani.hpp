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

[[nodiscard]] auto createBernsteinVazirani(const BitString& hiddenString)
    -> QuantumComputation;
[[nodiscard]] auto createBernsteinVazirani(Qubit nq, std::size_t seed = 0)
    -> QuantumComputation;
[[nodiscard]] auto createBernsteinVazirani(const BitString& hiddenString,
                                           Qubit nq) -> QuantumComputation;

[[nodiscard]] auto
createIterativeBernsteinVazirani(const BitString& hiddenString)
    -> QuantumComputation;
[[nodiscard]] auto createIterativeBernsteinVazirani(Qubit nq,
                                                    std::size_t seed = 0)
    -> QuantumComputation;
[[nodiscard]] auto
createIterativeBernsteinVazirani(const BitString& hiddenString, Qubit nq)
    -> QuantumComputation;
} // namespace qc
