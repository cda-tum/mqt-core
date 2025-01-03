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

namespace qc {
class WState : public QuantumComputation {
public:
  explicit WState(Qubit nq);
};
} // namespace qc
