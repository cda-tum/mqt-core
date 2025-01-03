/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include "ir/QuantumComputation.hpp"

#include <cstddef>

namespace qc {
class Entanglement : public QuantumComputation {
public:
  explicit Entanglement(std::size_t nq);
};
} // namespace qc
