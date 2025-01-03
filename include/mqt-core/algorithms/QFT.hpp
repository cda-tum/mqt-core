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
#include <ostream>

namespace qc {
class QFT : public QuantumComputation {
public:
  explicit QFT(std::size_t nq, bool includeMeas = true, bool dyn = false);

  std::ostream& printStatistics(std::ostream& os) const override;

  std::size_t precision{};
  bool includeMeasurements;
  bool dynamic;

protected:
  void createCircuit();
};
} // namespace qc
