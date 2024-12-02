/*
 * Copyright (c) 2024 Chair for Design Automation, TUM
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
#include <ostream>

namespace qc {
class QPE : public QuantumComputation {
public:
  fp lambda = 0.;
  std::size_t precision;
  bool iterative;

  explicit QPE(std::size_t nq, bool exact = true, bool iter = false);
  QPE(fp l, std::size_t prec, bool iter = false);

  std::ostream& printStatistics(std::ostream& os) const override;

protected:
  void createCircuit();
};
} // namespace qc
