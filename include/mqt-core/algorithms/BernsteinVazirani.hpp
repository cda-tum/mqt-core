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
#include <ostream>
#include <string>

namespace qc {
class BernsteinVazirani : public QuantumComputation {
public:
  BitString s = 0;
  std::size_t bitwidth = 1;
  bool dynamic = false;
  std::string expected;

  explicit BernsteinVazirani(const BitString& hiddenString, bool dyn = false);
  explicit BernsteinVazirani(std::size_t nq, bool dyn = false);
  BernsteinVazirani(const BitString& hiddenString, std::size_t nq,
                    bool dyn = false);

  std::ostream& printStatistics(std::ostream& os) const override;

protected:
  void createCircuit();
};
} // namespace qc
