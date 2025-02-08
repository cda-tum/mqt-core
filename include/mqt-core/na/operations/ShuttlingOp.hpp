/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include "na/entities/Atom.hpp"
#include "na/operations/Op.hpp"

#include <vector>
#include <utility>

namespace na {
class ShuttlingOp : public Op {
protected:
  std::vector<const Atom*> atoms;
  explicit ShuttlingOp(std::vector<const Atom*> atoms)
      : atoms(std::move(atoms)) {}

public:
  [[nodiscard]] auto getAtoms() -> decltype(atoms)& { return atoms; }
  [[nodiscard]] auto getAtoms() const -> const decltype(atoms)& {
    return atoms;
  }
  [[nodiscard]] virtual auto getTargetLocations() -> std::vector<Location>& = 0;
  [[nodiscard]] virtual auto getTargetLocations() const
      -> const std::vector<Location>& = 0;
};
} // namespace na
