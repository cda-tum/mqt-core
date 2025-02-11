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
#include "na/entities/Location.hpp"
#include "na/operations/ShuttlingOp.hpp"

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace na {

class MoveOp : public ShuttlingOp {
protected:
  std::vector<Location> targetLocations;

public:
  explicit MoveOp(std::vector<const Atom*> atoms,
                  std::vector<Location> targetLocations)
      : ShuttlingOp(std::move(atoms)),
        targetLocations(std::move(targetLocations)) {
    if (this->atoms.size() != this->targetLocations.size()) {
      throw std::invalid_argument(
          "Number of atoms and target locations must be equal.");
    }
  }
  [[nodiscard]] auto getTargetLocations()
      -> decltype(targetLocations)& override {
    return targetLocations;
  }
  [[nodiscard]] auto getTargetLocations() const -> const
      decltype(targetLocations)& override {
    return targetLocations;
  }
  [[nodiscard]] auto toString() const -> std::string override;
};
} // namespace na
