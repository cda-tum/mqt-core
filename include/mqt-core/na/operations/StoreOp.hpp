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

#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace na {

class StoreOp final : public ShuttlingOp {
protected:
  std::optional<std::vector<Location>> targetLocations = std::nullopt;

public:
  StoreOp(std::vector<const Atom*> atoms, std::vector<Location> targetLocations)
      : ShuttlingOp(std::move(atoms)),
        targetLocations(std::move(targetLocations)) {
    if (this->atoms.size() != this->targetLocations->size()) {
      throw std::invalid_argument(
          "Number of atoms and target locations must be equal.");
    }
  }
  explicit StoreOp(std::vector<const Atom*> atoms)
      : ShuttlingOp(std::move(atoms)) {}
  StoreOp(const Atom& atom, const Location& targetLocation)
      : StoreOp({&atom}, {targetLocation}) {}
  explicit StoreOp(const Atom& atom) : StoreOp({&atom}) {}
  [[nodiscard]] auto hasTargetLocations() const -> bool {
    return targetLocations.has_value();
  }
  [[nodiscard]] auto getTargetLocations() const
      -> const std::vector<Location>& override {
    if (!targetLocations.has_value()) {
      throw std::logic_error("Operation has no target locations set.");
    }
    return *targetLocations;
  }
  [[nodiscard]] auto toString() const -> std::string override;
};
} // namespace na
