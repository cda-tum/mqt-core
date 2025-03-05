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

class LoadOp final : public ShuttlingOp {
protected:
  std::optional<std::vector<Location>> targetLocations_ = std::nullopt;

public:
  LoadOp(std::vector<const Atom*> atoms, std::vector<Location> targetLocations)
      : ShuttlingOp(std::move(atoms)),
        targetLocations_(std::move(targetLocations)) {
    if (this->atoms.size() != this->targetLocations_->size()) {
      throw std::invalid_argument(
          "Number of atoms and target locations must be equal.");
    }
  }
  explicit LoadOp(std::vector<const Atom*> atoms)
      : ShuttlingOp(std::move(atoms)) {}
  LoadOp(const Atom& atom, const Location& targetLocation)
      : LoadOp({&atom}, {targetLocation}) {}
  explicit LoadOp(const Atom& atom) : LoadOp({&atom}) {}
  [[nodiscard]] auto hasTargetLocations() const -> bool {
    return targetLocations_.has_value();
  }
  [[nodiscard]] auto getTargetLocations() const
      -> const std::vector<Location>& override {
    if (!targetLocations_.has_value()) {
      throw std::logic_error("Operation has no target locations set.");
    }
    return *targetLocations_;
  }
  [[nodiscard]] auto toString() const -> std::string override;
};
} // namespace na
