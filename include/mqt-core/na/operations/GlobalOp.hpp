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
#include "na/entities/Zone.hpp"
#include "na/operations/Op.hpp"

#include <string>
#include <utility>
#include <vector>

namespace na {
/// Represents a global operation in the NA computation.
/// @details Global operations are applied to entire zones instead of to
/// individual atoms.
class GlobalOp : public Op {
protected:
  std::string name_;
  std::vector<qc::fp> params_;
  const Zone* zone_;
  /// Creates a new global operation in the given zone with the given
  /// parameters.
  /// @param zone The zone the operation is applied to.
  /// @param params The parameters of the operation.
  GlobalOp(const Zone& zone, std::vector<qc::fp> params)
      : params_(std::move(params)), zone_(&zone) {}

public:
  GlobalOp() = delete;
  /// Returns the parameters of the operation.
  /// @return The parameters of the operation.
  [[nodiscard]] auto getParams() const -> const decltype(params_)& {
    return params_;
  }
  /// Returns the zone the operation is applied to.
  /// @return The zone the operation is applied to.
  [[nodiscard]] auto getZone() const -> const Zone& { return *zone_; }
  /// Returns a string representation of the operation.
  /// @return A string representation of the operation.
  [[nodiscard]] auto toString() const -> std::string override;
};
} // namespace na
