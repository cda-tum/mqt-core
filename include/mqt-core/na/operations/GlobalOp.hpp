/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/** @file
 * @brief Defines a class for representing global operations in the
 * NAComputation.
 */

#pragma once

#include "Definitions.hpp"
#include "LocalOp.hpp"
#include "na/entities/Atom.hpp"
#include "na/entities/Location.hpp"
#include "na/entities/Zone.hpp"
#include "na/operations/Op.hpp"

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace na {
/// Represents a global operation in the NAComputation.
/// @details Global operations are applied to entire zones instead of to
/// individual atoms.
class GlobalOp : public Op {
protected:
  /// The name of the operation.
  std::string name_;
  /// The parameters of the operation.
  std::vector<qc::fp> params_;
  /// The zones the operation is applied to.
  std::vector<const Zone*> zones_;

  /// Creates a new global operation in the given zone with the given
  /// parameters.
  /// @param zones The zones the operation is applied to.
  /// @param params The parameters of the operation.
  GlobalOp(const std::vector<const Zone*>& zones, std::vector<qc::fp> params)
      : params_(std::move(params)), zones_(zones) {}

  /// Creates a new global operation in the given zones with the given
  /// parameters.
  /// @param zone The zone the operation is applied to.
  /// @param params The parameters of the operation.
  GlobalOp(const Zone& zone, std::vector<qc::fp> params)
      : params_(std::move(params)), zones_({&zone}) {}

public:
  GlobalOp() = delete;

  /// Returns the parameters of the operation.
  [[nodiscard]] auto getParams() const -> auto& { return params_; }

  /// Returns the zone the operation is applied to.
  [[nodiscard]] auto getZones() const -> auto& { return zones_; }

  /// Returns a string representation of the operation.
  [[nodiscard]] auto toString() const -> std::string override;

  /// Returns a local representation of the operation.
  /// @param atomsLocations The locations of the atoms.
  /// @param rydbergRadius The range of the Rydberg interaction.
  [[nodiscard]] virtual auto
  toLocal(const std::unordered_map<const Atom*, Location>& atomsLocations,
          double rydbergRadius) const -> std::unique_ptr<LocalOp> = 0;

private:
  /// Prints the parameters of the operation.
  static auto printParams(const std::vector<qc::fp>& params, std::ostream& os)
      -> void;
};
} // namespace na
