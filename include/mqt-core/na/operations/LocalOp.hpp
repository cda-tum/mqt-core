/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

/** @file
 * @brief Defines a class for representing local operations in the
 * NAComputation.
 */

#pragma once

#include "Definitions.hpp"
#include "na/entities/Atom.hpp"
#include "na/operations/Op.hpp"

#include <string>
#include <utility>
#include <vector>

namespace na {
/// Represents a local operation in the NAComputation.
/// @details A local operation is applied to individual atoms.
class LocalOp : public Op {
protected:
  /// The name of the operation.
  std::string name_;
  /// The parameters of the operation.
  std::vector<qc::fp> params_;
  /// The atoms the operation is applied to.
  std::vector<const Atom*> atoms_;

  /// Creates a new local operation with the given atoms and parameters.
  /// @param atoms The atoms the operation is applied to.
  /// @param params The parameters of the operation.
  LocalOp(std::vector<const Atom*> atoms, std::vector<qc::fp> params)
      : params_(std::move(params)), atoms_(std::move(atoms)) {}

public:
  LocalOp() = delete;

  /// Returns the atoms the operation is applied to.
  [[nodiscard]] auto getAtoms() const -> auto& { return atoms_; }

  /// Returns the parameters of the operation.
  [[nodiscard]] auto getParams() const -> auto& { return params_; }

  /// Returns a string representation of the operation.
  [[nodiscard]] auto toString() const -> std::string override;
};
} // namespace na
