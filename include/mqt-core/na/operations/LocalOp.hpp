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
#include "na/entities/Atom.hpp"
#include "na/operations/Op.hpp"

#include <string>
#include <utility>
#include <vector>

namespace na {
/// Represents a local operation in the NA computation.
/// @details A local operation is applied to individual atoms.
class LocalOp : public Op {
protected:
  std::string name_;
  std::vector<qc::fp> params_;
  std::vector<const Atom*> atoms_;
  /// Creates a new local operation with the given atoms and parameters.
  /// @param atoms The atoms the operation is applied to.
  /// @param params The parameters of the operation.
  LocalOp(std::vector<const Atom*> atoms, std::vector<qc::fp> params)
      : params_(std::move(params)), atoms_(std::move(atoms)) {}

public:
  LocalOp() = delete;
  /// Returns the atoms the operation is applied to.
  /// @return The atoms the operation is applied to.
  [[nodiscard]] auto getAtoms() const -> const decltype(atoms_)& {
    return atoms_;
  }
  /// Returns the parameters of the operation.
  /// @return The parameters of the operation.
  [[nodiscard]] auto getParams() const -> const decltype(params_)& {
    return params_;
  }
  /// Returns a string representation of the operation.
  /// @return A string representation of the operation.
  [[nodiscard]] auto toString() const -> std::string override;
};
} // namespace na
