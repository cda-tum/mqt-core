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

#include <algorithm>
#include <array>
#include <cstddef>
#include <iterator>
#include <ostream>
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
  /// @details This is a two-dimensional vector, where the first dimension
  /// represents the different (sets of) atoms the operation is applied to, and
  /// the second dimension represents the individual atoms in each set.
  /// For a one-qubit operation, the second dimension will always have size 1.
  /// For a two-qubit operation, e.g., CZ, the second dimension will have
  /// size 2.
  /// @note It is a variable length vector as the NAComputation might support,
  /// e.g., MCZ gates (multiple controlled Z gates) in the future.
  std::vector<std::vector<const Atom*>> atoms_;

  /// Creates a new local operation with the given atoms and parameters.
  /// @param atoms The atoms the operation is applied to.
  /// @param params The parameters of the operation.
  LocalOp(const std::vector<const Atom*>& atoms, std::vector<qc::fp> params);

  /// Creates a new local operation with the given atoms and parameters.
  /// @param atoms The atoms the operation is applied to.
  /// @param params The parameters of the operation.
  template <size_t N>
  LocalOp(const std::vector<std::array<const Atom*, N>>& atoms,
          std::vector<qc::fp> params);

public:
  LocalOp() = delete;

  /// Returns the atoms the operation is applied to.
  [[nodiscard]] auto getAtoms() const -> auto& { return atoms_; }

  /// Returns the parameters of the operation.
  [[nodiscard]] auto getParams() const -> auto& { return params_; }

  /// Returns a string representation of the operation.
  [[nodiscard]] auto toString() const -> std::string override;

private:
  /// Print the parameters of the operation to the output stream.
  static auto printParams(const std::vector<qc::fp>& params, std::ostream& os)
      -> void;

  /// Print the atoms of the operation to the output stream.
  static auto printAtoms(const std::vector<const Atom*>& atoms,
                         std::ostream& os) -> void;
};
template <size_t N>
LocalOp::LocalOp(const std::vector<std::array<const Atom*, N>>& atoms,
                 std::vector<qc::fp> params)
    : params_(std::move(params)) {
  std::transform(atoms.cbegin(), atoms.cend(), std::back_inserter(atoms_),
                 [](const auto& a) {
                   return std::vector<const Atom*>(a.cbegin(), a.cend());
                 });
}
} // namespace na
