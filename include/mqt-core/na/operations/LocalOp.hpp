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

#include <utility>
#include <vector>

namespace na {
class LocalOp : public Op {
protected:
  std::string name;
  std::vector<qc::fp> params;
  std::vector<const Atom*> atoms;

  LocalOp(std::vector<qc::fp> params, std::vector<const Atom*> atoms)
      : params(std::move(params)), atoms(std::move(atoms)) {}
  explicit LocalOp(std::vector<const Atom*>& atoms)
      : LocalOp({}, std::move(atoms)) {}
  explicit LocalOp(std::vector<qc::fp> params, const Atom* atom)
      : LocalOp(std::move(params), std::vector{atom}) {}
  explicit LocalOp(const Atom* atom) : LocalOp({}, atom) {}

public:
  LocalOp() = delete;
  [[nodiscard]] auto getAtoms() -> decltype(atoms)& { return atoms; }
  [[nodiscard]] auto getAtoms() const -> const decltype(atoms)& {
    return atoms;
  }
  [[nodiscard]] auto getParams() -> decltype(params)& { return params; }
  [[nodiscard]] auto getParams() const -> const decltype(params)& {
    return params;
  }
  [[nodiscard]] auto toString() const -> std::string override;
};
} // namespace na
