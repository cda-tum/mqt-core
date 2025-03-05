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
class LocalOp : public Op {
protected:
  std::string name_;
  std::vector<qc::fp> params_;
  std::vector<const Atom*> atoms_;

  LocalOp(std::vector<const Atom*> atoms, std::vector<qc::fp> params)
      : params_(std::move(params)), atoms_(std::move(atoms)) {}

public:
  LocalOp() = delete;
  [[nodiscard]] auto getAtoms() const -> const decltype(atoms_)& {
    return atoms_;
  }
  [[nodiscard]] auto getParams() const -> const decltype(params_)& {
    return params_;
  }
  [[nodiscard]] auto toString() const -> std::string override;
};
} // namespace na
