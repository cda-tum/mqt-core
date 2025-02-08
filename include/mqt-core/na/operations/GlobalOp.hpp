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

#include <utility>
#include <vector>

namespace na {
class GlobalOp : public Op {
protected:
  std::string name;
  std::vector<qc::fp> params;
  const Zone* zone;
  GlobalOp(std::vector<qc::fp> params, const Zone* zone)
      : params(std::move(params)), zone(zone) {}
  explicit GlobalOp(const Zone* zone) : GlobalOp({}, zone) {}

public:
  GlobalOp() = delete;
  [[nodiscard]] auto getParams() -> decltype(params)& { return params; }
  [[nodiscard]] auto getParams() const -> const decltype(params)& {
    return params;
  }
  [[nodiscard]] auto getZone() -> const Zone*& { return zone; }
  [[nodiscard]] auto getZone() const -> const Zone* { return zone; }
  [[nodiscard]] auto toString() const -> std::string override;
};
} // namespace na
