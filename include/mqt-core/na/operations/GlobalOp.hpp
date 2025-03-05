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
class GlobalOp : public Op {
protected:
  std::string name_;
  std::vector<qc::fp> params_;
  const Zone* zone_;
  GlobalOp(const Zone& zone, std::vector<qc::fp> params)
      : params_(std::move(params)), zone_(&zone) {}

public:
  GlobalOp() = delete;
  [[nodiscard]] auto getParams() const -> const decltype(params_)& {
    return params_;
  }
  [[nodiscard]] auto getZone() const -> const Zone& { return *zone_; }
  [[nodiscard]] auto toString() const -> std::string override;
};
} // namespace na
