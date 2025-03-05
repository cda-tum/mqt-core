/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include <ostream>
#include <string>
#include <utility>

namespace na {
class Zone final {
  std::string name_;

public:
  Zone() = default;
  explicit Zone(std::string name) : name_(std::move(name)) {}
  [[nodiscard]] auto getName() const -> std::string { return name_; }
  friend auto operator<<(std::ostream& os, const Zone& obj) -> std::ostream& {
    return os << obj.getName();
  }
  [[nodiscard]] auto operator==(const Zone& other) const -> bool {
    if (this == &other) {
      return true;
    }
    return name_ == other.name_;
  }
  [[nodiscard]] auto operator!=(const Zone& other) const -> bool {
    return !(*this == other);
  }
};
} // namespace na
