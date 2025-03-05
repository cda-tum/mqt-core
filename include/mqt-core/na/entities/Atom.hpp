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
class Atom final {
  std::string name_;

public:
  Atom() = default;
  explicit Atom(std::string name) : name_(std::move(name)) {}
  [[nodiscard]] auto getName() const -> std::string { return name_; }
  friend auto operator<<(std::ostream& os, const Atom& obj) -> std::ostream& {
    return os << obj.getName();
  }
  [[nodiscard]] auto operator==(const Atom& other) const -> bool {
    if (this == &other) {
      return true;
    }
    return name_ == other.name_;
  }
  [[nodiscard]] auto operator!=(const Atom& other) const -> bool {
    return !(*this == other);
  }
};
} // namespace na
