/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include <string>
#include <utility>

namespace na {
class Zone final {
  std::string name;

public:
  Zone() = default;
  explicit Zone(std::string name) : name(std::move(name)) {}
  Zone(const Zone& zone) = default;
  Zone(Zone&& zone) noexcept = default;
  Zone& operator=(const Zone& zone) = default;
  Zone& operator=(Zone&& zone) noexcept = default;
  ~Zone() = default;
  [[nodiscard]] auto getName() const -> std::string { return name; }
  [[nodiscard]] auto toString() const -> std::string { return name; }
  friend auto operator<<(std::ostream& os, const Zone& obj) -> std::ostream& {
    return os << obj.toString();
  }
  [[nodiscard]] auto operator==(const Zone& other) const -> bool {
    if (this == &other) {
      return true;
    }
    return name == other.name;
  }
  [[nodiscard]] auto operator!=(const Zone& other) const -> bool {
    return !(*this == other);
  }
};
} // namespace na
