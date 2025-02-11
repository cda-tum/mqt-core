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
class Atom final {
  std::string name;

public:
  Atom() = default;
  explicit Atom(std::string name) : name(std::move(name)) {}
  Atom(const Atom& atom) = default;
  Atom(Atom&& atom) noexcept = default;
  Atom& operator=(const Atom& atom) = default;
  Atom& operator=(Atom&& atom) noexcept = default;
  ~Atom() = default;
  [[nodiscard]] auto getName() const -> std::string { return name; }
  [[nodiscard]] auto toString() const -> std::string { return name; }
  friend auto operator<<(std::ostream& os, const Atom& obj) -> std::ostream& {
    return os << obj.toString();
  }
  [[nodiscard]] auto operator==(const Atom& other) const -> bool {
    if (this == &other) {
      return true;
    }
    return name == other.name;
  }
  [[nodiscard]] auto operator!=(const Atom& other) const -> bool {
    return !(*this == other);
  }
};
} // namespace na
template <> struct std::hash<const na::Atom*> {
  [[nodiscard]] auto operator()(const na::Atom* atom) const noexcept
      -> std::size_t {
    return hash<std::string>{}(atom->getName());
  }
}; // namespace std
