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

namespace na {
class Op {
public:
  virtual ~Op() = default;
  [[nodiscard]] virtual auto toString() const -> std::string = 0;
  friend auto operator<<(std::ostream& os, const Op& obj) -> std::ostream& {
    return os << obj.toString(); // Using toString() method
  }
  template <class T> [[nodiscard]] auto is() const -> bool {
    return dynamic_cast<const T*>(this) != nullptr;
  }
  template <class T> [[nodiscard]] auto as() -> T& {
    return dynamic_cast<T&>(*this);
  }
  template <class T> [[nodiscard]] auto as() const -> const T& {
    return dynamic_cast<const T&>(*this);
  }
};
} // namespace na
