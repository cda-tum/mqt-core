/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include <cmath>
#include <cstdint>
#include <iomanip>
#include <ios>
#include <ostream>
#include <sstream>
#include <string>
#include <utility>

namespace na {
/// Class to store two-dimensional coordinates
struct Location final {
  double x = 0;
  double y = 0;
  Location operator-(const Location& loc) const {
    return {x - loc.x, y - loc.y};
  }
  Location operator+(const Location& loc) const {
    return {x + loc.x, y + loc.y};
  }
  [[nodiscard]] auto length() const -> double { return std::hypot(x, y); }
  [[nodiscard]] auto toString() const -> std::string {
    std::stringstream ss;
    ss << std::setprecision(3) << std::fixed;
    ss << "(" << x << ", " << y << ")";
    return ss.str();
  }
  friend auto operator<<(std::ostream& os, const Location& obj)
      -> std::ostream& {
    return os << obj.toString();
  }
  [[nodiscard]] auto operator==(const Location& other) const -> bool {
    return x == other.x && y == other.y;
  }
  [[nodiscard]] auto operator!=(const Location& other) const -> bool {
    return !(*this == other);
  }
  [[nodiscard]] auto operator<(const Location& other) const -> bool {
    return x < other.x || (x == other.x && y < other.y);
  }
  [[nodiscard]] auto operator>(const Location& other) const -> bool {
    return other < *this;
  }
  [[nodiscard]] auto operator>=(const Location& other) const -> bool {
    return !(other < *this);
  }
  [[nodiscard]] auto operator<=(const Location& other) const -> bool {
    return *this >= other;
  }
  [[nodiscard]] auto getEuclideanDistance(const Location& loc) const -> double {
    return (*this - loc).length();
  }
  [[nodiscard]] auto getManhattanDistanceX(const Location& loc) const
      -> double {
    return std::abs(x - loc.x);
  }
  [[nodiscard]] auto getManhattanDistanceY(const Location& loc) const
      -> double {
    return std::abs(y - loc.y);
  }
};
} // namespace na
