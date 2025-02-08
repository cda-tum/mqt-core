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
#include <iomanip>
#include <ostream>
#include <sstream>
#include <string>

namespace na {
/// Class to store two-dimensional coordinates
struct Location final {
  double x;
  double y;
  Location(const double x, const double y) : x(x), y(y) {};
  Location(const Location& p) = default;
  ~Location() = default;
  Location& operator=(const Location& loc) = default;
  Location operator-(const Location& loc) const {
    return {x - loc.x, y - loc.y};
  }
  Location operator-(const Location&& loc) const {
    return {x - loc.x, y - loc.y};
  }
  Location operator+(const Location& loc) const {
    return {x + loc.x, y + loc.y};
  }
  Location operator+(const Location&& loc) const {
    return {x + loc.x, y + loc.y};
  }
  [[nodiscard]] auto length() const -> double {
    return std::sqrt((x * x) + (y * y));
  }
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
  [[maybe_unused]] [[nodiscard]] auto
  getEuclideanDistance(const Location& loc) const -> double {
    return (*this - loc).length();
  }
  [[maybe_unused]] [[nodiscard]] auto
  getManhattanDistanceX(const Location& loc) const -> double {
    return std::abs(x - loc.x);
  }
  [[maybe_unused]] [[nodiscard]] auto
  getManhattanDistanceY(const Location& loc) const -> double {
    return std::abs(y - loc.y);
  }
};
} // namespace na
