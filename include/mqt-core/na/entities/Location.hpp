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
#include <ostream>
#include <sstream>
#include <string>
#include <utility>

namespace na {
/// Class to store two-dimensional coordinates
struct Location final {
  double x = 0;
  double y = 0;
  Location(const float x, const float y)
      : x(static_cast<double>(x)), y(static_cast<double>(y)) {};
  Location(const double x, const double y) : x(x), y(y) {};
  Location(const std::int32_t x, const std::int32_t y)
      : x(static_cast<double>(x)), y(static_cast<double>(y)) {};
  Location(const std::int64_t x, const std::int64_t y)
      : x(static_cast<double>(x)), y(static_cast<double>(y)) {};
  Location(const std::uint32_t x, const std::uint32_t y)
      : x(static_cast<double>(x)), y(static_cast<double>(y)) {};
  Location(const std::uint64_t x, const std::uint64_t y)
      : x(static_cast<double>(x)), y(static_cast<double>(y)) {};
  Location(const std::size_t x, const std::size_t y)
      : x(static_cast<double>(x)), y(static_cast<double>(y)) {};
  explicit Location(const std::pair<float, float> p)
      : Location(p.first, p.second) {};
  explicit Location(const std::pair<double, double> p)
      : Location(p.first, p.second) {};
  explicit Location(const std::pair<std::int32_t, std::int32_t> p)
      : Location(p.first, p.second) {};
  explicit Location(const std::pair<std::int64_t, std::int64_t> p)
      : Location(p.first, p.second) {};
  explicit Location(const std::pair<std::uint32_t, std::uint32_t> p)
      : Location(p.first, p.second) {};
  explicit Location(const std::pair<std::uint64_t, std::uint64_t> p)
      : Location(p.first, p.second) {};
  explicit Location(const std::pair<std::size_t, std::size_t> p)
      : Location(p.first, p.second) {};
  Location() = default;
  Location(const Location& loc) = default;
  Location(Location&& loc) noexcept = default;
  Location& operator=(const Location& loc) = default;
  Location& operator=(Location&& loc) noexcept = default;
  ~Location() = default;
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
  [[nodiscard]] auto operator!=(const Location& other) const -> bool {
    return !(*this == other);
  }
  [[nodiscard]] auto operator<(const Location& other) const -> bool {
    return x < other.x || (x == other.x && y < other.y);
  }
  [[nodiscard]] auto operator>(const Location& other) const -> bool {
    return x > other.x || (x == other.x && y > other.y);
  }
  [[nodiscard]] auto operator<=(const Location& other) const -> bool {
    return x <= other.x || (x == other.x && y <= other.y);
  }
  [[nodiscard]] auto operator>=(const Location& other) const -> bool {
    return x >= other.x || (x == other.x && y >= other.y);
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
