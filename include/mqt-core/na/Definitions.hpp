//
// This file is part of the MQT QMAP library released under the MIT license.
// See README.md or go to https://github.com/cda-tum/qmap for more information.
//
#pragma once

#include <cmath>
#include <cstdint>
namespace na {
/**
 * @brief Class to store two-dimensional coordinates
 */
class Point {
public:
  std::int32_t x;

  std::int32_t y;
  Point(std::int32_t x, std::int32_t y) : x(x), y(y){};
  inline Point operator-(const Point&& p) {
    x -= p.x;
    y -= p.y;
    return *this;
  }
  inline Point operator+(const Point&& p) {
    x += p.x;
    y += p.y;
    return *this;
  }
  [[nodiscard]] auto length() const {
    return static_cast<std::uint32_t>(std::round(std::sqrt(x * x + y * y)));
  }
};
} // namespace na