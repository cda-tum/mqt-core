/*
 * This file is part of the JKQ DD Package which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum_dd/ for more information.
 */

#ifndef DD_PACKAGE_UTIL_H
#define DD_PACKAGE_UTIL_H

#include "DDpackage.h"

#include <cassert>
#include <string>

// X gate matrix
constexpr dd::Matrix2x2 Xmat = {{{0., 0.}, {1., 0.}, {1., 0.}, {0., 0.}}};
// Hadamard gate matrix
constexpr dd::Matrix2x2 Hmat = {{{dd::SQRT_2, 0.}, {dd::SQRT_2, 0.}, {dd::SQRT_2, 0.}, {-dd::SQRT_2, 0.}}};

#endif //DD_PACKAGE_UTIL_H
