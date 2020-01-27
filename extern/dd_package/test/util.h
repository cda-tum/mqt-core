//
// Created by stefan on 04.12.19.
//

#ifndef DD_PACKAGE_UTIL_H
#define DD_PACKAGE_UTIL_H

#include "DDpackage.h"
#include <string>
#include <cassert>

// X gate matrix
constexpr dd::Matrix2x2 Xmat = {{{ 0, 0 }, { 1, 0 } }, {{ 1, 0 }, { 0, 0 } }};
// Hadamard gate matrix
constexpr dd::Matrix2x2 Hmat = {{{ dd::SQRT_2, 0 }, { dd::SQRT_2,  0 }},
                                {{ dd::SQRT_2, 0 }, { -dd::SQRT_2, 0 }}};

#endif //DD_PACKAGE_UTIL_H
