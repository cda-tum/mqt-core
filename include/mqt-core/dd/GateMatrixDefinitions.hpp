/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include "dd/DDDefinitions.hpp"
#include "ir/operations/OpType.hpp"

#include <cmath>
#include <complex>
#include <unordered_map>
#include <vector>

namespace dd {

// Gate matrices
constexpr GateMatrix MEAS_ZERO_MAT{1, 0, 0, 0};
constexpr GateMatrix MEAS_ONE_MAT{0, 0, 0, 1};

extern const std::unordered_map<qc::OpType,
                                GateMatrix (*const)(const std::vector<fp>&)>
    MATS_GENERATORS;

extern const std::unordered_map<qc::OpType, TwoQubitGateMatrix (*const)(
                                                const std::vector<fp>&)>
    TWO_GATE_MATS_GENERATORS;

inline GateMatrix uMat(const fp lambda, const fp phi, const fp theta) {
  return GateMatrix{{{std::cos(theta / 2.), 0.},
                     {-std::cos(lambda) * std::sin(theta / 2.),
                      -std::sin(lambda) * std::sin(theta / 2.)},
                     {std::cos(phi) * std::sin(theta / 2.),
                      std::sin(phi) * std::sin(theta / 2.)},
                     {std::cos(lambda + phi) * std::cos(theta / 2.),
                      std::sin(lambda + phi) * std::cos(theta / 2.)}}};
}

inline GateMatrix u2Mat(const fp lambda, const fp phi) {
  return GateMatrix{
      SQRT2_2,
      {-std::cos(lambda) * SQRT2_2, -std::sin(lambda) * SQRT2_2},
      {std::cos(phi) * SQRT2_2, std::sin(phi) * SQRT2_2},
      {std::cos(lambda + phi) * SQRT2_2, std::sin(lambda + phi) * SQRT2_2}};
}

inline GateMatrix pMat(const fp lambda) {
  return GateMatrix{1, 0, 0, {std::cos(lambda), std::sin(lambda)}};
}

inline GateMatrix rxMat(const fp lambda) {
  return GateMatrix{{{std::cos(lambda / 2.), 0.},
                     {0., -std::sin(lambda / 2.)},
                     {0., -std::sin(lambda / 2.)},
                     {std::cos(lambda / 2.), 0.}}};
}

inline GateMatrix ryMat(const fp lambda) {
  return GateMatrix{{{std::cos(lambda / 2.), 0.},
                     {-std::sin(lambda / 2.), 0.},
                     {std::sin(lambda / 2.), 0.},
                     {std::cos(lambda / 2.), 0.}}};
}

inline GateMatrix rzMat(const fp lambda) {
  return GateMatrix{{{std::cos(lambda / 2.), -std::sin(lambda / 2.)},
                     0,
                     0,
                     {std::cos(lambda / 2.), std::sin(lambda / 2.)}}};
}

inline TwoQubitGateMatrix rxxMat(const fp theta) {
  const auto cosTheta = std::cos(theta / 2.);
  const auto sinTheta = std::sin(theta / 2.);

  return TwoQubitGateMatrix{{{cosTheta, 0, 0, {0., -sinTheta}},
                             {0, cosTheta, {0., -sinTheta}, 0},
                             {0, {0., -sinTheta}, cosTheta, 0},
                             {std::complex{0., -sinTheta}, 0, 0, cosTheta}}};
}

inline TwoQubitGateMatrix ryyMat(const fp theta) {
  const auto cosTheta = std::cos(theta / 2.);
  const auto sinTheta = std::sin(theta / 2.);

  return TwoQubitGateMatrix{{{cosTheta, 0, 0, {0., sinTheta}},
                             {0, cosTheta, {0., -sinTheta}, 0},
                             {0, {0., -sinTheta}, cosTheta, 0},
                             {std::complex{0., sinTheta}, 0, 0, cosTheta}}};
}

inline TwoQubitGateMatrix rzzMat(const fp theta) {
  const auto cosTheta = std::cos(theta / 2.);
  const auto sinTheta = std::sin(theta / 2.);

  return TwoQubitGateMatrix{{{std::complex{cosTheta, -sinTheta}, 0, 0, 0},
                             {0, {cosTheta, sinTheta}, 0, 0},
                             {0, 0, {cosTheta, sinTheta}, 0},
                             {0, 0, 0, {cosTheta, -sinTheta}}}};
}

inline TwoQubitGateMatrix rzxMat(const fp theta) {
  const auto cosTheta = std::cos(theta / 2.);
  const auto sinTheta = std::sin(theta / 2.);

  return TwoQubitGateMatrix{{{cosTheta, {0., -sinTheta}, 0, 0},
                             {std::complex{0., -sinTheta}, cosTheta, 0, 0},
                             {0, 0, cosTheta, {0., sinTheta}},
                             {0, 0, {0., sinTheta}, cosTheta}}};
}

inline TwoQubitGateMatrix xxMinusYYMat(const fp theta, const fp beta = 0.) {
  const auto cosTheta = std::cos(theta / 2.);
  const auto sinTheta = std::sin(theta / 2.);
  const auto cosBeta = std::cos(beta);
  const auto sinBeta = std::sin(beta);

  return TwoQubitGateMatrix{
      {{cosTheta, 0, 0, {-sinBeta * sinTheta, -cosBeta * sinTheta}},
       {0, 1, 0, 0},
       {0, 0, 1, 0},
       {std::complex{sinBeta * sinTheta, -cosBeta * sinTheta}, 0, 0,
        cosTheta}}};
}

inline TwoQubitGateMatrix xxPlusYYMat(const fp theta, const fp beta = 0.) {
  const auto cosTheta = std::cos(theta / 2.);
  const auto sinTheta = std::sin(theta / 2.);
  const auto cosBeta = std::cos(beta);
  const auto sinBeta = std::sin(beta);

  return TwoQubitGateMatrix{
      {{1, 0, 0, 0},
       {0, cosTheta, {sinBeta * sinTheta, -cosBeta * sinTheta}, 0},
       {0, {-sinBeta * sinTheta, -cosBeta * sinTheta}, cosTheta, 0},
       {0, 0, 0, 1}}};
}

inline GateMatrix opToSingleGateMatrix(qc::OpType t,
                                       const std::vector<fp>& params = {}) {
  return MATS_GENERATORS.at(t)(params);
}

inline TwoQubitGateMatrix
opToTwoQubitGateMatrix(qc::OpType t, const std::vector<fp>& params = {}) {
  return TWO_GATE_MATS_GENERATORS.at(t)(params);
}

} // namespace dd
