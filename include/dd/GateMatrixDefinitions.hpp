#pragma once

#include "dd/DDDefinitions.hpp"

#include <array>
#include <cmath>
#include <complex>

namespace dd {
// Complex constants
// NOLINTBEGIN(readability-identifier-naming) As these constants are used by
// other projects, we keep the naming
constexpr std::complex<fp> complex_one = {1., 0.};
constexpr std::complex<fp> complex_mone = {-1., 0.};
constexpr std::complex<fp> complex_zero = {0., 0.};
constexpr std::complex<fp> complex_i = {0., 1.};
constexpr std::complex<fp> complex_mi = {0., -1.};
constexpr std::complex<fp> complex_SQRT2_2 = {SQRT2_2, 0.};
constexpr std::complex<fp> complex_mSQRT2_2 = {-SQRT2_2, 0.};
constexpr std::complex<fp> complex_iSQRT2_2 = {0., SQRT2_2};
constexpr std::complex<fp> complex_miSQRT2_2 = {0., -SQRT2_2};
constexpr std::complex<fp> complex_1plusi = {SQRT2_2, SQRT2_2};
constexpr std::complex<fp> complex_1minusi = {SQRT2_2, -SQRT2_2};
constexpr std::complex<fp> complex_1plusi_2 = {0.5, 0.5};
constexpr std::complex<fp> complex_1minusi_2 = {0.5, -0.5};

// Gate matrices
using GateMatrix = std::array<std::complex<fp>, NEDGE>;
constexpr GateMatrix Imat{complex_one, complex_zero, complex_zero, complex_one};
constexpr GateMatrix Hmat{complex_SQRT2_2, complex_SQRT2_2, complex_SQRT2_2,
                          complex_mSQRT2_2};
constexpr GateMatrix Xmat{complex_zero, complex_one, complex_one, complex_zero};
constexpr GateMatrix Ymat{complex_zero, complex_mi, complex_i, complex_zero};
constexpr GateMatrix Zmat{complex_one, complex_zero, complex_zero,
                          complex_mone};
constexpr GateMatrix Smat{complex_one, complex_zero, complex_zero, complex_i};
constexpr GateMatrix Sdagmat{complex_one, complex_zero, complex_zero,
                             complex_mi};
constexpr GateMatrix Tmat{complex_one, complex_zero, complex_zero,
                          complex_1plusi};
constexpr GateMatrix Tdagmat{complex_one, complex_zero, complex_zero,
                             complex_1minusi};
constexpr GateMatrix SXmat{complex_1plusi_2, complex_1minusi_2,
                           complex_1minusi_2, complex_1plusi_2};
constexpr GateMatrix SXdagmat{complex_1minusi_2, complex_1plusi_2,
                              complex_1plusi_2, complex_1minusi_2};
constexpr GateMatrix Vmat{complex_SQRT2_2, complex_miSQRT2_2, complex_miSQRT2_2,
                          complex_SQRT2_2};
constexpr GateMatrix Vdagmat{complex_SQRT2_2, complex_iSQRT2_2,
                             complex_iSQRT2_2, complex_SQRT2_2};

inline GateMatrix U3mat(fp lambda, fp phi, fp theta) {
  return GateMatrix{{{std::cos(theta / 2.), 0.},
                     {-std::cos(lambda) * std::sin(theta / 2.),
                      -std::sin(lambda) * std::sin(theta / 2.)},
                     {std::cos(phi) * std::sin(theta / 2.),
                      std::sin(phi) * std::sin(theta / 2.)},
                     {std::cos(lambda + phi) * std::cos(theta / 2.),
                      std::sin(lambda + phi) * std::cos(theta / 2.)}}};
}

inline GateMatrix U2mat(fp lambda, fp phi) {
  return GateMatrix{
      complex_SQRT2_2,
      {-std::cos(lambda) * SQRT2_2, -std::sin(lambda) * SQRT2_2},
      {std::cos(phi) * SQRT2_2, std::sin(phi) * SQRT2_2},
      {std::cos(lambda + phi) * SQRT2_2, std::sin(lambda + phi) * SQRT2_2}};
}

inline GateMatrix Phasemat(fp lambda) {
  return GateMatrix{complex_one,
                    complex_zero,
                    complex_zero,
                    {std::cos(lambda), std::sin(lambda)}};
}

inline GateMatrix RXmat(fp lambda) {
  return GateMatrix{{{std::cos(lambda / 2.), 0.},
                     {0., -std::sin(lambda / 2.)},
                     {0., -std::sin(lambda / 2.)},
                     {std::cos(lambda / 2.), 0.}}};
}

inline GateMatrix RYmat(fp lambda) {
  return GateMatrix{{{std::cos(lambda / 2.), 0.},
                     {-std::sin(lambda / 2.), 0.},
                     {std::sin(lambda / 2.), 0.},
                     {std::cos(lambda / 2.), 0.}}};
}

inline GateMatrix RZmat(fp lambda) {
  return GateMatrix{{{std::cos(lambda / 2.), -std::sin(lambda / 2.)},
                     complex_zero,
                     complex_zero,
                     {std::cos(lambda / 2.), std::sin(lambda / 2.)}}};
}

using TwoQubitGateMatrix =
    std::array<std::array<std::complex<fp>, NEDGE>, NEDGE>;
constexpr TwoQubitGateMatrix CXmat{
    {{complex_one, complex_zero, complex_zero, complex_zero},
     {complex_zero, complex_one, complex_zero, complex_zero},
     {complex_zero, complex_zero, complex_zero, complex_one},
     {complex_zero, complex_zero, complex_one, complex_zero}}};

constexpr TwoQubitGateMatrix CZmat{
    {{complex_one, complex_zero, complex_zero, complex_zero},
     {complex_zero, complex_one, complex_zero, complex_zero},
     {complex_zero, complex_zero, complex_one, complex_zero},
     {complex_zero, complex_zero, complex_zero, complex_mone}}};

constexpr TwoQubitGateMatrix SWAPmat{
    {{complex_one, complex_zero, complex_zero, complex_zero},
     {complex_zero, complex_zero, complex_one, complex_zero},
     {complex_zero, complex_one, complex_zero, complex_zero},
     {complex_zero, complex_zero, complex_zero, complex_one}}};

constexpr TwoQubitGateMatrix iSWAPmat{
    {{complex_one, complex_zero, complex_zero, complex_zero},
     {complex_zero, complex_zero, complex_i, complex_zero},
     {complex_zero, complex_i, complex_zero, complex_zero},
     {complex_zero, complex_zero, complex_zero, complex_one}}};

constexpr TwoQubitGateMatrix iSWAPinvmat{
    {{complex_one, complex_zero, complex_zero, complex_zero},
     {complex_zero, complex_zero, complex_mi, complex_zero},
     {complex_zero, complex_mi, complex_zero, complex_zero},
     {complex_zero, complex_zero, complex_zero, complex_one}}};

constexpr TwoQubitGateMatrix ECRmat{
    {{complex_zero, complex_zero, complex_SQRT2_2, complex_iSQRT2_2},
     {complex_zero, complex_zero, complex_iSQRT2_2, complex_SQRT2_2},
     {complex_SQRT2_2, complex_miSQRT2_2, complex_zero, complex_zero},
     {complex_miSQRT2_2, complex_SQRT2_2, complex_zero, complex_zero}}};

constexpr TwoQubitGateMatrix DCXmat{
    {{complex_one, complex_zero, complex_zero, complex_zero},
     {complex_zero, complex_zero, complex_zero, complex_one},
     {complex_zero, complex_one, complex_zero, complex_zero},
     {complex_zero, complex_zero, complex_one, complex_zero}}};

inline TwoQubitGateMatrix RXXmat(const fp theta) {
  const auto cosTheta = std::cos(theta / 2.);
  const auto sinTheta = std::sin(theta / 2.);

  return TwoQubitGateMatrix{{{std::complex<fp>{cosTheta, 0.}, complex_zero,
                              complex_zero, std::complex<fp>{0., -sinTheta}},
                             {complex_zero, std::complex<fp>{cosTheta, 0.},
                              std::complex<fp>{0., -sinTheta}, complex_zero},
                             {complex_zero, std::complex<fp>{0., -sinTheta},
                              std::complex<fp>{cosTheta, 0.}, complex_zero},
                             {std::complex<fp>{0., -sinTheta}, complex_zero,
                              complex_zero, std::complex<fp>{cosTheta, 0.}}}};
}

inline TwoQubitGateMatrix RYYmat(const fp theta) {
  const auto cosTheta = std::cos(theta / 2.);
  const auto sinTheta = std::sin(theta / 2.);

  return TwoQubitGateMatrix{{{std::complex<fp>{cosTheta, 0.}, complex_zero,
                              complex_zero, std::complex<fp>{0., sinTheta}},
                             {complex_zero, std::complex<fp>{cosTheta, 0.},
                              std::complex<fp>{0., -sinTheta}, complex_zero},
                             {complex_zero, std::complex<fp>{0., -sinTheta},
                              std::complex<fp>{cosTheta, 0.}, complex_zero},
                             {std::complex<fp>{0., sinTheta}, complex_zero,
                              complex_zero, std::complex<fp>{cosTheta, 0.}}}};
}

inline TwoQubitGateMatrix RZZmat(const fp theta) {
  const auto cosTheta = std::cos(theta / 2.);
  const auto sinTheta = std::sin(theta / 2.);

  return TwoQubitGateMatrix{
      {{std::complex<fp>{cosTheta, -sinTheta}, complex_zero, complex_zero,
        complex_zero},
       {complex_zero, std::complex<fp>{cosTheta, sinTheta}, complex_zero,
        complex_zero},
       {complex_zero, complex_zero, std::complex<fp>{cosTheta, sinTheta},
        complex_zero},
       {complex_zero, complex_zero, complex_zero,
        std::complex<fp>{cosTheta, -sinTheta}}}};
}

inline TwoQubitGateMatrix RZXmat(const fp theta) {
  const auto cosTheta = std::cos(theta / 2.);
  const auto sinTheta = std::sin(theta / 2.);

  return TwoQubitGateMatrix{
      {{std::complex<fp>{cosTheta, 0.}, std::complex<fp>{0., -sinTheta},
        complex_zero, complex_zero},
       {std::complex<fp>{0., -sinTheta}, std::complex<fp>{cosTheta, 0.},
        complex_zero, complex_zero},
       {complex_zero, complex_zero, std::complex<fp>{cosTheta, 0.},
        std::complex<fp>{0., sinTheta}},
       {complex_zero, complex_zero, std::complex<fp>{0., sinTheta},
        std::complex<fp>{cosTheta, 0.}}}};
}

inline TwoQubitGateMatrix XXMinusYYmat(const fp theta, const fp beta = 0.) {
  const auto cosTheta = std::cos(theta / 2.);
  const auto sinTheta = std::sin(theta / 2.);
  const auto cosBeta = std::cos(beta);
  const auto sinBeta = std::sin(beta);

  return TwoQubitGateMatrix{
      {{std::complex<fp>{cosTheta, 0.}, complex_zero, complex_zero,
        std::complex<fp>{sinBeta * sinTheta, -cosBeta * sinTheta}},
       {complex_zero, complex_one, complex_zero, complex_zero},
       {complex_zero, complex_zero, complex_one, complex_zero},
       {std::complex<fp>{-sinBeta * sinTheta, -cosBeta * sinTheta},
        complex_zero, complex_zero, std::complex<fp>{cosTheta, 0.}}}};
}

inline TwoQubitGateMatrix XXPlusYYmat(const fp theta, const fp beta = 0.) {
  const auto cosTheta = std::cos(theta / 2.);
  const auto sinTheta = std::sin(theta / 2.);
  const auto cosBeta = std::cos(beta);
  const auto sinBeta = std::sin(beta);

  return TwoQubitGateMatrix{
      {{complex_one, complex_zero, complex_zero, complex_zero},
       {complex_zero, std::complex<fp>{cosTheta, 0.},
        std::complex<fp>{sinBeta * sinTheta, -cosBeta * sinTheta},
        complex_zero},
       {complex_zero,
        std::complex<fp>{-sinBeta * sinTheta, -cosBeta * sinTheta},
        std::complex<fp>{cosTheta, 0.}, complex_zero},
       {complex_zero, complex_zero, complex_zero, complex_one}}};
}
// NOLINTEND(readability-identifier-naming)
} // namespace dd
