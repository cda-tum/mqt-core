#pragma once

#include "GateMatrixInterface.hpp"
#include "dd/DDDefinitions.hpp"
#include "operations/OpType.hpp"
#include "operations/StandardOperation.hpp"

namespace qc {
template <typename MatrixType>
class SXDGGate : public GateMatrixInterface<MatrixType>, StandardOperation {
  MatrixType getGateMatrix() override {
    if (std::is_same<MatrixType, dd::GateMatrix>::value) {
      return sxDgMat;
    }

    throw std::runtime_error("Unsupported type for template object SXDGGate!");
  }

  MatrixType getInverseGateMatrix() override { return sxMat; }

  bool isSingleTargetGate() override {
    return std::is_same<MatrixType, dd::GateMatrix>::value;
  }

  bool isTwoTargetGate() override { return false; }
  bool isThreeOrMoreTargetGate() override { return false; }

  void invert() override {
    if (type != OpType::SXdg) {
      throw std::runtime_error(
          "Object SXDGGate does not contain correct operation type!");
    }

    type = SX;
  }

private:
  dd::GateMatrix sxDgMat{
      std::complex{0.5, -0.5}, {0.5, 0.5}, {0.5, 0.5}, {0.5, -0.5}};
  dd::GateMatrix sxMat{
      std::complex{0.5, 0.5}, {0.5, -0.5}, {0.5, -0.5}, {0.5, 0.5}};
};
} // namespace qc
