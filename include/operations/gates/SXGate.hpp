#pragma once

#include "GateMatrixInterface.hpp"
#include "dd/DDDefinitions.hpp"
#include "operations/OpType.hpp"
#include "operations/StandardOperation.hpp"

namespace qc {
template <typename MatrixType>
class SXGate : public GateMatrixInterface<MatrixType>, StandardOperation {
  MatrixType getGateMatrix() override {
    if (std::is_same<MatrixType, dd::GateMatrix>::value) {
      return sxMat;
    }

    throw std::runtime_error("Unsupported type for template object SXGate!");
  }

  MatrixType getInverseGateMatrix() override { return sxDgMat; }

  bool isSingleTargetGate() override {
    return std::is_same<MatrixType, dd::GateMatrix>::value;
  }

  bool isTwoTargetGate() override { return false; }
  bool isThreeOrMoreTargetGate() override { return false; }

  void invert() override {
    if (type != OpType::SX) {
      throw std::runtime_error(
          "Object SXGate does not contain correct operation type!");
    }

    type = SXdg;
  }

private:
  dd::GateMatrix sxMat{
      std::complex{0.5, 0.5}, {0.5, -0.5}, {0.5, -0.5}, {0.5, 0.5}};
  dd::GateMatrix sxDgMat{
      std::complex{0.5, -0.5}, {0.5, 0.5}, {0.5, 0.5}, {0.5, -0.5}};
};
} // namespace qc
