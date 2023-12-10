#pragma once

#include "GateMatrixInterface.hpp"
#include "dd/DDDefinitions.hpp"
#include "operations/OpType.hpp"
#include "operations/StandardOperation.hpp"

namespace qc {
template <typename MatrixType>
class YGate : public GateMatrixInterface<MatrixType>, StandardOperation {
  MatrixType getGateMatrix() override {
    if (std::is_same<MatrixType, dd::GateMatrix>::value) {
      return yMat;
    }

    throw std::runtime_error("Unsupported type for template object YGate!");
  }

  MatrixType getInverseGateMatrix() override { return getGateMatrix(); }

  bool isSingleTargetGate() override {
    return std::is_same<MatrixType, dd::GateMatrix>::value;
  }

  bool isTwoTargetGate() override { return false; }
  bool isThreeOrMoreTargetGate() override { return false; }

  void invert() override {
    if (type != OpType::Y) {
      throw std::runtime_error(
          "Object YGate does not contain correct operation type!");
    }

    // leave yMat as it is since Y gate is self-inverting
  }

private:
  dd::GateMatrix yMat{0, {0, -1}, {0, 1}, 0};
};
} // namespace qc
