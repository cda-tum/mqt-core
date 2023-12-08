#pragma once

#include "GateMatrixInterface.hpp"
#include "dd/DDDefinitions.hpp"
#include "operations/OpType.hpp"
#include "operations/StandardOperation.hpp"

namespace qc {
template <typename MatrixType>
class ZGate : public GateMatrixInterface<MatrixType>, StandardOperation {
  MatrixType getGateMatrix() override {
    if (std::is_same<MatrixType, dd::GateMatrix>::value) {
      return zMat;
    }

    throw std::runtime_error("Unsupported type for template object XGate!");
  }

  MatrixType getInverseGateMatrix() override { return getGateMatrix(); }

  bool isSingleTargetGate() override {
    return static_cast<bool>(std::is_same<MatrixType, dd::GateMatrix>::value);
  }

  bool isTwoTargetGate() override { return false; }
  bool isThreeOrMoreTargetGate() override { return false; }

  void invert() override {
    if (type != OpType::Z) {
      throw std::runtime_error(
          "Object ZGate does not contain correct operation type!");
    }

    // leave zMat as it is since Z gate is self-inverting
  }

private:
  dd::GateMatrix zMat{1, 0, 0, -1};
};
} // namespace qc
