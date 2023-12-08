#pragma once

#include "GateMatrixInterface.hpp"
#include "dd/DDDefinitions.hpp"
#include "operations/OpType.hpp"
#include "operations/StandardOperation.hpp"

namespace qc {
template <typename MatrixType>
class XGate : public GateMatrixInterface<MatrixType>, StandardOperation {
  MatrixType getGateMatrix() override {
    if (std::is_same<MatrixType, dd::GateMatrix>::value) {
      return xMat;
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
    if (type != OpType::X) {
      throw std::runtime_error(
          "Object XGate does not contain correct operation type!");
    }

    // leave xMat as it is since X gate is self-inverting
  }

private:
  dd::GateMatrix xMat{0, 1, 1, 0};
};
} // namespace qc
