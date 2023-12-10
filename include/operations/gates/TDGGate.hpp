#pragma once

#include "GateMatrixInterface.hpp"
#include "dd/DDDefinitions.hpp"
#include "operations/OpType.hpp"
#include "operations/StandardOperation.hpp"

namespace qc {
template <typename MatrixType>
class TDGGate : public GateMatrixInterface<MatrixType>, StandardOperation {
  MatrixType getGateMatrix() override {
    if (std::is_same<MatrixType, dd::GateMatrix>::value) {
      return tdgMat;
    }

    throw std::runtime_error("Unsupported type for template object TDGGate!");
  }

  MatrixType getInverseGateMatrix() override { return tMat; }

  bool isSingleTargetGate() override {
    return std::is_same<MatrixType, dd::GateMatrix>::value;
  }

  bool isTwoTargetGate() override { return false; }
  bool isThreeOrMoreTargetGate() override { return false; }

  void invert() override {
    if (type != OpType::Tdg) {
      throw std::runtime_error(
          "Object TDGGate does not contain correct operation type!");
    }

    type = T;
  }

private:
  dd::GateMatrix tdgMat{1, 0, 0, {dd::SQRT2_2, -dd::SQRT2_2}};
  dd::GateMatrix tMat{1, 0, 0, {dd::SQRT2_2, dd::SQRT2_2}};
};
} // namespace qc
