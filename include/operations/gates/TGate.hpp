#pragma once

#include "GateMatrixInterface.hpp"
#include "dd/DDDefinitions.hpp"
#include "operations/OpType.hpp"
#include "operations/StandardOperation.hpp"

namespace qc {
template <typename MatrixType>
class TGate : public GateMatrixInterface<MatrixType>, StandardOperation {
  MatrixType getGateMatrix() override {
    if (std::is_same<MatrixType, dd::GateMatrix>::value) {
      return tMat;
    }

    throw std::runtime_error("Unsupported type for template object TGate!");
  }

  MatrixType getInverseGateMatrix() override { return tdgMat; }

  bool isSingleTargetGate() override {
    return std::is_same<MatrixType, dd::GateMatrix>::value;
  }

  bool isTwoTargetGate() override { return false; }
  bool isThreeOrMoreTargetGate() override { return false; }

  void invert() override {
    if (type != OpType::T) {
      throw std::runtime_error(
          "Object TGate does not contain correct operation type!");
    }

    type = Tdg;
  }

private:
  dd::GateMatrix tMat{1, 0, 0, {dd::SQRT2_2, dd::SQRT2_2}};
  dd::GateMatrix tdgMat{1, 0, 0, {dd::SQRT2_2, -dd::SQRT2_2}};
};
} // namespace qc
