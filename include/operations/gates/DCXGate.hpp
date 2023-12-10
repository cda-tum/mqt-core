#pragma once

#include "GateMatrixInterface.hpp"
#include "dd/DDDefinitions.hpp"
#include "operations/OpType.hpp"
#include "operations/StandardOperation.hpp"

namespace qc {
template <typename MatrixType>
class DCXGate : public GateMatrixInterface<MatrixType>, StandardOperation {
  MatrixType getGateMatrix() override {
    if (std::is_same<MatrixType, dd::TwoQubitGateMatrix>::value) {
      return dcxMat;
    }

    throw std::runtime_error("Unsupported type for template object DCXGate!");
  }

  MatrixType getInverseGateMatrix() override { return getGateMatrix(); }

  bool isSingleTargetGate() override { return false; }

  bool isTwoTargetGate() override {
    return std::is_same<MatrixType, dd::TwoQubitGateMatrix>::value;
  }

  bool isThreeOrMoreTargetGate() override { return false; }

  void invert() override {
    if (type != OpType::DCX) {
      throw std::runtime_error(
          "Object DCXGate does not contain correct operation type!");
    }

    // DCX is a self-inverting gate, nothing needs to be changed
  }

private:
  dd::TwoQubitGateMatrix dcxMat{
      {{1, 0, 0, 0}, {0, 0, 0, 1}, {0, 1, 0, 0}, {0, 0, 1, 0}}};
};
} // namespace qc
