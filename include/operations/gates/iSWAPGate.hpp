#pragma once

#include "GateMatrixInterface.hpp"
#include "dd/DDDefinitions.hpp"
#include "operations/OpType.hpp"
#include "operations/StandardOperation.hpp"

namespace qc {
template <typename MatrixType>
class iSWAPGate : public GateMatrixInterface<MatrixType>, StandardOperation {
  MatrixType getGateMatrix() override {
    if (std::is_same<MatrixType, dd::TwoQubitGateMatrix>::value) {
      return iswapMat;
    }

    throw std::runtime_error("Unsupported type for template object iSWAPGate!");
  }

  MatrixType getInverseGateMatrix() override { return iswapdgMat; }

  bool isSingleTargetGate() override { return false; }

  bool isTwoTargetGate() override {
    return std::is_same<MatrixType, dd::TwoQubitGateMatrix>::value;
  }

  bool isThreeOrMoreTargetGate() override { return false; }

  void invert() override {
    if (type != OpType::iSWAP) {
      throw std::runtime_error(
          "Object iSWAPGate does not contain correct operation type!");
    }

    type = iSWAP;
  }

private:
  dd::TwoQubitGateMatrix iswapMat{
      {{1, 0, 0, 0}, {0, 0, {0, 1}, 0}, {0, {0, 1}, 0, 0}, {0, 0, 0, 1}}};
  dd::TwoQubitGateMatrix iswapdgMat{
      {{1, 0, 0, 0}, {0, 0, {0, -1}, 0}, {0, {0, -1}, 0, 0}, {0, 0, 0, 1}}};
};
} // namespace qc
