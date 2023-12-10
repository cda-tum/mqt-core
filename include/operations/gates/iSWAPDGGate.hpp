#pragma once

#include "GateMatrixInterface.hpp"
#include "dd/DDDefinitions.hpp"
#include "operations/OpType.hpp"
#include "operations/StandardOperation.hpp"

namespace qc {
template <typename MatrixType>
class iSWAPDGGate : public GateMatrixInterface<MatrixType>, StandardOperation {
  MatrixType getGateMatrix() override {
    if (std::is_same<MatrixType, dd::TwoQubitGateMatrix>::value) {
      return iswapdgMat;
    }

    throw std::runtime_error(
        "Unsupported type for template object iSWAPDGGate!");
  }

  MatrixType getInverseGateMatrix() override { return iswapMat; }

  bool isSingleTargetGate() override { return false; }

  bool isTwoTargetGate() override {
    return std::is_same<MatrixType, dd::TwoQubitGateMatrix>::value;
  }

  bool isThreeOrMoreTargetGate() override { return false; }

  void invert() override {
    if (type != OpType::iSWAPdg) {
      throw std::runtime_error(
          "Object iSWAPDGGate does not contain correct operation type!");
    }

    type = iSWAP;
  }

private:
  dd::TwoQubitGateMatrix iswapdgMat{
      {{1, 0, 0, 0}, {0, 0, {0, -1}, 0}, {0, {0, -1}, 0, 0}, {0, 0, 0, 1}}};
  dd::TwoQubitGateMatrix iswapMat{
      {{1, 0, 0, 0}, {0, 0, {0, 1}, 0}, {0, {0, 1}, 0, 0}, {0, 0, 0, 1}}};
};
} // namespace qc
