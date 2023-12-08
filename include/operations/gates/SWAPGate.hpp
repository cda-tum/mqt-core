#pragma once

#include "GateMatrixInterface.hpp"
#include "dd/DDDefinitions.hpp"
#include "operations/OpType.hpp"
#include "operations/StandardOperation.hpp"

namespace qc {
template <typename MatrixType>
class SWAPGate : public GateMatrixInterface<MatrixType>, StandardOperation {
  MatrixType getGateMatrix() override {
    if (std::is_same<MatrixType, dd::GateMatrix>::value) {
      return swapMat;
    }

    throw std::runtime_error("Unsupported type for template object SWAPGate!");
  }

  MatrixType getInverseGateMatrix() override { return getGateMatrix(); }

  bool isSingleTargetGate() override { return false; }

  bool isTwoTargetGate() override {
    return static_cast<bool>(
        std::is_same<MatrixType, dd::TwoQubitGateMatrix>::value);
  }

  bool isThreeOrMoreTargetGate() override { return false; }

  void invert() override {
    if (type != OpType::SWAP) {
      throw std::runtime_error(
          "Object SWAPGate does not contain correct operation type!");
    }

    // leave swapMat as it is since SWAP gate is self-inverting
  }

private:
  dd::TwoQubitGateMatrix swapMat{
      {{1, 0, 0, 0}, {0, 0, 1, 0}, {0, 1, 0, 0}, {0, 0, 0, 1}}};
};
} // namespace qc
