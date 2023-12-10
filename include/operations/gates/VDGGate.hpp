#pragma once

#include "GateMatrixInterface.hpp"
#include "dd/DDDefinitions.hpp"
#include "operations/OpType.hpp"
#include "operations/StandardOperation.hpp"

namespace qc {
template <typename MatrixType>
class VDGGate : public GateMatrixInterface<MatrixType>, StandardOperation {
  MatrixType getGateMatrix() override {
    if (std::is_same<MatrixType, dd::GateMatrix>::value) {
      return vdgMat;
    }

    throw std::runtime_error("Unsupported type for template object VDGGate!");
  }

  MatrixType getInverseGateMatrix() override { return vMat; }

  bool isSingleTargetGate() override {
    return std::is_same<MatrixType, dd::GateMatrix>::value;
  }

  bool isTwoTargetGate() override { return false; }
  bool isThreeOrMoreTargetGate() override { return false; }

  void invert() override {
    if (type != OpType::Vdg) {
      throw std::runtime_error(
          "Object VDGGate does not contain correct operation type!");
    }

    type = V;
  }

private:
  dd::GateMatrix vdgMat{
      dd::SQRT2_2, {0, dd::SQRT2_2}, {0, dd::SQRT2_2}, dd::SQRT2_2};
  dd::GateMatrix vMat{
      dd::SQRT2_2, {0, -dd::SQRT2_2}, {0, -dd::SQRT2_2}, dd::SQRT2_2};
};
} // namespace qc
