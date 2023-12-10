#pragma once

#include "GateMatrixInterface.hpp"
#include "dd/DDDefinitions.hpp"
#include "operations/OpType.hpp"
#include "operations/StandardOperation.hpp"

namespace qc {
template <typename MatrixType>
class ECRGate : public GateMatrixInterface<MatrixType>, StandardOperation {
  MatrixType getGateMatrix() override {
    if (std::is_same<MatrixType, dd::TwoQubitGateMatrix>::value) {
      return ecrMat;
    }

    throw std::runtime_error("Unsupported type for template object ECRGate!");
  }

  MatrixType getInverseGateMatrix() override { return getGateMatrix(); }

  bool isSingleTargetGate() override { return false; }

  bool isTwoTargetGate() override {
    return std::is_same<MatrixType, dd::TwoQubitGateMatrix>::value;
  }

  bool isThreeOrMoreTargetGate() override { return false; }

  void invert() override {
    if (type != OpType::ECR) {
      throw std::runtime_error(
          "Object ECRGate does not contain correct operation type!");
    }

    // ECR is a self-inverting gate, nothing needs to be changed
  }

private:
  dd::TwoQubitGateMatrix ecrMat{
      {{0, 0, dd::SQRT2_2, {0, dd::SQRT2_2}},
       {0, 0, {0, dd::SQRT2_2}, dd::SQRT2_2},
       {dd::SQRT2_2, {0, -dd::SQRT2_2}, 0, 0},
       {std::complex{0., -dd::SQRT2_2}, dd::SQRT2_2, 0, 0}}};
};
} // namespace qc