#pragma once

#include "GateMatrixInterface.hpp"
#include "dd/DDDefinitions.hpp"
#include "operations/OpType.hpp"
#include "operations/StandardOperation.hpp"

namespace qc {
template <typename MatrixType>
class HGate : public GateMatrixInterface<MatrixType>, StandardOperation {
  MatrixType getGateMatrix() override {
    if (std::is_same<MatrixType, dd::GateMatrix>::value) {
      return hMat;
    }

    throw std::runtime_error("Unsupported type for template object HGate!");
  }

  MatrixType getInverseGateMatrix() override { return getGateMatrix(); }

  bool isSingleTargetGate() override {
    return static_cast<bool>(std::is_same<MatrixType, dd::GateMatrix>::value);
  }

  bool isTwoTargetGate() override { return false; }
  bool isThreeOrMoreTargetGate() override { return false; }

  void invert() override {
    if (type != OpType::H) {
      throw std::runtime_error(
          "Object HGate does not contain correct operation type!");
    }

    // leave hMat as it is since H gate is self-inverting
  }

private:
  dd::GateMatrix hMat{dd::SQRT2_2, dd::SQRT2_2, dd::SQRT2_2, dd::SQRT2_2};
};
} // namespace qc
