#pragma once

#include "GateMatrixInterface.hpp"
#include "dd/DDDefinitions.hpp"
#include "operations/OpType.hpp"
#include "operations/StandardOperation.hpp"

namespace qc {
template <typename MatrixType>
class SDGGate : public GateMatrixInterface<MatrixType>, StandardOperation {
  MatrixType getGateMatrix() override {
    if (std::is_same<MatrixType, dd::GateMatrix>::value) {
      return sdgMat;
    }

    throw std::runtime_error("Unsupported type for template object SDGGate!");
  }

  MatrixType getInverseGateMatrix() override { return sdgMat; }

  bool isSingleTargetGate() override {
    return std::is_same<MatrixType, dd::GateMatrix>::value;
  }

  bool isTwoTargetGate() override { return false; }
  bool isThreeOrMoreTargetGate() override { return false; }

  void invert() override {
    if (type != OpType::Sdg) {
      throw std::runtime_error(
          "Object SDGGate does not contain correct operation type!");
    }

    type = S;

    // TODO: semantic problem: we would have to change this object from being an
    //  SDGGate to SGate.
  }

private:
  dd::GateMatrix sdgMat{1, 0, 0, {0, -1}};
  dd::GateMatrix sMat{1, 0, 0, {0, 1}};
};
} // namespace qc
