#pragma once

#include "GateMatrixInterface.hpp"
#include "dd/DDDefinitions.hpp"
#include "operations/OpType.hpp"
#include "operations/StandardOperation.hpp"

namespace qc {
template <typename MatrixType>
class SGate : public GateMatrixInterface<MatrixType>, StandardOperation {
  MatrixType getGateMatrix() override {
    if (std::is_same<MatrixType, dd::GateMatrix>::value) {
      return sMat;
    }

    throw std::runtime_error("Unsupported type for template object SGate!");
  }

  MatrixType getInverseGateMatrix() override { return sdgMat; }

  bool isSingleTargetGate() override {
    return static_cast<bool>(std::is_same<MatrixType, dd::GateMatrix>::value);
  }

  bool isTwoTargetGate() override { return false; }
  bool isThreeOrMoreTargetGate() override { return false; }

  void invert() override {
    if (type != OpType::S) {
      throw std::runtime_error(
          "Object SGate does not contain correct operation type!");
    }

    type = Sdg;

    // TODO: in theory here arises a problem, because we would have to
    //  change this object from being an SGate to SDGGate. This issue would
    //  happen to all kind of gates which are not self-inverting or are not
    //  parameterised.
    // A solution might be to have invert() return a pointer, but this might
    // force us to free the original gate after inverting it.
  }

private:
  dd::GateMatrix sMat{1, 0, 0, {0, 1}};
  dd::GateMatrix sdgMat{1, 0, 0, {0, -1}};
};
} // namespace qc
