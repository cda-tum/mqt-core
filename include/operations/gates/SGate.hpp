#pragma once

#include "GateMatrixInterface.hpp"
#include "dd/DDDefinitions.hpp"
#include "operations/OpType.hpp"
#include "operations/StandardOperation.hpp"

namespace qc {
class SGate : public GateMatrixInterface, StandardOperation {
  dd::GateMatrix getGateMatrix() override { return sMat; }

  bool isSingleTargetGate() override { return true; }
  bool isTwoTargetGate() override { return false; }
  bool isThreeOrMoreTargetGate() override { return false; }

  dd::GateMatrix getInverseGateMatrix() override { return sdgMat; }

  void invert() override {
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
