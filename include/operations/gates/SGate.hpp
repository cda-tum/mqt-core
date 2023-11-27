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

  dd::GateMatrix getInverseGateMatrix() override { return sMat; }

private:
  dd::GateMatrix sMat{1, 0, 0, {0, 1}};
};
}  // namespace qc
