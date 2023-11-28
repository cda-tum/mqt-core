#pragma once

#include "GateMatrixInterface.hpp"
#include "dd/DDDefinitions.hpp"
#include "operations/OpType.hpp"
#include "operations/StandardOperation.hpp"

namespace qc {
class YGate : public GateMatrixInterface, StandardOperation {
  dd::GateMatrix getGateMatrix() override { return yMat; }

  bool isSingleTargetGate() override { return true; }
  bool isTwoTargetGate() override { return false; }
  bool isThreeOrMoreTargetGate() override { return false; }

  dd::GateMatrix getInverseGateMatrix() override { return yMat; }

private:
  dd::GateMatrix yMat{0, {0, -1}, {0, 1}, 0};
};
} // namespace qc
