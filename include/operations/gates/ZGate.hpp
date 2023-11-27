#pragma once

#include "GateMatrixInterface.hpp"
#include "dd/DDDefinitions.hpp"
#include "operations/OpType.hpp"
#include "operations/StandardOperation.hpp"

namespace qc {
class XGate : public GateMatrixInterface, StandardOperation {
  dd::GateMatrix getGateMatrix() override { return zMat; }

  bool isSingleTargetGate() override { return true; }
  bool isTwoTargetGate() override { return false; }
  bool isThreeOrMoreTargetGate() override { return false; }

  dd::GateMatrix getInverseGateMatrix() override { return zMat; }

private:
  dd::GateMatrix zMat{1, 0, 0, -1};
};
}  // namespace qc
