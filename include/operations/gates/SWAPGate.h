#pragma once

#include "GateMatrixInterface.hpp"
#include "dd/DDDefinitions.hpp"
#include "operations/OpType.hpp"
#include "operations/StandardOperation.hpp"

namespace qc {
class SWAPGate : public GateMatrixInterface, StandardOperation {
  //  dd::TwoQubitGateMatrix getGateMatrix() override { return swapMat; }

  bool isSingleTargetGate() override { return false; }
  bool isTwoTargetGate() override { return true; }
  bool isThreeOrMoreTargetGate() override { return false; }

  //  dd::TwoQubitGateMatrix getInverseGateMatrix() override { return swapMat; }

private:
  dd::TwoQubitGateMatrix swapMat{
      {{1, 0, 0, 0}, {0, 0, 1, 0}, {0, 1, 0, 0}, {0, 0, 0, 1}}};
};
} // namespace qc
