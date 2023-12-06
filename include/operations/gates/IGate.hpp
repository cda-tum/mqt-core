#pragma once

#include "GateMatrixInterface.hpp"
#include "dd/DDDefinitions.hpp"
#include "operations/OpType.hpp"
#include "operations/StandardOperation.hpp"

namespace qc {
class IGate : public GateMatrixInterface, StandardOperation {
  dd::GateMatrix getGateMatrix() override { return iMat; }

  bool isSingleTargetGate() override { return true; }
  bool isTwoTargetGate() override { return false; }
  bool isThreeOrMoreTargetGate() override { return false; }

  dd::GateMatrix getInverseGateMatrix() override { return iMat; }

private:
  dd::GateMatrix iMat{1, 0, 0, 1};
};
} // namespace qc
