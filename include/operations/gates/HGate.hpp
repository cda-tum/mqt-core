#pragma once

#include "GateMatrixInterface.hpp"
#include "dd/DDDefinitions.hpp"
#include "operations/OpType.hpp"
#include "operations/StandardOperation.hpp"

namespace qc {
class HGate : public GateMatrixInterface, StandardOperation {
  dd::GateMatrix getGateMatrix() override { return hMat; }

  bool isSingleTargetGate() override { return true; }
  bool isTwoTargetGate() override { return false; }
  bool isThreeOrMoreTargetGate() override { return false; }

  dd::GateMatrix getInverseGateMatrix() override { return hMat; }

private:
  dd::GateMatrix hMat{dd::SQRT2_2, dd::SQRT2_2, dd::SQRT2_2, dd::SQRT2_2};
};
}  // namespace qc
