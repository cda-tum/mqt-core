#pragma once

#include "dd/DDDefinitions.hpp"
#include "dd/GateMatrixDefinitions.hpp"

namespace qc {
class GateMatrixInterface {
public:
  virtual ~GateMatrixInterface() = default;

  virtual dd::GateMatrix getGateMatrix() = 0;
  //  virtual dd::TwoQubitGateMatrix getGateMatrix() = 0;
  virtual bool isSingleTargetGate() = 0;
  virtual bool isTwoTargetGate() = 0;
  virtual bool isThreeOrMoreTargetGate() = 0;
  virtual dd::GateMatrix getInverseGateMatrix() = 0;

  // TODO: add other useful methods
};
} // namespace qc
