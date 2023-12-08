#pragma once

#include "dd/DDDefinitions.hpp"
#include "dd/GateMatrixDefinitions.hpp"

namespace qc {
template <typename MatrixType> class GateMatrixInterface {
public:
  virtual ~GateMatrixInterface() = default;

  virtual MatrixType getGateMatrix() = 0;
  virtual MatrixType getInverseGateMatrix() = 0;
  virtual bool isSingleTargetGate() = 0;
  virtual bool isTwoTargetGate() = 0;
  virtual bool isThreeOrMoreTargetGate() = 0;

  // TODO: add other useful methods (e.g. pow - raise a gate matrix to a certain
  //  power)
};
} // namespace qc
