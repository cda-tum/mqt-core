#pragma once

#include "Definitions.hpp"
#include "ir/QuantumComputation.hpp"

namespace qc {
class WState : public QuantumComputation {
public:
  explicit WState(Qubit nq);
};
} // namespace qc
