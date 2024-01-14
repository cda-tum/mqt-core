#pragma once

#include "Definitions.hpp"
#include "QuantumComputation.hpp"
#include "mqt_core_export.h"

namespace qc {
class MQT_CORE_EXPORT WState : public QuantumComputation {
public:
  explicit WState(Qubit nq);
};
} // namespace qc
