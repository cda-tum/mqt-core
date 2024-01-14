#pragma once

#include "QuantumComputation.hpp"
#include "mqt_core_export.h"

#include <cstddef>

namespace qc {
class MQT_CORE_EXPORT Entanglement : public QuantumComputation {
public:
  explicit Entanglement(std::size_t nq);
};
} // namespace qc
