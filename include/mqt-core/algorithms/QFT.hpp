#pragma once

#include "QuantumComputation.hpp"
#include "mqt_core_export.h"

#include <cstddef>
#include <ostream>

namespace qc {
class MQT_CORE_EXPORT QFT : public QuantumComputation {
public:
  explicit QFT(std::size_t nq, bool includeMeas = true, bool dyn = false);

  std::ostream& printStatistics(std::ostream& os) const override;

  std::size_t precision{};
  bool includeMeasurements;
  bool dynamic;

protected:
  void createCircuit();
};
} // namespace qc
