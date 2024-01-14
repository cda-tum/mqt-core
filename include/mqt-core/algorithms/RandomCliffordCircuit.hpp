#pragma once

#include "Definitions.hpp"
#include "QuantumComputation.hpp"
#include "mqt_core_export.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <ostream>

namespace qc {
class MQT_CORE_EXPORT RandomCliffordCircuit : public QuantumComputation {
protected:
  std::function<std::uint16_t()> cliffordGenerator;

  void append1QClifford(std::uint16_t idx, Qubit target);
  void append2QClifford(std::uint16_t, Qubit control, Qubit target);

public:
  std::size_t depth = 1;
  std::size_t seed = 0;

  explicit RandomCliffordCircuit(std::size_t nq, std::size_t depth = 1,
                                 std::size_t seed = 0);

  std::ostream& printStatistics(std::ostream& os) const override;
};
} // namespace qc
