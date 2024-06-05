#pragma once

#include "QuantumComputation.hpp"

#include <cstddef>

namespace qc {
class Entanglement : public QuantumComputation {
public:
  explicit Entanglement(std::size_t nq);
};
} // namespace qc
