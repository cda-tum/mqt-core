#pragma once

#include <QuantumComputation.hpp>

namespace qc {
class Entanglement : public QuantumComputation {
public:
  explicit Entanglement(std::size_t nq);
};
} // namespace qc
