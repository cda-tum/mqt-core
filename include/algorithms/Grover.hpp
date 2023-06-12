#pragma once

#include "QuantumComputation.hpp"

#include <bitset>
#include <functional>
#include <random>

namespace qc {
class Grover : public QuantumComputation {
public:
  std::size_t seed = 0;
  BitString targetValue = 0;
  std::size_t iterations = 1;
  std::string expected{};
  std::size_t nDataQubits{};

  explicit Grover(std::size_t nq, std::size_t s = 0);

  void setup(QuantumComputation& qc) const;

  void oracle(QuantumComputation& qc) const;

  void diffusion(QuantumComputation& qc) const;

  void fullGrover(QuantumComputation& qc) const;

  std::ostream& printStatistics(std::ostream& os) const override;
};
} // namespace qc
