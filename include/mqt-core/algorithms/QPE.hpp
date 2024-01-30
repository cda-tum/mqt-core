#pragma once

#include "QuantumComputation.hpp"

namespace qc {
class QPE : public QuantumComputation {
public:
  fp lambda = 0.;
  std::size_t precision;
  bool iterative;

  explicit QPE(std::size_t nq, bool exact = true, bool iter = false);
  QPE(fp l, std::size_t prec, bool iter = false);

  std::ostream& printStatistics(std::ostream& os) const override;

protected:
  void createCircuit();
};
} // namespace qc
