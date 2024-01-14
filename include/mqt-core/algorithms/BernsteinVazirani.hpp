#pragma once

#include "Definitions.hpp"
#include "QuantumComputation.hpp"
#include "mqt_core_export.h"

#include <cstddef>
#include <ostream>
#include <string>

namespace qc {
class MQT_CORE_EXPORT BernsteinVazirani : public QuantumComputation {
public:
  BitString s = 0;
  std::size_t bitwidth = 1;
  bool dynamic = false;
  std::string expected{};

  explicit BernsteinVazirani(const BitString& hiddenString, bool dyn = false);
  explicit BernsteinVazirani(std::size_t nq, bool dyn = false);
  BernsteinVazirani(const BitString& hiddenString, std::size_t nq,
                    bool dyn = false);

  std::ostream& printStatistics(std::ostream& os) const override;

protected:
  void createCircuit();
};
} // namespace qc
