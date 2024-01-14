#pragma once

#include "mqt_core_export.h"
#include "parsers/qasm3_parser/Statement.hpp"

namespace qasm3 {
class MQT_CORE_EXPORT CompilerPass {
public:
  virtual ~CompilerPass() = default;

  virtual void processStatement(Statement& statement) = 0;
};
} // namespace qasm3
