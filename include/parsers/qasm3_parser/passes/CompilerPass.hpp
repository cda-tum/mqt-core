#pragma once

#include "parsers/qasm3_parser/Statement.hpp"

#include <vector>

namespace qasm3 {
class CompilerPass {
public:
  virtual ~CompilerPass() = default;

  virtual void processStatement(Statement& statement) = 0;
};
} // namespace qasm3
