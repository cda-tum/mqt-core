#pragma once

#include "NestedEnvironment.hpp"
#include "QuantumComputation.hpp"
#include "Statement.hpp"

namespace qasm3 {
struct GateInfo {
public:
  size_t nControls;
  size_t nTargets;
  size_t nParameters;
  qc::OpType type;
};

struct Gate {
public:
  virtual ~Gate() = default;

  virtual size_t getNControls() = 0;
  virtual size_t getNTargets() = 0;
  virtual size_t getNParameters() = 0;
};

struct StandardGate : public Gate {
public:
  GateInfo info;

  explicit StandardGate(GateInfo info) : info(info) {}

  size_t getNControls() override { return info.nControls; }

  size_t getNTargets() override { return info.nTargets; }
  size_t getNParameters() override { return info.nParameters; }
};

struct CompoundGate : public Gate {
public:
  std::vector<std::string> parameterNames;
  std::vector<std::string> targetNames;
  std::vector<std::shared_ptr<GateCallStatement>> body;

  explicit CompoundGate(std::vector<std::string> parameterNames,
                        std::vector<std::string> targetNames,
                        std::vector<std::shared_ptr<GateCallStatement>> body)
      : parameterNames(std::move(parameterNames)),
        targetNames(std::move(targetNames)), body(std::move(body)) {}

  size_t getNControls() override { return 0; }

  size_t getNTargets() override { return targetNames.size(); }
  size_t getNParameters() override { return parameterNames.size(); }
};
} // namespace qasm3
