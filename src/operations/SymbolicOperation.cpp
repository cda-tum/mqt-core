#include "operations/SymbolicOperation.hpp"

#include "Definitions.hpp"
#include "operations/StandardOperation.hpp"

#include <variant>

namespace qc {

void SymbolicOperation::storeSymbolOrNumber(const SymbolOrNumber& param,
                                            const std::size_t i) {
  if (std::holds_alternative<fp>(param)) {
    parameter.at(i) = std::get<fp>(param);
  } else {
    symbolicParameter.at(i) = std::get<Symbolic>(param);
  }
}

OpType SymbolicOperation::parseU3([[maybe_unused]] const Symbolic& theta,
                                  fp& phi, fp& lambda) {
  if (std::abs(lambda) < PARAMETER_TOLERANCE) {
    lambda = 0.L;
    if (std::abs(phi) < PARAMETER_TOLERANCE) {
      phi = 0.L;
    }
  }

  if (std::abs(lambda - PI_2) < PARAMETER_TOLERANCE) {
    lambda = PI_2;
    if (std::abs(phi - PI_2) < PARAMETER_TOLERANCE) {
      phi = PI_2;
    }
  }

  if (std::abs(lambda - PI) < PARAMETER_TOLERANCE) {
    lambda = PI;
    if (std::abs(phi) < PARAMETER_TOLERANCE) {
      phi = 0.L;
    }
  }

  // parse a real u3 gate
  checkInteger(lambda);
  checkFractionPi(lambda);
  checkInteger(phi);
  checkFractionPi(phi);

  return U;
}
OpType SymbolicOperation::parseU3(fp& theta, const Symbolic& phi, fp& lambda) {
  if (std::abs(theta - PI_2) < PARAMETER_TOLERANCE) {
    theta = PI_2;
    return parseU2(phi, lambda);
  }

  if (std::abs(lambda) < PARAMETER_TOLERANCE) {
    lambda = 0.L;
  }

  if (std::abs(lambda - PI_2) < PARAMETER_TOLERANCE) {
    lambda = PI_2;
  }

  if (std::abs(lambda - PI) < PARAMETER_TOLERANCE) {
    lambda = PI;
  }

  // parse a real u3 gate
  checkInteger(lambda);
  checkFractionPi(lambda);
  checkInteger(theta);
  checkFractionPi(theta);

  return U;
}
OpType SymbolicOperation::parseU3(fp& theta, fp& phi,
                                  [[maybe_unused]] const Symbolic& lambda) {
  if (std::abs(theta) < PARAMETER_TOLERANCE &&
      std::abs(phi) < PARAMETER_TOLERANCE) {
    phi = 0.L;
    theta = 0.L;
    return SymbolicOperation::parseU1(lambda);
  }

  if (std::abs(theta - PI_2) < PARAMETER_TOLERANCE) {
    theta = PI_2;
    return parseU2(phi, lambda);
  }
  // parse a real u3 gate
  checkInteger(phi);
  checkFractionPi(phi);
  checkInteger(theta);
  checkFractionPi(theta);

  return U;
}
OpType SymbolicOperation::parseU3([[maybe_unused]] const Symbolic& theta,
                                  [[maybe_unused]] const Symbolic& phi,
                                  fp& lambda) {
  // parse a real u3 gate
  checkInteger(lambda);
  checkFractionPi(lambda);

  return U;
}
OpType SymbolicOperation::parseU3([[maybe_unused]] const Symbolic& theta,
                                  fp& phi,
                                  [[maybe_unused]] const Symbolic& lambda) {
  // parse a real u3 gate
  checkInteger(phi);
  checkFractionPi(phi);

  return U;
}
OpType SymbolicOperation::parseU3(fp& theta, const Symbolic& phi,
                                  const Symbolic& lambda) {
  if (std::abs(theta - PI_2) < PARAMETER_TOLERANCE) {
    theta = PI_2;
    return parseU2(phi, lambda);
  }

  // parse a real u3 gate

  checkInteger(theta);
  checkFractionPi(theta);

  return U;
}

OpType SymbolicOperation::parseU2([[maybe_unused]] const Symbolic& phi,
                                  [[maybe_unused]] const Symbolic& lambda) {
  return U2;
}

OpType SymbolicOperation::parseU2([[maybe_unused]] const Symbolic& phi,
                                  fp& lambda) {
  checkInteger(lambda);
  checkFractionPi(lambda);

  return U2;
}
OpType SymbolicOperation::parseU2(fp& phi,
                                  [[maybe_unused]] const Symbolic& lambda) {
  checkInteger(phi);
  checkFractionPi(phi);

  return U2;
}

OpType SymbolicOperation::parseU1([[maybe_unused]] const Symbolic& lambda) {
  return P;
}

void SymbolicOperation::checkSymbolicUgate() {
  // NOLINTBEGIN(bugprone-unchecked-optional-access) - we check for this
  if (type == P) {
    if (!isSymbolicParameter(0)) {
      type = StandardOperation::parseU1(parameter[0]);
    }
  } else if (type == U2) {
    if (!isSymbolicParameter(0) && !isSymbolicParameter(1)) {
      type = StandardOperation::parseU2(parameter[0], parameter[1]);
    } else if (isSymbolicParameter(0)) {
      type = parseU2(symbolicParameter[0].value(), parameter[1]);
    } else if (isSymbolicParameter(1)) {
      type = parseU2(parameter[0], symbolicParameter[1].value());
    }
  } else if (type == U) {
    if (!isSymbolicParameter(0) && !isSymbolicParameter(1) &&
        !isSymbolicParameter(2)) {
      type =
          StandardOperation::parseU3(parameter[0], parameter[1], parameter[2]);
    } else if (!isSymbolicParameter(0) && !isSymbolicParameter(1)) {
      type = parseU3(parameter[0], parameter[1], symbolicParameter[2].value());
    } else if (!isSymbolicParameter(0) && !isSymbolicParameter(2)) {
      type = parseU3(parameter[0], symbolicParameter[1].value(), parameter[2]);
    } else if (!isSymbolicParameter(1) && !isSymbolicParameter(2)) {
      type = parseU3(symbolicParameter[0].value(), parameter[1], parameter[2]);
    } else if (!isSymbolicParameter(0)) {
      type = parseU3(parameter[0], symbolicParameter[1].value(),
                     symbolicParameter[2].value());
    } else if (!isSymbolicParameter(1)) {
      type = parseU3(symbolicParameter[0].value(), parameter[1],
                     symbolicParameter[2].value());
    } else if (!isSymbolicParameter(2)) {
      type = parseU3(symbolicParameter[0].value(), symbolicParameter[1].value(),
                     parameter[2]);
    }
  }
  // NOLINTEND(bugprone-unchecked-optional-access)
}

void SymbolicOperation::setup(const std::size_t nq,
                              const std::vector<SymbolOrNumber>& params,
                              const Qubit startingQubit) {
  nqubits = nq;
  const auto numParams = params.size();
  parameter.resize(numParams);
  symbolicParameter.resize(numParams);
  for (std::size_t i = 0; i < numParams; ++i) {
    storeSymbolOrNumber(params[i], i);
  }
  startQubit = startingQubit;
  checkSymbolicUgate();
  name = toString(type);
}

[[nodiscard]] fp
SymbolicOperation::getInstantiation(const SymbolOrNumber& symOrNum,
                                    const VariableAssignment& assignment) {
  return std::visit(
      Overload{[&](const fp num) { return num; },
               [&](const Symbolic& sym) { return sym.evaluate(assignment); }},
      symOrNum);
}

SymbolicOperation::SymbolicOperation(const std::size_t nq, const Qubit target,
                                     const OpType g,
                                     const std::vector<SymbolOrNumber>& params,
                                     const Qubit startingQubit) {
  type = g;
  setup(nq, params, startingQubit);
  targets.emplace_back(target);
}

SymbolicOperation::SymbolicOperation(const std::size_t nq, const Targets& targ,
                                     const OpType g,
                                     const std::vector<SymbolOrNumber>& params,
                                     const Qubit startingQubit) {
  type = g;
  setup(nq, params, startingQubit);
  targets = targ;
}

SymbolicOperation::SymbolicOperation(const std::size_t nq,
                                     const Control control, const Qubit target,
                                     const OpType g,
                                     const std::vector<SymbolOrNumber>& params,
                                     const Qubit startingQubit)
    : SymbolicOperation(nq, target, g, params, startingQubit) {
  controls.insert(control);
}

SymbolicOperation::SymbolicOperation(const std::size_t nq,
                                     const Control control, const Targets& targ,
                                     const OpType g,
                                     const std::vector<SymbolOrNumber>& params,
                                     const Qubit startingQubit)
    : SymbolicOperation(nq, targ, g, params, startingQubit) {
  controls.insert(control);
}

SymbolicOperation::SymbolicOperation(const std::size_t nq, const Controls& c,
                                     const Qubit target, const OpType g,
                                     const std::vector<SymbolOrNumber>& params,
                                     const Qubit startingQubit)
    : SymbolicOperation(nq, target, g, params, startingQubit) {
  controls = c;
}

SymbolicOperation::SymbolicOperation(const std::size_t nq, const Controls& c,
                                     const Targets& targ, const OpType g,
                                     const std::vector<SymbolOrNumber>& params,
                                     const Qubit startingQubit)
    : SymbolicOperation(nq, targ, g, params, startingQubit) {
  controls = c;
}

// MCF (cSWAP), Peres, parameterized two target Constructor
SymbolicOperation::SymbolicOperation(const std::size_t nq, const Controls& c,
                                     const Qubit target0, const Qubit target1,
                                     const OpType g,
                                     const std::vector<SymbolOrNumber>& params,
                                     const Qubit startingQubit)
    : SymbolicOperation(nq, c, {target0, target1}, g, params, startingQubit) {}

bool SymbolicOperation::equals(const Operation& op, const Permutation& perm1,
                               const Permutation& perm2) const {
  if (!op.isSymbolicOperation() && !isStandardOperation()) {
    return false;
  }
  if (isStandardOperation() &&
      qc::StandardOperation::equals(op, perm1, perm2)) {
    return true;
  }

  if (!op.isSymbolicOperation()) {
    return false;
  }
  const auto& symOp = dynamic_cast<const SymbolicOperation&>(op);
  for (std::size_t i = 0; i < symbolicParameter.size(); ++i) {
    const auto& symParam = symbolicParameter.at(i);
    const auto& symOpParam = symOp.symbolicParameter.at(i);
    const auto symParamIsSymbolic = symParam.has_value();
    const auto symOpParamIsSymbolic = symOpParam.has_value();

    if (symParamIsSymbolic != symOpParamIsSymbolic) {
      return false;
    }

    if (symParamIsSymbolic) {
      return symParam.value() == symOpParam.value();
    }
  }
  return true;
}

[[noreturn]] void SymbolicOperation::dumpOpenQASM(
    [[maybe_unused]] std::ostream& of,
    [[maybe_unused]] const RegisterNames& qreg,
    [[maybe_unused]] const RegisterNames& creg) const {
  throw QFRException("OpenQASM 2.0 doesn't support parametrized gates!");
}

StandardOperation SymbolicOperation::getInstantiatedOperation(
    const VariableAssignment& assignment) const {
  std::vector<fp> parameters;
  const auto size = symbolicParameter.size();
  parameters.reserve(size);
  for (std::size_t i = 0; i < size; ++i) {
    parameters.emplace_back(getInstantiation(getParameter(i), assignment));
  }
  return {nqubits, targets, type, parameters, startQubit};
}

// Instantiates this Operation
// Afterwards casting to StandardOperation can be done if assignment is total
void SymbolicOperation::instantiate(const VariableAssignment& assignment) {
  for (std::size_t i = 0; i < symbolicParameter.size(); ++i) {
    parameter.at(i) = getInstantiation(getParameter(i), assignment);
    symbolicParameter.at(i).reset();
  }
  checkUgate();
}

void SymbolicOperation::negateSymbolicParameter(const std::size_t index) {
  if (isSymbolicParameter(index)) {
    // NOLINTBEGIN(bugprone-unchecked-optional-access) - we check for this
    symbolicParameter.at(index) = -symbolicParameter.at(index).value();
    // NOLINTEND(bugprone-unchecked-optional-access)
  } else {
    parameter.at(index) = -parameter.at(index);
  }
}

void SymbolicOperation::addToSymbolicParameter(const std::size_t index,
                                               const fp value) {
  if (isSymbolicParameter(index)) {
    // NOLINTBEGIN(bugprone-unchecked-optional-access) - we check for this
    symbolicParameter.at(index) = symbolicParameter.at(index).value() + value;
    // NOLINTEND(bugprone-unchecked-optional-access)
  } else {
    parameter.at(index) += value;
  }
}

void SymbolicOperation::invert() {
  switch (type) {
  case GPhase:
  case P:
  case RX:
  case RY:
  case RZ:
  case RXX:
  case RYY:
  case RZZ:
  case RZX:
    negateSymbolicParameter(0);
    break;
  case U2:
    negateSymbolicParameter(0);
    negateSymbolicParameter(1);

    addToSymbolicParameter(0, -PI);
    addToSymbolicParameter(1, PI);
    std::swap(parameter[0], parameter[1]);
    std::swap(symbolicParameter[0], symbolicParameter[1]);
    break;
  case U:
    negateSymbolicParameter(0);
    negateSymbolicParameter(1);
    negateSymbolicParameter(2);

    std::swap(parameter[1], parameter[2]);
    std::swap(symbolicParameter[1], symbolicParameter[2]);
    break;
  case XXminusYY:
  case XXplusYY:
    negateSymbolicParameter(0);
    break;
  default:
    StandardOperation::invert();
  }
}
} // namespace qc
