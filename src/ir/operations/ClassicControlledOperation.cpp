/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "ir/operations/ClassicControlledOperation.hpp"

#include "Definitions.hpp"
#include "ir/Permutation.hpp"
#include "ir/operations/OpType.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <ostream>
#include <string>
#include <utility>

namespace qc {

std::string toString(const ComparisonKind& kind) {
  switch (kind) {
  case Eq:
    return "==";
  case Neq:
    return "!=";
  case Lt:
    return "<";
  case Leq:
    return "<=";
  case Gt:
    return ">";
  case Geq:
    return ">=";
  default:
    unreachable();
  }
}

std::ostream& operator<<(std::ostream& os, const ComparisonKind& kind) {
  os << toString(kind);
  return os;
}

ClassicControlledOperation::ClassicControlledOperation(
    std::unique_ptr<Operation>&& operation, ClassicalRegister controlReg,
    const std::uint64_t expectedVal, ComparisonKind kind)
    : op(std::move(operation)), controlRegister(std::move(controlReg)),
      expectedValue(expectedVal), comparisonKind(kind) {
  name = "c_" + shortName(op->getType());
  parameter.reserve(3);
  parameter.emplace_back(static_cast<fp>(controlRegister.first));
  parameter.emplace_back(static_cast<fp>(controlRegister.second));
  parameter.emplace_back(static_cast<fp>(expectedValue));
  type = ClassicControlled;
}
ClassicControlledOperation::ClassicControlledOperation(
    const ClassicControlledOperation& ccop)
    : Operation(ccop), controlRegister(ccop.controlRegister),
      expectedValue(ccop.expectedValue) {
  op = ccop.op->clone();
}
ClassicControlledOperation&
ClassicControlledOperation::operator=(const ClassicControlledOperation& ccop) {
  if (this != &ccop) {
    Operation::operator=(ccop);
    controlRegister = ccop.controlRegister;
    expectedValue = ccop.expectedValue;
    op = ccop.op->clone();
  }
  return *this;
}

bool ClassicControlledOperation::equals(const Operation& operation,
                                        const Permutation& perm1,
                                        const Permutation& perm2) const {
  if (const auto* classic =
          dynamic_cast<const ClassicControlledOperation*>(&operation)) {
    if (controlRegister != classic->controlRegister) {
      return false;
    }

    if (expectedValue != classic->expectedValue ||
        comparisonKind != classic->comparisonKind) {
      return false;
    }

    return op->equals(*classic->op, perm1, perm2);
  }
  return false;
}
void ClassicControlledOperation::dumpOpenQASM(std::ostream& of,
                                              const RegisterNames& qreg,
                                              const RegisterNames& creg,
                                              const std::size_t indent,
                                              const bool openQASM3) const {
  of << std::string(indent * OUTPUT_INDENT_SIZE, ' ');
  of << "if (";
  if (isWholeQubitRegister(creg, controlRegister.first,
                           controlRegister.first + controlRegister.second -
                               1)) {
    of << creg[controlRegister.first].first;
  } else {
    // This might use slices in the future to address multiple bits.
    if (controlRegister.second != 1) {
      throw QFRException(
          "Control register of classically controlled operation may either"
          " be a single bit or a whole register.");
    }
    of << creg[controlRegister.first].second;
  }
  of << " " << comparisonKind << " " << expectedValue << ") ";
  if (openQASM3) {
    of << "{\n";
  }
  op->dumpOpenQASM(of, qreg, creg, indent + 1, openQASM3);
  if (openQASM3) {
    of << "}\n";
  }
}

ComparisonKind getInvertedComparisonKind(const ComparisonKind kind) {
  switch (kind) {
  case Lt:
    return Geq;
  case Leq:
    return Gt;
  case Gt:
    return Leq;
  case Geq:
    return Lt;
  case Eq:
    return Neq;
  case Neq:
    return Eq;
  default:
    unreachable();
  }
}
} // namespace qc
