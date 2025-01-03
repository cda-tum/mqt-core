/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include "Control.hpp"
#include "Definitions.hpp"
#include "OpType.hpp"
#include "Operation.hpp"
#include "ir/Permutation.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <ostream>
#include <string>
#include <utility>

namespace qc {

enum ComparisonKind : std::uint8_t {
  Eq,
  Neq,
  Lt,
  Leq,
  Gt,
  Geq,
};

ComparisonKind getInvertedComparisonKind(ComparisonKind kind);

std::string toString(const ComparisonKind& kind);

std::ostream& operator<<(std::ostream& os, const ComparisonKind& kind);

class ClassicControlledOperation final : public Operation {
private:
  std::unique_ptr<Operation> op;
  ClassicalRegister controlRegister;
  std::uint64_t expectedValue = 1U;
  ComparisonKind comparisonKind = ComparisonKind::Eq;

public:
  // Applies operation `_op` if the creg starting at index `control` has the
  // expected value
  ClassicControlledOperation(std::unique_ptr<qc::Operation>&& operation,
                             ClassicalRegister controlReg,
                             std::uint64_t expectedVal = 1U,
                             ComparisonKind kind = ComparisonKind::Eq)
      : op(std::move(operation)), controlRegister(std::move(controlReg)),
        expectedValue(expectedVal), comparisonKind(kind) {
    name = "c_" + shortName(op->getType());
    parameter.reserve(3);
    parameter.emplace_back(static_cast<fp>(controlRegister.first));
    parameter.emplace_back(static_cast<fp>(controlRegister.second));
    parameter.emplace_back(static_cast<fp>(expectedValue));
    type = ClassicControlled;
  }

  ClassicControlledOperation(const ClassicControlledOperation& ccop)
      : Operation(ccop), controlRegister(ccop.controlRegister),
        expectedValue(ccop.expectedValue) {
    op = ccop.op->clone();
  }

  ClassicControlledOperation&
  operator=(const ClassicControlledOperation& ccop) {
    if (this != &ccop) {
      Operation::operator=(ccop);
      controlRegister = ccop.controlRegister;
      expectedValue = ccop.expectedValue;
      op = ccop.op->clone();
    }
    return *this;
  }

  [[nodiscard]] std::unique_ptr<Operation> clone() const override {
    return std::make_unique<ClassicControlledOperation>(*this);
  }

  [[nodiscard]] auto getControlRegister() const { return controlRegister; }

  [[nodiscard]] auto getExpectedValue() const { return expectedValue; }

  [[nodiscard]] auto getOperation() const { return op.get(); }

  [[nodiscard]] auto getComparisonKind() const { return comparisonKind; }

  [[nodiscard]] const Targets& getTargets() const override {
    return op->getTargets();
  }

  Targets& getTargets() override { return op->getTargets(); }

  [[nodiscard]] std::size_t getNtargets() const override {
    return op->getNtargets();
  }

  [[nodiscard]] const Controls& getControls() const override {
    return op->getControls();
  }

  Controls& getControls() override { return op->getControls(); }

  [[nodiscard]] std::size_t getNcontrols() const override {
    return op->getNcontrols();
  }

  [[nodiscard]] bool isUnitary() const override { return false; }

  [[nodiscard]] bool isClassicControlledOperation() const override {
    return true;
  }

  [[nodiscard]] bool actsOn(Qubit i) const override { return op->actsOn(i); }

  void addControl(const Control c) override { op->addControl(c); }

  void clearControls() override { op->clearControls(); }

  void removeControl(const Control c) override { op->removeControl(c); }

  Controls::iterator removeControl(const Controls::iterator it) override {
    return op->removeControl(it);
  }

  [[nodiscard]] bool equals(const Operation& operation,
                            const Permutation& perm1,
                            const Permutation& perm2) const override {
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
  [[nodiscard]] bool equals(const Operation& operation) const override {
    return equals(operation, {}, {});
  }

  void dumpOpenQASM(std::ostream& of, const RegisterNames& qreg,
                    const RegisterNames& creg, std::size_t indent,
                    bool openQASM3) const override {
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

  void invert() override { op->invert(); }
};
} // namespace qc

namespace std {
template <> struct hash<qc::ClassicControlledOperation> {
  std::size_t
  operator()(qc::ClassicControlledOperation const& ccop) const noexcept {
    auto seed = qc::combineHash(ccop.getControlRegister().first,
                                ccop.getControlRegister().second);
    qc::hashCombine(seed, ccop.getExpectedValue());
    qc::hashCombine(seed, std::hash<qc::Operation>{}(*ccop.getOperation()));
    return seed;
  }
};
} // namespace std
