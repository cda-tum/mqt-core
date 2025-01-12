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
public:
  // Applies operation `_op` if the creg starting at index `control` has the
  // expected value
  ClassicControlledOperation(std::unique_ptr<Operation>&& operation,
                             ClassicalRegister controlReg,
                             std::uint64_t expectedVal = 1U,
                             ComparisonKind kind = Eq);

  ClassicControlledOperation(const ClassicControlledOperation& ccop);

  ClassicControlledOperation& operator=(const ClassicControlledOperation& ccop);

  [[nodiscard]] std::unique_ptr<Operation> clone() const override {
    return std::make_unique<ClassicControlledOperation>(*this);
  }

  [[nodiscard]] auto getControlRegister() const noexcept {
    return controlRegister;
  }

  [[nodiscard]] auto getExpectedValue() const noexcept { return expectedValue; }

  [[nodiscard]] auto getOperation() const { return op.get(); }

  [[nodiscard]] auto getComparisonKind() const noexcept {
    return comparisonKind;
  }

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

  [[nodiscard]] bool isClassicControlledOperation() const noexcept override {
    return true;
  }

  [[nodiscard]] bool actsOn(const Qubit i) const override {
    return op->actsOn(i);
  }

  void addControl(const Control c) override { op->addControl(c); }

  void clearControls() override { op->clearControls(); }

  void removeControl(const Control c) override { op->removeControl(c); }

  Controls::iterator removeControl(const Controls::iterator it) override {
    return op->removeControl(it);
  }

  [[nodiscard]] bool equals(const Operation& operation,
                            const Permutation& perm1,
                            const Permutation& perm2) const override;
  [[nodiscard]] bool equals(const Operation& operation) const override {
    return equals(operation, {}, {});
  }

  void dumpOpenQASM(std::ostream& of, const RegisterNames& qreg,
                    const RegisterNames& creg, std::size_t indent,
                    bool openQASM3) const override;

  void invert() override { op->invert(); }

private:
  std::unique_ptr<Operation> op;
  ClassicalRegister controlRegister;
  std::uint64_t expectedValue = 1U;
  ComparisonKind comparisonKind = Eq;
};
} // namespace qc

template <> struct std::hash<qc::ClassicControlledOperation> {
  std::size_t
  operator()(qc::ClassicControlledOperation const& ccop) const noexcept {
    auto seed = qc::combineHash(ccop.getControlRegister().first,
                                ccop.getControlRegister().second);
    qc::hashCombine(seed, ccop.getExpectedValue());
    qc::hashCombine(seed, std::hash<qc::Operation>{}(*ccop.getOperation()));
    return seed;
  }
};
