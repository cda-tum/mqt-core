#pragma once

#include "Operation.hpp"

#include <utility>

namespace qc {

class ClassicControlledOperation final : public Operation {
private:
  std::unique_ptr<Operation> op;
  ClassicalRegister controlRegister{};
  std::uint64_t expectedValue = 1U;

public:
  // Applies operation `_op` if the creg starting at index `control` has the
  // expected value
  ClassicControlledOperation(std::unique_ptr<qc::Operation>& operation,
                             ClassicalRegister controlReg,
                             std::uint64_t expectedVal = 1U)
      : op(std::move(operation)), controlRegister(std::move(controlReg)),
        expectedValue(expectedVal) {
    nqubits = op->getNqubits();
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

  void setNqubits(std::size_t nq) override {
    nqubits = nq;
    op->setNqubits(nq);
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

      if (expectedValue != classic->expectedValue) {
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
                    const RegisterNames& creg) const override {
    of << "if(";
    of << creg[controlRegister.first].first;
    of << " == " << expectedValue << ") ";
    op->dumpOpenQASM(of, qreg, creg);
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
