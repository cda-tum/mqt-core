#pragma once

#include "Operation.hpp"

namespace qc {

class NonUnitaryOperation final : public Operation {
protected:
  std::vector<Bit> classics{}; // vector for the classical bits to measure into

  void printMeasurement(std::ostream& os, const std::vector<Qubit>& q,
                        const std::vector<Bit>& c,
                        const Permutation& permutation) const;
  void printReset(std::ostream& os, const std::vector<Qubit>& q,
                  const Permutation& permutation) const;

public:
  // Measurement constructor
  NonUnitaryOperation(std::size_t nq, std::vector<Qubit> qubitRegister,
                      std::vector<Bit> classicalRegister);
  NonUnitaryOperation(std::size_t nq, Qubit qubit, Bit cbit);

  // General constructor
  NonUnitaryOperation(std::size_t nq, Targets qubits, OpType op = Reset);

  [[nodiscard]] std::unique_ptr<Operation> clone() const override {
    if (getType() == qc::Measure) {
      return std::make_unique<NonUnitaryOperation>(getNqubits(), getTargets(),
                                                   getClassics());
    }
    return std::make_unique<NonUnitaryOperation>(getNqubits(), getTargets(),
                                                 getType());
  }

  [[nodiscard]] bool isUnitary() const override { return false; }

  [[nodiscard]] bool isNonUnitaryOperation() const override { return true; }

  [[nodiscard]] const std::vector<Bit>& getClassics() const { return classics; }
  std::vector<Bit>& getClassics() { return classics; }
  [[nodiscard]] size_t getNclassics() const { return classics.size(); }

  void addControls(const Controls& /*c*/) override {
    throw QFRException("Cannot add controls to non-unitary operation.");
  }

  void clearControls() override {
    throw QFRException("Cannot clear controls from non-unitary operation.");
  }

  void removeControls(const Controls& /*c*/) override {
    throw QFRException("Cannot remove controls from non-unitary operation.");
  }

  [[nodiscard]] bool equals(const Operation& op, const Permutation& perm1,
                            const Permutation& perm2) const override;
  [[nodiscard]] bool equals(const Operation& operation) const override {
    return equals(operation, {}, {});
  }

  std::ostream& print(std::ostream& os) const override { return print(os, {}); }
  std::ostream& print(std::ostream& os,
                      const Permutation& permutation) const override;

  void dumpOpenQASM(std::ostream& of, const RegisterNames& qreg,
                    const RegisterNames& creg) const override;
};
} // namespace qc
