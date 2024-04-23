#pragma once

#include "Definitions.hpp"
#include "OpType.hpp"
#include "Permutation.hpp"

#include <array>
#include <fstream>
#include <map>
#include <memory>
#include <set>
#include <vector>

namespace qc {
class Operation {
protected:
  Controls controls;
  Targets targets;
  std::vector<fp> parameter;

  OpType type = None;
  std::string name;

  static bool isWholeQubitRegister(const RegisterNames& reg, std::size_t start,
                                   std::size_t end) {
    return !reg.empty() && reg[start].first == reg[end].first &&
           (start == 0 || reg[start].first != reg[start - 1].first) &&
           (end == reg.size() - 1 || reg[end].first != reg[end + 1].first);
  }

public:
  Operation() = default;
  Operation(const Operation& op) = default;
  Operation(Operation&& op) noexcept = default;
  Operation& operator=(const Operation& op) = default;
  Operation& operator=(Operation&& op) noexcept = default;

  // Virtual Destructor
  virtual ~Operation() = default;

  [[nodiscard]] virtual std::unique_ptr<Operation> clone() const = 0;

  // Getters
  [[nodiscard]] virtual const Targets& getTargets() const { return targets; }
  virtual Targets& getTargets() { return targets; }
  [[nodiscard]] virtual std::size_t getNtargets() const {
    return targets.size();
  }

  [[nodiscard]] virtual const Controls& getControls() const { return controls; }
  virtual Controls& getControls() { return controls; }
  [[nodiscard]] virtual std::size_t getNcontrols() const {
    return controls.size();
  }

  [[nodiscard]] const std::vector<fp>& getParameter() const {
    return parameter;
  }
  std::vector<fp>& getParameter() { return parameter; }

  [[nodiscard]] const std::string& getName() const { return name; }
  [[nodiscard]] virtual OpType getType() const { return type; }

  [[nodiscard]] virtual std::set<Qubit> getUsedQubits() const {
    const auto& opTargets = getTargets();
    const auto& opControls = getControls();
    std::set<Qubit> usedQubits = {opTargets.begin(), opTargets.end()};
    for (const auto& control : opControls) {
      usedQubits.insert(control.qubit);
    }
    return usedQubits;
  }

  [[nodiscard]] std::unique_ptr<Operation> getInverted() const {
    auto op = clone();
    op->invert();
    return op;
  }

  // Setter
  virtual void setTargets(const Targets& t) { targets = t; }

  virtual void setControls(const Controls& c) {
    clearControls();
    addControls(c);
  }

  virtual void addControl(Control c) = 0;

  void addControls(const Controls& c) {
    for (const auto& control : c) {
      addControl(control);
    }
  }

  virtual void clearControls() = 0;

  virtual void removeControl(Control c) = 0;

  virtual Controls::iterator removeControl(Controls::iterator it) = 0;

  void removeControls(const Controls& c) {
    for (auto it = c.begin(); it != c.end();) {
      it = removeControl(it);
    }
  }

  virtual void setGate(const OpType g) {
    type = g;
    name = toString(g);
  }

  virtual void setParameter(const std::vector<fp>& p) { parameter = p; }

  virtual void apply(const Permutation& permutation);

  [[nodiscard]] virtual bool isUnitary() const { return true; }

  [[nodiscard]] virtual bool isStandardOperation() const { return false; }

  [[nodiscard]] virtual bool isCompoundOperation() const { return false; }

  [[nodiscard]] virtual bool isNonUnitaryOperation() const { return false; }

  [[nodiscard]] virtual bool isClassicControlledOperation() const {
    return false;
  }

  [[nodiscard]] virtual bool isSymbolicOperation() const { return false; }

  [[nodiscard]] virtual auto isDiagonal() const -> bool {
    return std::find(DIAGONAL_GATES.begin(), DIAGONAL_GATES.end(), type) !=
           DIAGONAL_GATES.end();
  }

  /// An operation is individual if it acts on a single qubit individually, i.e.
  /// does not entangle qubits.
  [[nodiscard]] virtual auto isIndividual() -> bool {
    return getNcontrols() == 0 and isSingleQubitGate(type);
  }

  [[nodiscard]] virtual bool isControlled() const { return !controls.empty(); }

  [[nodiscard]] virtual bool actsOn(const Qubit i) const {
    for (const auto& t : targets) {
      if (t == i) {
        return true;
      }
    }
    return controls.count(i) > 0;
  }

  virtual void addDepthContribution(std::vector<std::size_t>& depths) const;

  [[nodiscard]] virtual bool equals(const Operation& op,
                                    const Permutation& perm1,
                                    const Permutation& perm2) const;
  [[nodiscard]] virtual bool equals(const Operation& op) const {
    return equals(op, {}, {});
  }

  virtual std::ostream& printParameters(std::ostream& os) const;
  std::ostream& print(std::ostream& os, const std::size_t nqubits) const {
    return print(os, {}, 0, nqubits);
  }
  virtual std::ostream& print(std::ostream& os, const Permutation& permutation,
                              std::size_t prefixWidth,
                              std::size_t nqubits) const;

  void dumpOpenQASM2(std::ostream& of, const RegisterNames& qreg,
                     const RegisterNames& creg) const {
    dumpOpenQASM(of, qreg, creg, 0, false);
  }
  void dumpOpenQASM3(std::ostream& of, const RegisterNames& qreg,
                     const RegisterNames& creg) const {
    dumpOpenQASM(of, qreg, creg, 0, true);
  }
  virtual void dumpOpenQASM(std::ostream& of, const RegisterNames& qreg,
                            const RegisterNames& creg, size_t indent,
                            bool openQASM3) const = 0;

  /// Checks whether operation commutes with other operation on a given qubit.
  [[nodiscard]] virtual auto
  commutesAtQubit(const Operation& /*other*/,
                  const Qubit& /*qubit*/) const -> bool {
    return false;
  }

  [[nodiscard]] virtual auto
  isInverseOf(const Operation& /*other*/) const -> bool {
    return false;
  }

  virtual void invert() = 0;

  virtual bool operator==(const Operation& rhs) const { return equals(rhs); }
  bool operator!=(const Operation& rhs) const { return !(*this == rhs); }
};
} // namespace qc

namespace std {
template <> struct hash<qc::Operation> {
  std::size_t operator()(const qc::Operation& op) const noexcept {
    std::size_t seed = 0U;
    qc::hashCombine(seed, hash<qc::OpType>{}(op.getType()));
    for (const auto& control : op.getControls()) {
      qc::hashCombine(seed, hash<qc::Qubit>{}(control.qubit));
      if (control.type == qc::Control::Type::Neg) {
        seed ^= 1ULL;
      }
    }
    for (const auto& target : op.getTargets()) {
      qc::hashCombine(seed, hash<qc::Qubit>{}(target));
    }
    for (const auto& param : op.getParameter()) {
      qc::hashCombine(seed, hash<qc::fp>{}(param));
    }
    return seed;
  }
};
} // namespace std
