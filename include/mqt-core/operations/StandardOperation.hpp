#pragma once

#include "Definitions.hpp"
#include "Operation.hpp"
#include "Permutation.hpp"
#include "operations/Control.hpp"
#include "operations/OpType.hpp"

#include <cmath>
#include <cstddef>
#include <memory>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

namespace qc {
class StandardOperation : public Operation {
protected:
  static void checkInteger(fp& ld) {
    const fp nearest = std::nearbyint(ld);
    if (std::abs(ld - nearest) < PARAMETER_TOLERANCE) {
      ld = nearest;
    }
  }

  static void checkFractionPi(fp& ld) {
    const fp div = PI / ld;
    const fp nearest = std::nearbyint(div);
    if (std::abs(div - nearest) < PARAMETER_TOLERANCE) {
      ld = PI / nearest;
    }
  }

  OpType parseU3(fp& theta, fp& phi, fp& lambda);
  OpType parseU2(fp& phi, fp& lambda);
  OpType parseU1(fp& lambda);

  void checkUgate();
  void setup();

  void dumpOpenQASMTeleportation(std::ostream& of,
                                 const RegisterNames& qreg) const;

public:
  StandardOperation() = default;

  // Standard Constructors
  StandardOperation(Qubit target, OpType g, std::vector<fp> params = {});
  StandardOperation(const Targets& targ, OpType g, std::vector<fp> params = {});

  StandardOperation(Control control, Qubit target, OpType g,
                    const std::vector<fp>& params = {});
  StandardOperation(Control control, const Targets& targ, OpType g,
                    const std::vector<fp>& params = {});

  StandardOperation(const Controls& c, Qubit target, OpType g,
                    const std::vector<fp>& params = {});
  StandardOperation(const Controls& c, const Targets& targ, OpType g,
                    const std::vector<fp>& params = {});

  // MCF (cSWAP), Peres, parameterized two target Constructor
  StandardOperation(const Controls& c, Qubit target0, Qubit target1, OpType g,
                    const std::vector<fp>& params = {});

  [[nodiscard]] std::unique_ptr<Operation> clone() const override {
    return std::make_unique<StandardOperation>(*this);
  }

  [[nodiscard]] bool isStandardOperation() const override { return true; }

  void addControl(const Control c) override {
    if (actsOn(c.qubit)) {
      throw QFRException("Cannot add control on qubit " +
                         std::to_string(c.qubit) +
                         " to operation it already acts on the qubit.");
    }

    controls.emplace(c);
  }

  void clearControls() override { controls.clear(); }

  void removeControl(const Control c) override {
    if (controls.erase(c) == 0) {
      throw QFRException("Cannot remove control on qubit " +
                         std::to_string(c.qubit) +
                         " from operation as it is not a control.");
    }
  }

  Controls::iterator removeControl(const Controls::iterator it) override {
    return controls.erase(it);
  }

  [[nodiscard]] bool equals(const Operation& op, const Permutation& perm1,
                            const Permutation& perm2) const override {
    return Operation::equals(op, perm1, perm2);
  }
  [[nodiscard]] bool equals(const Operation& operation) const override {
    return equals(operation, {}, {});
  }

  void dumpOpenQASM(std::ostream& of, const RegisterNames& qreg,
                    const RegisterNames& creg, std::size_t indent,
                    bool openQASM3) const override;

  [[nodiscard]] auto commutesAtQubit(const Operation& other,
                                     const Qubit& qubit) const -> bool override;

  void invert() override;

protected:
  void dumpOpenQASM2(std::ostream& of, std::ostringstream& op,
                     const RegisterNames& qreg) const;
  void dumpOpenQASM3(std::ostream& of, std::ostringstream& op,
                     const RegisterNames& qreg) const;

  void dumpGateType(std::ostream& of, std::ostringstream& op,
                    const RegisterNames& qreg) const;

  void dumpControls(std::ostringstream& op) const;
};

} // namespace qc
