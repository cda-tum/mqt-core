#include "operations/SymbolicOperation.hpp"

#include "Definitions.hpp"
#include "operations/StandardOperation.hpp"

#include <variant>

namespace qc {

    void SymbolicOperation::storeSymbolOrNumber(const SymbolOrNumber& param, std::size_t i) {
        if (std::holds_alternative<double>(param)) {
            parameter[i] = std::get<double>(param);
        } else {
            symbolicParameter[i] = std::get<Symbolic>(param);
        }
    }

    OpType SymbolicOperation::parseU3(const Symbolic& lambda, dd::fp& phi,
                                      dd::fp& theta) {
        if (std::abs(theta) < PARAMETER_TOLERANCE &&
            std::abs(phi) < PARAMETER_TOLERANCE) {
            phi   = 0.L;
            theta = 0.L;
            return SymbolicOperation::parseU1(lambda);
        }

        if (std::abs(theta - dd::PI_2) < PARAMETER_TOLERANCE) {
            theta    = dd::PI_2;
            auto res = parseU2(lambda, phi);
            if (res != U2)
                theta = 0.L;
            return res;
        }
        // parse a real u3 gate
        checkInteger(phi);
        checkFractionPi(phi);
        checkInteger(theta);
        checkFractionPi(theta);

        return U3;
    }
    OpType SymbolicOperation::parseU3(dd::fp& lambda, const Symbolic& phi,
                                      dd::fp& theta) {
        if (std::abs(theta - dd::PI_2) < PARAMETER_TOLERANCE) {
            theta    = dd::PI_2;
            auto res = parseU2(lambda, phi);
            if (res != U2)
                theta = 0.L;
            return res;
        }

        if (std::abs(lambda) < PARAMETER_TOLERANCE) {
            lambda = 0.L;
        }

        if (std::abs(lambda - dd::PI_2) < PARAMETER_TOLERANCE) {
            lambda = dd::PI_2;
        }

        if (std::abs(lambda - dd::PI) < PARAMETER_TOLERANCE) {
            lambda = dd::PI;
        }

        // parse a real u3 gate
        checkInteger(lambda);
        checkFractionPi(lambda);
        checkInteger(theta);
        checkFractionPi(theta);

        return U3;
    }
    OpType SymbolicOperation::parseU3(dd::fp& lambda, dd::fp& phi,
                                      [[maybe_unused]] const Symbolic& theta) {
        if (std::abs(lambda) < PARAMETER_TOLERANCE) {
            lambda = 0.L;
            if (std::abs(phi) < PARAMETER_TOLERANCE) {
                phi = 0.L;
            }
        }

        if (std::abs(lambda - dd::PI_2) < PARAMETER_TOLERANCE) {
            lambda = dd::PI_2;

            if (std::abs(phi - dd::PI_2) < PARAMETER_TOLERANCE) {
                phi = dd::PI_2;
            }
        }

        if (std::abs(lambda - dd::PI) < PARAMETER_TOLERANCE) {
            lambda = dd::PI;
            if (std::abs(phi) < PARAMETER_TOLERANCE) {
                phi = 0.L;
            }
        }

        // parse a real u3 gate
        checkInteger(lambda);
        checkFractionPi(lambda);
        checkInteger(phi);
        checkFractionPi(phi);

        return U3;
    }
    OpType SymbolicOperation::parseU3(const Symbolic& lambda,
                                      const Symbolic& phi, dd::fp& theta) {
        if (std::abs(theta - dd::PI_2) < PARAMETER_TOLERANCE) {
            theta    = dd::PI_2;
            auto res = parseU2(lambda, phi);
            if (res != U2)
                theta = 0.L;
            return res;
        }

        // parse a real u3 gate

        checkInteger(theta);
        checkFractionPi(theta);

        return U3;
    }
    OpType SymbolicOperation::parseU3([[maybe_unused]] const Symbolic& lambda, dd::fp& phi,
                                      [[maybe_unused]] const Symbolic& theta) {
        // parse a real u3 gate
        checkInteger(phi);
        checkFractionPi(phi);

        return U3;
    }
    OpType SymbolicOperation::parseU3(dd::fp& lambda, [[maybe_unused]] const Symbolic& phi,
                                      [[maybe_unused]] const Symbolic& theta) {
        // parse a real u3 gate
        checkInteger(lambda);
        checkFractionPi(lambda);

        return U3;
    }

    OpType SymbolicOperation::parseU2([[maybe_unused]] const Symbolic& lambda, [[maybe_unused]] const Symbolic& phi) {
        return U2;
    }

    OpType SymbolicOperation::parseU2([[maybe_unused]] const Symbolic& lambda, dd::fp& phi) {
        checkInteger(phi);
        checkFractionPi(phi);

        return U2;
    }
    OpType SymbolicOperation::parseU2(dd::fp& lambda, [[maybe_unused]] const Symbolic& phi) {
        checkInteger(lambda);
        checkFractionPi(lambda);

        return U2;
    }

    OpType SymbolicOperation::parseU1([[maybe_unused]] const Symbolic& lambda) {
        return Phase;
    }

    void SymbolicOperation::checkUgate() {
        if (type == Phase) {
            if (!isSymbolicParameter(0))
                type = StandardOperation::parseU1(parameter[0]);
        } else if (type == U2) {
            if (!isSymbolicParameter(0) && !isSymbolicParameter(1))
                type = StandardOperation::parseU2(parameter[0], parameter[1]);
            else if (isSymbolicParameter(0))
                type = parseU2(symbolicParameter[0].value(), parameter[1]);
            else if (isSymbolicParameter(1))
                type = parseU2(parameter[1], symbolicParameter[1].value());
        } else if (type == U3) {
            if (!isSymbolicParameter(0) && !isSymbolicParameter(1) && !isSymbolicParameter(2))
                type = StandardOperation::parseU3(parameter[0], parameter[1], parameter[2]);
            else if (!isSymbolicParameter(0) && !isSymbolicParameter(1))
                type = parseU3(parameter[0], parameter[1], symbolicParameter[2].value());
            else if (!isSymbolicParameter(0) && !isSymbolicParameter(2))
                type = parseU3(parameter[0], symbolicParameter[1].value(), parameter[2]);
            else if (!isSymbolicParameter(1) && !isSymbolicParameter(2))
                type = parseU3(symbolicParameter[0].value(), parameter[1], parameter[2]);
            else if (!isSymbolicParameter(0))
                type = parseU3(parameter[0], symbolicParameter[1].value(), symbolicParameter[2].value());
            else if (!isSymbolicParameter(1))
                type = parseU3(symbolicParameter[0].value(), parameter[1], symbolicParameter[2].value());
            else if (!isSymbolicParameter(2))
                type = parseU3(symbolicParameter[0].value(), symbolicParameter[1].value(), parameter[2]);
        }
    }

    void SymbolicOperation::setup(dd::QubitCount nq, const SymbolOrNumber& par0, const SymbolOrNumber& par1, const SymbolOrNumber& par2, dd::Qubit startingQubit) {
        nqubits = nq;
        storeSymbolOrNumber(par0, 0);
        storeSymbolOrNumber(par1, 1);
        storeSymbolOrNumber(par2, 2);
        startQubit = startingQubit;
        checkUgate();
        setName();
    }

    [[nodiscard]] dd::fp SymbolicOperation::getInstantiation(const SymbolOrNumber& symOrNum, const VariableAssignment& assignment) {
        return std::visit(Overload{
                                  [&](const dd::fp num) { return num; },
                                  [&](const Symbolic& sym) { return sym.evaluate(assignment); }},
                          symOrNum);
    }

    SymbolicOperation::SymbolicOperation(dd::QubitCount nq, dd::Qubit target, OpType g, const SymbolOrNumber& lambda, const SymbolOrNumber& phi, const SymbolOrNumber& theta, dd::Qubit startingQubit) {
        type = g;
        setup(nq, lambda, phi, theta, startingQubit);
        targets.emplace_back(target);
    }

    SymbolicOperation::SymbolicOperation(dd::QubitCount nq, const Targets& targets, OpType g, const SymbolOrNumber& lambda, const SymbolOrNumber& phi, const SymbolOrNumber& theta, dd::Qubit startingQubit) {
        type = g;
        setup(nq, lambda, phi, theta, startingQubit);
        this->targets = targets;
    }

    SymbolicOperation::SymbolicOperation(dd::QubitCount nq, dd::Control control, dd::Qubit target, OpType g, const SymbolOrNumber& lambda, const SymbolOrNumber& phi, const SymbolOrNumber& theta, dd::Qubit startingQubit):
        SymbolicOperation(nq, target, g, lambda, phi, theta, startingQubit) {
        controls.insert(control);
    }

    SymbolicOperation::SymbolicOperation(dd::QubitCount nq, dd::Control control, const Targets& targets, OpType g, const SymbolOrNumber& lambda, const SymbolOrNumber& phi, const SymbolOrNumber& theta, dd::Qubit startingQubit):
        SymbolicOperation(nq, targets, g, lambda, phi, theta, startingQubit) {
        controls.insert(control);
    }

    SymbolicOperation::SymbolicOperation(dd::QubitCount nq, const dd::Controls& controls, dd::Qubit target, OpType g, const SymbolOrNumber& lambda, const SymbolOrNumber& phi, const SymbolOrNumber& theta, dd::Qubit startingQubit):
        SymbolicOperation(nq, target, g, lambda, phi, theta, startingQubit) {
        this->controls = controls;
    }

    SymbolicOperation::SymbolicOperation(dd::QubitCount nq, const dd::Controls& controls, const Targets& targets, OpType g, const SymbolOrNumber& lambda, const SymbolOrNumber& phi, const SymbolOrNumber& theta, dd::Qubit startingQubit):
        SymbolicOperation(nq, targets, g, lambda, phi, theta, startingQubit) {
        this->controls = controls;
    }

    // MCF (cSWAP), Peres, paramterized two target Constructor
    SymbolicOperation::SymbolicOperation(dd::QubitCount nq, const dd::Controls& controls, dd::Qubit target0, dd::Qubit target1, OpType g, const SymbolOrNumber& lambda, const SymbolOrNumber& phi, const SymbolOrNumber& theta, dd::Qubit startingQubit):
        SymbolicOperation(nq, controls, {target0, target1}, g, lambda, phi, theta, startingQubit) {
    }

    bool SymbolicOperation::equals(const Operation& op, const Permutation& perm1, const Permutation& perm2) const {
        if (!op.isSymbolicOperation() && !isStandardOperation()) {
            return false;
        }
        if (isStandardOperation() && qc::StandardOperation::equals(op, perm1, perm2)) return true;

        if (!op.isSymbolicOperation()) return false;
        const auto& symOp = dynamic_cast<const SymbolicOperation&>(op);
        for (std::size_t i = 0; i < symbolicParameter.size(); ++i) {
            if (symbolicParameter[i].has_value() != symOp.symbolicParameter[i].has_value())
                return false;

            if (symbolicParameter[i].has_value())
                return symbolicParameter[i].value() == symOp.symbolicParameter[i].value();
        }
        return true;
    }

    void SymbolicOperation::dumpOpenQASM([[maybe_unused]] std::ostream& of, [[maybe_unused]] const RegisterNames& qreg, [[maybe_unused]] const RegisterNames& creg) const {
        throw QFRException("OpenQasm2.0 doesn't support parametrized gates!");
    }
    void SymbolicOperation::dumpQiskit([[maybe_unused]] std::ostream& of, [[maybe_unused]] const RegisterNames& qreg, [[maybe_unused]] const RegisterNames& creg, [[maybe_unused]] const char* anc_reg_name) const {
        throw QFRException("Dumping Qiskit Circuit is not supported for parameterized gates.");
    }

    StandardOperation SymbolicOperation::getInstantiatedOperation(const VariableAssignment& assignment) const {
        auto lambda = getInstantiation(getParameter(0), assignment);
        auto phi    = getInstantiation(getParameter(1), assignment);
        auto theta  = getInstantiation(getParameter(2), assignment);
        return StandardOperation(nqubits, targets, type, lambda, phi, theta, startQubit);
    }

    // Instantiates this Operation
    // Afterwards casting to StandardOperation can be done if assignment is total
    void SymbolicOperation::instantiate(const VariableAssignment& assignment) {
        for (std::size_t i = 0; i < symbolicParameter.size(); ++i) {
            parameter[i] = getInstantiation(getParameter(i), assignment);
            symbolicParameter[i].reset();
        }
        checkUgate();
    }
} // namespace qc
