#include "operations/SymbolicOperation.hpp"

#include "Definitions.hpp"
#include "operations/StandardOperation.hpp"

#include <variant>

namespace qc {

    void SymbolicOperation::storeSymbolOrNumber(const SymbolOrNumber& param, std::size_t i) {
        if (std::holds_alternative<double>(param)) {
            parameter.at(i) = std::get<double>(param);
        } else {
            symbolicParameter.at(i) = std::get<Symbolic>(param);
        }
    }

    OpType SymbolicOperation::parseU3(const Symbolic& lambda, fp& phi,
                                      fp& theta) {
        if (std::abs(theta) < PARAMETER_TOLERANCE &&
            std::abs(phi) < PARAMETER_TOLERANCE) {
            phi   = 0.L;
            theta = 0.L;
            return SymbolicOperation::parseU1(lambda);
        }

        if (std::abs(theta - PI_2) < PARAMETER_TOLERANCE) {
            theta = PI_2;
            return parseU2(lambda, phi);
        }
        // parse a real u3 gate
        checkInteger(phi);
        checkFractionPi(phi);
        checkInteger(theta);
        checkFractionPi(theta);

        return U3;
    }
    OpType SymbolicOperation::parseU3(fp& lambda, const Symbolic& phi,
                                      fp& theta) {
        if (std::abs(theta - PI_2) < PARAMETER_TOLERANCE) {
            theta = PI_2;
            return parseU2(lambda, phi);
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

        return U3;
    }
    OpType SymbolicOperation::parseU3(fp& lambda, fp& phi,
                                      [[maybe_unused]] const Symbolic& theta) {
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

        return U3;
    }
    OpType SymbolicOperation::parseU3(const Symbolic& lambda,
                                      const Symbolic& phi, fp& theta) {
        if (std::abs(theta - PI_2) < PARAMETER_TOLERANCE) {
            theta = PI_2;
            return parseU2(lambda, phi);
        }

        // parse a real u3 gate

        checkInteger(theta);
        checkFractionPi(theta);

        return U3;
    }
    OpType SymbolicOperation::parseU3([[maybe_unused]] const Symbolic& lambda, fp& phi,
                                      [[maybe_unused]] const Symbolic& theta) {
        // parse a real u3 gate
        checkInteger(phi);
        checkFractionPi(phi);

        return U3;
    }
    OpType SymbolicOperation::parseU3(fp& lambda, [[maybe_unused]] const Symbolic& phi,
                                      [[maybe_unused]] const Symbolic& theta) {
        // parse a real u3 gate
        checkInteger(lambda);
        checkFractionPi(lambda);

        return U3;
    }

    OpType SymbolicOperation::parseU2([[maybe_unused]] const Symbolic& lambda, [[maybe_unused]] const Symbolic& phi) {
        return U2;
    }

    OpType SymbolicOperation::parseU2([[maybe_unused]] const Symbolic& lambda, fp& phi) {
        checkInteger(phi);
        checkFractionPi(phi);

        return U2;
    }
    OpType SymbolicOperation::parseU2(fp& lambda, [[maybe_unused]] const Symbolic& phi) {
        checkInteger(lambda);
        checkFractionPi(lambda);

        return U2;
    }

    OpType SymbolicOperation::parseU1([[maybe_unused]] const Symbolic& lambda) {
        return Phase;
    }

    void SymbolicOperation::checkSymbolicUgate() {
        if (type == Phase) {
            if (!isSymbolicParameter(0)) {
                type = StandardOperation::parseU1(parameter[0]);
            }
        } else if (type == U2) {
            if (!isSymbolicParameter(0) && !isSymbolicParameter(1)) {
                type = StandardOperation::parseU2(parameter[0], parameter[1]);
            } else if (isSymbolicParameter(0)) {
                type = parseU2(symbolicParameter[0].value(), parameter[1]);
            } else if (isSymbolicParameter(1)) {
                type = parseU2(parameter[1], symbolicParameter[1].value());
            }
        } else if (type == U3) {
            if (!isSymbolicParameter(0) && !isSymbolicParameter(1) && !isSymbolicParameter(2)) {
                type = StandardOperation::parseU3(parameter[0], parameter[1], parameter[2]);
            } else if (!isSymbolicParameter(0) && !isSymbolicParameter(1)) {
                type = parseU3(parameter[0], parameter[1], symbolicParameter[2].value());
            } else if (!isSymbolicParameter(0) && !isSymbolicParameter(2)) {
                type = parseU3(parameter[0], symbolicParameter[1].value(), parameter[2]);
            } else if (!isSymbolicParameter(1) && !isSymbolicParameter(2)) {
                type = parseU3(symbolicParameter[0].value(), parameter[1], parameter[2]);
            } else if (!isSymbolicParameter(0)) {
                type = parseU3(parameter[0], symbolicParameter[1].value(), symbolicParameter[2].value());
            } else if (!isSymbolicParameter(1)) {
                type = parseU3(symbolicParameter[0].value(), parameter[1], symbolicParameter[2].value());
            } else if (!isSymbolicParameter(2)) {
                type = parseU3(symbolicParameter[0].value(), symbolicParameter[1].value(), parameter[2]);
            }
        }
    }

    void SymbolicOperation::setup(const std::size_t nq, const SymbolOrNumber& par0, const SymbolOrNumber& par1, const SymbolOrNumber& par2, const Qubit startingQubit) {
        nqubits = nq;
        storeSymbolOrNumber(par0, 0);
        storeSymbolOrNumber(par1, 1);
        storeSymbolOrNumber(par2, 2);
        startQubit = startingQubit;
        checkSymbolicUgate();
        setName();
    }

    [[nodiscard]] fp SymbolicOperation::getInstantiation(const SymbolOrNumber& symOrNum, const VariableAssignment& assignment) {
        return std::visit(Overload{
                                  [&](const fp num) { return num; },
                                  [&](const Symbolic& sym) { return sym.evaluate(assignment); }},
                          symOrNum);
    }

    SymbolicOperation::SymbolicOperation(const std::size_t nq, const Qubit target, const OpType g, const SymbolOrNumber& lambda, const SymbolOrNumber& phi, const SymbolOrNumber& theta, const Qubit startingQubit) {
        type = g;
        setup(nq, lambda, phi, theta, startingQubit);
        targets.emplace_back(target);
    }

    SymbolicOperation::SymbolicOperation(const std::size_t nq, const Targets& targ, const OpType g, const SymbolOrNumber& lambda, const SymbolOrNumber& phi, const SymbolOrNumber& theta, const Qubit startingQubit) {
        type = g;
        setup(nq, lambda, phi, theta, startingQubit);
        targets = targ;
    }

    SymbolicOperation::SymbolicOperation(const std::size_t nq, const Control control, const Qubit target, const OpType g, const SymbolOrNumber& lambda, const SymbolOrNumber& phi, const SymbolOrNumber& theta, const Qubit startingQubit):
        SymbolicOperation(nq, target, g, lambda, phi, theta, startingQubit) {
        controls.insert(control);
    }

    SymbolicOperation::SymbolicOperation(const std::size_t nq, const Control control, const Targets& targ, const OpType g, const SymbolOrNumber& lambda, const SymbolOrNumber& phi, const SymbolOrNumber& theta, const Qubit startingQubit):
        SymbolicOperation(nq, targ, g, lambda, phi, theta, startingQubit) {
        controls.insert(control);
    }

    SymbolicOperation::SymbolicOperation(const std::size_t nq, const Controls& c, const Qubit target, const OpType g, const SymbolOrNumber& lambda, const SymbolOrNumber& phi, const SymbolOrNumber& theta, const Qubit startingQubit):
        SymbolicOperation(nq, target, g, lambda, phi, theta, startingQubit) {
        controls = c;
    }

    SymbolicOperation::SymbolicOperation(const std::size_t nq, const Controls& c, const Targets& targ, const OpType g, const SymbolOrNumber& lambda, const SymbolOrNumber& phi, const SymbolOrNumber& theta, const Qubit startingQubit):
        SymbolicOperation(nq, targ, g, lambda, phi, theta, startingQubit) {
        controls = c;
    }

    // MCF (cSWAP), Peres, paramterized two target Constructor
    SymbolicOperation::SymbolicOperation(const std::size_t nq, const Controls& c, const Qubit target0, const Qubit target1, const OpType g, const SymbolOrNumber& lambda, const SymbolOrNumber& phi, const SymbolOrNumber& theta, const Qubit startingQubit):
        SymbolicOperation(nq, c, {target0, target1}, g, lambda, phi, theta, startingQubit) {
    }

    bool SymbolicOperation::equals(const Operation& op, const Permutation& perm1, const Permutation& perm2) const {
        if (!op.isSymbolicOperation() && !isStandardOperation()) {
            return false;
        }
        if (isStandardOperation() && qc::StandardOperation::equals(op, perm1, perm2)) {
            return true;
        }

        if (!op.isSymbolicOperation()) {
            return false;
        }
        const auto& symOp = dynamic_cast<const SymbolicOperation&>(op);
        for (std::size_t i = 0; i < symbolicParameter.size(); ++i) {
            if (symbolicParameter.at(i).has_value() != symOp.symbolicParameter.at(i).has_value()) {
                return false;
            }

            if (symbolicParameter.at(i).has_value()) {
                return symbolicParameter.at(i).value() == symOp.symbolicParameter.at(i).value();
            }
        }
        return true;
    }

    [[noreturn]] void SymbolicOperation::dumpOpenQASM([[maybe_unused]] std::ostream& of, [[maybe_unused]] const RegisterNames& qreg, [[maybe_unused]] const RegisterNames& creg) const {
        throw QFRException("OpenQasm2.0 doesn't support parametrized gates!");
    }

    StandardOperation SymbolicOperation::getInstantiatedOperation(const VariableAssignment& assignment) const {
        auto lambda = getInstantiation(getParameter(0), assignment);
        auto phi    = getInstantiation(getParameter(1), assignment);
        auto theta  = getInstantiation(getParameter(2), assignment);
        return {nqubits, targets, type, lambda, phi, theta, startQubit};
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
} // namespace qc
