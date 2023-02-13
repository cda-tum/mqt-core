/*
 * This file is part of MQT QFR library which is released under the MIT license.
 * See file README.md or go to https://www.cda.cit.tum.de/research/quantum/ for more information.
 */

#include "operations/StandardOperation.hpp"

#include <sstream>
#include <variant>

namespace qc {
    /***
     * Protected Methods
     ***/
    OpType StandardOperation::parseU3(fp& lambda, fp& phi, fp& theta) {
        if (std::abs(theta) < PARAMETER_TOLERANCE && std::abs(phi) < PARAMETER_TOLERANCE) {
            phi   = 0.L;
            theta = 0.L;
            return parseU1(lambda);
        }

        if (std::abs(theta - PI_2) < PARAMETER_TOLERANCE) {
            theta    = PI_2;
            auto res = parseU2(lambda, phi);
            if (res != U2) {
                theta = 0.L;
            }
            return res;
        }

        if (std::abs(lambda) < PARAMETER_TOLERANCE) {
            lambda = 0.L;
            if (std::abs(phi) < PARAMETER_TOLERANCE) {
                phi = 0.L;
                checkInteger(theta);
                checkFractionPi(theta);
                lambda = theta;
                theta  = 0.L;
                return RY;
            }
        }

        if (std::abs(lambda - PI_2) < PARAMETER_TOLERANCE) {
            lambda = PI_2;
            if (std::abs(phi + PI_2) < PARAMETER_TOLERANCE) {
                phi = 0.L;
                checkInteger(theta);
                checkFractionPi(theta);
                lambda = theta;
                theta  = 0.L;
                return RX;
            }

            if (std::abs(phi - PI_2) < PARAMETER_TOLERANCE) {
                phi = PI_2;
                if (std::abs(theta - PI) < PARAMETER_TOLERANCE) {
                    lambda = 0.L;
                    phi    = 0.L;
                    theta  = 0.L;
                    return Y;
                }
            }
        }

        if (std::abs(lambda - PI) < PARAMETER_TOLERANCE) {
            lambda = PI;
            if (std::abs(phi) < PARAMETER_TOLERANCE) {
                phi = 0.L;
                if (std::abs(theta - PI) < PARAMETER_TOLERANCE) {
                    theta  = 0.L;
                    lambda = 0.L;
                    return X;
                }
            }
        }

        // parse a real u3 gate
        checkInteger(lambda);
        checkFractionPi(lambda);
        checkInteger(phi);
        checkFractionPi(phi);
        checkInteger(theta);
        checkFractionPi(theta);

        return U3;
    }

    OpType StandardOperation::parseU2(fp& lambda, fp& phi) {
        if (std::abs(phi) < PARAMETER_TOLERANCE) {
            phi = 0.L;
            if (std::abs(std::abs(lambda) - PI) < PARAMETER_TOLERANCE) {
                lambda = 0.L;
                return H;
            }
            if (std::abs(lambda) < PARAMETER_TOLERANCE) {
                lambda = PI_2;
                return RY;
            }
        }

        if (std::abs(lambda - PI_2) < PARAMETER_TOLERANCE) {
            lambda = PI_2;
            if (std::abs(phi + PI_2) < PARAMETER_TOLERANCE) {
                phi = 0.L;
                return RX;
            }
        }

        checkInteger(lambda);
        checkFractionPi(lambda);
        checkInteger(phi);
        checkFractionPi(phi);

        return U2;
    }

    OpType StandardOperation::parseU1(fp& lambda) {
        if (std::abs(lambda) < PARAMETER_TOLERANCE) {
            lambda = 0.L;
            return I;
        }
        const bool sign = std::signbit(lambda);

        if (std::abs(std::abs(lambda) - PI) < PARAMETER_TOLERANCE) {
            lambda = 0.L;
            return Z;
        }

        if (std::abs(std::abs(lambda) - PI_2) < PARAMETER_TOLERANCE) {
            lambda = 0.L;
            return sign ? Sdag : S;
        }

        if (std::abs(std::abs(lambda) - PI_4) < PARAMETER_TOLERANCE) {
            lambda = 0.L;
            return sign ? Tdag : T;
        }

        checkInteger(lambda);
        checkFractionPi(lambda);

        return Phase;
    }

    void StandardOperation::checkUgate() {
        if (type == Phase) {
            type = parseU1(parameter[0]);
        } else if (type == U2) {
            type = parseU2(parameter[0], parameter[1]);
        } else if (type == U3) {
            type = parseU3(parameter[0], parameter[1], parameter[2]);
        }
    }

    void StandardOperation::setup(const std::size_t nq, const fp par0, const fp par1, const fp par2, const Qubit startingQubit) {
        nqubits      = nq;
        parameter[0] = par0;
        parameter[1] = par1;
        parameter[2] = par2;
        startQubit   = startingQubit;
        checkUgate();
        setName();
    }

    /***
     * Constructors
     ***/
    StandardOperation::StandardOperation(const std::size_t nq, const Qubit target, const OpType g, const fp lambda, const fp phi, const fp theta, const Qubit startingQubit) {
        type = g;
        setup(nq, lambda, phi, theta, startingQubit);
        targets.emplace_back(target);
    }

    StandardOperation::StandardOperation(const std::size_t nq, const Targets& targ, const OpType g, const fp lambda, const fp phi, const fp theta, const Qubit startingQubit) {
        type = g;
        setup(nq, lambda, phi, theta, startingQubit);
        targets = targ;
    }

    StandardOperation::StandardOperation(const std::size_t nq, const Control control, const Qubit target, const OpType g, const fp lambda, const fp phi, const fp theta, const Qubit startingQubit):
        StandardOperation(nq, target, g, lambda, phi, theta, startingQubit) {
        controls.insert(control);
    }

    StandardOperation::StandardOperation(const std::size_t nq, const Control control, const Targets& targ, const OpType g, const fp lambda, const fp phi, const fp theta, const Qubit startingQubit):
        StandardOperation(nq, targ, g, lambda, phi, theta, startingQubit) {
        controls.insert(control);
    }

    StandardOperation::StandardOperation(const std::size_t nq, const Controls& c, const Qubit target, const OpType g, const fp lambda, const fp phi, const fp theta, const Qubit startingQubit):
        StandardOperation(nq, target, g, lambda, phi, theta, startingQubit) {
        controls = c;
    }

    StandardOperation::StandardOperation(const std::size_t nq, const Controls& c, const Targets& targ, const OpType g, const fp lambda, const fp phi, const fp theta, const Qubit startingQubit):
        StandardOperation(nq, targ, g, lambda, phi, theta, startingQubit) {
        controls = c;
    }

    // MCT Constructor
    StandardOperation::StandardOperation(const std::size_t nq, const Controls& c, const Qubit target, const Qubit startingQubit):
        StandardOperation(nq, c, target, X, 0., 0., 0., startingQubit) {
    }

    // MCF (cSWAP), Peres, paramterized two target Constructor
    StandardOperation::StandardOperation(const std::size_t nq, const Controls& c, const Qubit target0, const Qubit target1, const OpType g, const fp lambda, const fp phi, const fp theta, const Qubit startingQubit):
        StandardOperation(nq, c, {target0, target1}, g, lambda, phi, theta, startingQubit) {
    }

    /***
     * Public Methods
    ***/
    void StandardOperation::dumpOpenQASM(std::ostream& of, const RegisterNames& qreg, [[maybe_unused]] const RegisterNames& creg) const {
        std::ostringstream op;
        op << std::setprecision(std::numeric_limits<fp>::digits10);
        if ((controls.size() > 1 && type != X) || controls.size() > 2) {
            std::cout << "[WARNING] Multiple controlled gates are not natively supported by OpenQASM. "
                      << "However, this library can parse .qasm files with multiple controlled gates (e.g., cccx) correctly. "
                      << "Thus, while not valid vanilla OpenQASM, the dumped file will work with this library. " << std::endl;
        }

        // safe the numbers of controls as a prefix to the operation name
        op << std::string(controls.size(), 'c');

        switch (type) {
            case I:
                op << "id";
                break;
            case H:
                op << "h";
                break;
            case X:
                op << "x";
                break;
            case Y:
                op << "y";
                break;
            case Z:
                op << "z";
                break;
            case S:
                if (!controls.empty()) {
                    op << "p(pi/2)";
                } else {
                    op << "s";
                }
                break;
            case Sdag:
                if (!controls.empty()) {
                    op << "p(-pi/2)";
                } else {
                    op << "sdg";
                }
                break;
            case T:
                if (!controls.empty()) {
                    op << "p(pi/4)";
                } else {
                    op << "t";
                }
                break;
            case Tdag:
                if (!controls.empty()) {
                    op << "p(-pi/4)";
                } else {
                    op << "tdg";
                }
                break;
            case V:
                op << "u3(pi/2, -pi/2, pi/2)";
                break;
            case Vdag:
                op << "u3(pi/2, pi/2, -pi/2)";
                break;
            case U3:
                op << "u3(" << parameter[2] << "," << parameter[1] << "," << parameter[0] << ")";
                break;
            case U2:
                op << "u3(pi/2, " << parameter[1] << "," << parameter[0] << ")";
                break;
            case Phase:
                op << "p(" << parameter[0] << ")";
                break;
            case SX:
                op << "sx";
                break;
            case SXdag:
                op << "sxdg";
                break;
            case RX:
                op << "rx(" << parameter[0] << ")";
                break;
            case RY:
                op << "ry(" << parameter[0] << ")";
                break;
            case RZ:
                op << "rz(" << parameter[0] << ")";
                break;
            case SWAP:
                dumpOpenQASMSwap(of, qreg);
                return;
            case iSWAP:
                dumpOpenQASMiSwap(of, qreg);
                return;
            case Peres:
                of << op.str() << "cx";
                for (const auto& c: controls) {
                    of << " " << qreg[c.qubit].second << ",";
                }
                of << " " << qreg[targets[1]].second << ", " << qreg[targets[0]].second << ";\n";

                of << op.str() << "x";
                for (const auto& c: controls) {
                    of << " " << qreg[c.qubit].second << ",";
                }
                of << " " << qreg[targets[1]].second << ";\n";
                return;
            case Peresdag:
                of << op.str() << "x";
                for (const auto& c: controls) {
                    of << " " << qreg[c.qubit].second << ",";
                }
                of << " " << qreg[targets[1]].second << ";\n";

                of << op.str() << "cx";
                for (const auto& c: controls) {
                    of << " " << qreg[c.qubit].second << ",";
                }
                of << " " << qreg[targets[1]].second << ", " << qreg[targets[0]].second << ";\n";
                return;
            case Teleportation:
                dumpOpenQASMTeleportation(of, qreg);
                return;
            default:
                std::cerr << "gate type (index) " << static_cast<int>(type) << " could not be converted to OpenQASM" << std::endl;
        }

        // apply X operations to negate the respective controls
        for (const auto& c: controls) {
            if (c.type == Control::Type::Neg) {
                of << "x " << qreg[c.qubit].second << ";\n";
            }
        }
        // apply the operation
        of << op.str();
        // add controls and targets of the operation
        for (const auto& c: controls) {
            of << " " << qreg[c.qubit].second << ",";
        }
        for (const auto& target: targets) {
            of << " " << qreg[target].second << ";\n";
        }
        // apply X operations to negate the respective controls again
        for (const auto& c: controls) {
            if (c.type == Control::Type::Neg) {
                of << "x " << qreg[c.qubit].second << ";\n";
            }
        }
    }

    void StandardOperation::dumpOpenQASMSwap(std::ostream& of, const RegisterNames& qreg) const {
        for (const auto& c: controls) {
            if (c.type == Control::Type::Neg) {
                of << "x " << qreg[c.qubit].second << ";\n";
            }
        }

        of << std::string(controls.size(), 'c') << "swap";
        for (const auto& c: controls) {
            of << " " << qreg[c.qubit].second << ",";
        }
        of << " " << qreg[targets[0]].second << ", " << qreg[targets[1]].second << ";\n";

        for (const auto& c: controls) {
            if (c.type == Control::Type::Neg) {
                of << "x " << qreg[c.qubit].second << ";\n";
            }
        }
    }

    void StandardOperation::dumpOpenQASMiSwap(std::ostream& of, const RegisterNames& qreg) const {
        const auto ctrlString = std::string(controls.size(), 'c');
        for (const auto& c: controls) {
            if (c.type == Control::Type::Neg) {
                of << "x " << qreg[c.qubit].second << ";\n";
            }
        }
        of << ctrlString << "swap";
        for (const auto& c: controls) {
            of << " " << qreg[c.qubit].second << ",";
        }
        of << " " << qreg[targets[0]].second << ", " << qreg[targets[1]].second << ";\n";

        of << ctrlString << "s";
        for (const auto& c: controls) {
            of << " " << qreg[c.qubit].second << ",";
        }
        of << " " << qreg[targets[0]].second << ";\n";

        of << ctrlString << "s";
        for (const auto& c: controls) {
            of << " " << qreg[c.qubit].second << ",";
        }
        of << " " << qreg[targets[1]].second << ";\n";

        of << ctrlString << "cz";
        for (const auto& c: controls) {
            of << " " << qreg[c.qubit].second << ",";
        }
        of << " " << qreg[targets[0]].second << ", " << qreg[targets[1]].second << ";\n";

        for (const auto& c: controls) {
            if (c.type == Control::Type::Neg) {
                of << "x " << qreg[c.qubit].second << ";\n";
            }
        }
    }

    void StandardOperation::dumpOpenQASMTeleportation(std::ostream& of, const RegisterNames& qreg) const {
        if (!controls.empty() || targets.size() != 3) {
            std::cerr << "controls = ";
            for (const auto& c: controls) {
                std::cerr << qreg.at(c.qubit).second << " ";
            }
            std::cerr << "\ntargets = ";
            for (const auto& t: targets) {
                std::cerr << qreg.at(t).second << " ";
            }
            std::cerr << "\n";

            throw QFRException("Teleportation needs three targets");
        }
        /*
                                            ░      ┌───┐ ░ ┌─┐    ░
                        |ψ⟩ q_0: ───────────░───■──┤ H ├─░─┤M├────░─────────────── |0⟩ or |1⟩
                                 ┌───┐      ░ ┌─┴─┐└───┘ ░ └╥┘┌─┐ ░
                        |0⟩ a_0: ┤ H ├──■───░─┤ X ├──────░──╫─┤M├─░─────────────── |0⟩ or |1⟩
                                 └───┘┌─┴─┐ ░ └───┘      ░  ║ └╥┘ ░  ┌───┐  ┌───┐
                        |0⟩ a_1: ─────┤ X ├─░────────────░──╫──╫──░──┤ X ├──┤ Z ├─ |ψ⟩
                                      └───┘ ░            ░  ║  ║  ░  └─┬─┘  └─┬─┘
                                                            ║  ║    ┌──┴──┐   │
                      bitflip: 1/═══════════════════════════╩══╬════╡ = 1 ╞═══╪═══
                                                            0  ║    └─────┘┌──┴──┐
                    phaseflip: 1/══════════════════════════════╩═══════════╡ = 1 ╞
                                                               0           └─────┘
                */
        of << "// teleport q_0, a_0, a_1; q_0 --> a_1  via a_0\n";
        of << "teleport "
           << qreg[targets[0]].second << ", "
           << qreg[targets[1]].second << ", "
           << qreg[targets[2]].second << ";\n";
    }
} // namespace qc
