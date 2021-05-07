/*
 * This file is part of JKQ QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#include "operations/StandardOperation.hpp"

namespace qc {
    /***
     * Protected Methods
     ***/
    OpType StandardOperation::parseU3(dd::fp& lambda, dd::fp& phi, dd::fp& theta) {
        if (std::abs(theta) < PARAMETER_TOLERANCE && std::abs(phi) < PARAMETER_TOLERANCE) {
            phi   = 0.L;
            theta = 0.L;
            return parseU1(lambda);
        }

        if (std::abs(theta - dd::PI_2) < PARAMETER_TOLERANCE) {
            theta    = dd::PI_2;
            auto res = parseU2(lambda, phi);
            if (res != U2)
                theta = 0.L;
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

        if (std::abs(lambda - dd::PI_2) < PARAMETER_TOLERANCE) {
            lambda = dd::PI_2;
            if (std::abs(phi + dd::PI_2) < PARAMETER_TOLERANCE) {
                phi = 0.L;
                checkInteger(theta);
                checkFractionPi(theta);
                lambda = theta;
                theta  = 0.L;
                return RX;
            }

            if (std::abs(phi - dd::PI_2) < PARAMETER_TOLERANCE) {
                phi = dd::PI_2;
                if (std::abs(theta - dd::PI) < PARAMETER_TOLERANCE) {
                    lambda = 0.L;
                    phi    = 0.L;
                    theta  = 0.L;
                    return Y;
                }
            }
        }

        if (std::abs(lambda - dd::PI) < PARAMETER_TOLERANCE) {
            lambda = dd::PI;
            if (std::abs(phi) < PARAMETER_TOLERANCE) {
                phi = 0.L;
                if (std::abs(theta - dd::PI) < PARAMETER_TOLERANCE) {
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

    OpType StandardOperation::parseU2(dd::fp& lambda, dd::fp& phi) {
        if (std::abs(phi) < PARAMETER_TOLERANCE) {
            phi = 0.L;
            if (std::abs(std::abs(lambda) - dd::PI) < PARAMETER_TOLERANCE) {
                lambda = 0.L;
                return H;
            }
            if (std::abs(lambda) < PARAMETER_TOLERANCE) {
                lambda = dd::PI_2;
                return RY;
            }
        }

        if (std::abs(lambda - dd::PI_2) < PARAMETER_TOLERANCE) {
            lambda = dd::PI_2;
            if (std::abs(phi + dd::PI_2) < PARAMETER_TOLERANCE) {
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

    OpType StandardOperation::parseU1(dd::fp& lambda) {
        if (std::abs(lambda) < PARAMETER_TOLERANCE) {
            lambda = 0.L;
            return I;
        }
        bool sign = std::signbit(lambda);

        if (std::abs(std::abs(lambda) - dd::PI) < PARAMETER_TOLERANCE) {
            lambda = 0.L;
            return Z;
        }

        if (std::abs(std::abs(lambda) - dd::PI_2) < PARAMETER_TOLERANCE) {
            lambda = 0.L;
            return sign ? Sdag : S;
        }

        if (std::abs(std::abs(lambda) - dd::PI_4) < PARAMETER_TOLERANCE) {
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

    void StandardOperation::setup(dd::QubitCount nq, dd::fp par0, dd::fp par1, dd::fp par2, dd::Qubit startingQubit) {
        nqubits      = nq;
        parameter[0] = par0;
        parameter[1] = par1;
        parameter[2] = par2;
        startQubit   = startingQubit;
        checkUgate();
        setName();
    }

    MatrixDD StandardOperation::getStandardOperationDD(std::unique_ptr<dd::Package>& dd, const dd::Controls& controls, dd::Qubit target, bool inverse) const {
        MatrixDD       e{};
        dd::GateMatrix gm;

        switch (type) {
            case I: gm = dd::Imat; break;
            case H: gm = dd::Hmat; break;
            case X:
                if (controls.size() > 1) { // Toffoli
                    e = dd->toffoliTable.lookup(nqubits, controls, targets[0]);
                    if (e.p == nullptr) {
                        e = dd->makeGateDD(dd::Xmat, nqubits, controls, targets[0], startQubit);
                        dd->toffoliTable.insert(nqubits, controls, targets[0], e);
                    }
                    return e;
                }
                gm = dd::Xmat;
                break;
            case Y: gm = dd::Ymat; break;
            case Z: gm = dd::Zmat; break;
            case S: gm = inverse ? dd::Sdagmat : dd::Smat; break;
            case Sdag: gm = inverse ? dd::Smat : dd::Sdagmat; break;
            case T: gm = inverse ? dd::Tdagmat : dd::Tmat; break;
            case Tdag: gm = inverse ? dd::Tmat : dd::Tdagmat; break;
            case V: gm = inverse ? dd::Vdagmat : dd::Vmat; break;
            case Vdag: gm = inverse ? dd::Vmat : dd::Vdagmat; break;
            case U3: gm = inverse ? dd::U3mat(-parameter[1], -parameter[0], -parameter[2]) : dd::U3mat(parameter[0], parameter[1], parameter[2]); break;
            case U2: gm = inverse ? dd::U2mat(-parameter[1] + dd::PI, -parameter[0] - dd::PI) : dd::U2mat(parameter[0], parameter[1]); break;
            case Phase: gm = inverse ? dd::Phasemat(-parameter[0]) : dd::Phasemat(parameter[0]); break;
            case SX: gm = inverse ? dd::SXdagmat : dd::SXmat; break;
            case SXdag: gm = inverse ? dd::SXmat : dd::SXdagmat; break;
            case RX: gm = inverse ? dd::RXmat(-parameter[0]) : dd::RXmat(parameter[0]); break;
            case RY: gm = inverse ? dd::RYmat(-parameter[0]) : dd::RYmat(parameter[0]); break;
            case RZ: gm = inverse ? dd::RZmat(-parameter[0]) : dd::RZmat(parameter[0]); break;
            default:
                std::ostringstream oss{};
                oss << "DD for gate" << name << " not available!";
                throw QFRException(oss.str());
        }
        return dd->makeGateDD(gm, nqubits, controls, target, startQubit);
    }

    MatrixDD StandardOperation::getStandardOperationDD(std::unique_ptr<dd::Package>& dd, const dd::Controls& controls, dd::Qubit target0, dd::Qubit target1, bool inverse) const {
        switch (type) {
            case SWAP:
                return dd->makeSWAPDD(nqubits, controls, target0, target1, startQubit);
            case iSWAP:
                if (inverse) {
                    return dd->makeiSWAPinvDD(nqubits, controls, target0, target1, startQubit);
                } else {
                    return dd->makeiSWAPDD(nqubits, controls, target0, target1, startQubit);
                }
            case Peres:
                if (inverse) {
                    return dd->makePeresdagDD(nqubits, controls, target0, target1, startQubit);
                } else {
                    return dd->makePeresDD(nqubits, controls, target0, target1, startQubit);
                }
            case Peresdag:
                if (inverse) {
                    return dd->makePeresDD(nqubits, controls, target0, target1, startQubit);
                } else {
                    return dd->makePeresdagDD(nqubits, controls, target0, target1, startQubit);
                }
            default:
                std::ostringstream oss{};
                oss << "DD for gate" << name << " not available!";
                throw QFRException(oss.str());
        }
    }

    /***
     * Constructors
     ***/
    StandardOperation::StandardOperation(dd::QubitCount nq, dd::Qubit target, OpType g, dd::fp lambda, dd::fp phi, dd::fp theta, dd::Qubit startingQubit) {
        type = g;
        setup(nq, lambda, phi, theta, startingQubit);
        targets.emplace_back(target);
    }

    StandardOperation::StandardOperation(dd::QubitCount nq, const Targets& targets, OpType g, dd::fp lambda, dd::fp phi, dd::fp theta, dd::Qubit startingQubit) {
        type = g;
        setup(nq, lambda, phi, theta, startingQubit);
        this->targets = targets;
    }

    StandardOperation::StandardOperation(dd::QubitCount nq, dd::Control control, dd::Qubit target, OpType g, dd::fp lambda, dd::fp phi, dd::fp theta, dd::Qubit startingQubit):
        StandardOperation(nq, target, g, lambda, phi, theta, startingQubit) {
        controls.insert(control);
    }

    StandardOperation::StandardOperation(dd::QubitCount nq, dd::Control control, const Targets& targets, OpType g, dd::fp lambda, dd::fp phi, dd::fp theta, dd::Qubit startingQubit):
        StandardOperation(nq, targets, g, lambda, phi, theta, startingQubit) {
        controls.insert(control);
    }

    StandardOperation::StandardOperation(dd::QubitCount nq, const dd::Controls& controls, dd::Qubit target, OpType g, dd::fp lambda, dd::fp phi, dd::fp theta, dd::Qubit startingQubit):
        StandardOperation(nq, target, g, lambda, phi, theta, startingQubit) {
        this->controls = controls;
    }

    StandardOperation::StandardOperation(dd::QubitCount nq, const dd::Controls& controls, const Targets& targets, OpType g, dd::fp lambda, dd::fp phi, dd::fp theta, dd::Qubit startingQubit):
        StandardOperation(nq, targets, g, lambda, phi, theta, startingQubit) {
        this->controls = controls;
    }

    // MCT Constructor
    StandardOperation::StandardOperation(dd::QubitCount nq, const dd::Controls& controls, dd::Qubit target, dd::Qubit startingQubit):
        StandardOperation(nq, controls, target, X, 0., 0., 0., startingQubit) {
    }

    // MCF (cSWAP), Peres, paramterized two target Constructor
    StandardOperation::StandardOperation(dd::QubitCount nq, const dd::Controls& controls, dd::Qubit target0, dd::Qubit target1, OpType g, dd::fp lambda, dd::fp phi, dd::fp theta, dd::Qubit startingQubit):
        StandardOperation(nq, controls, {target0, target1}, g, lambda, phi, theta, startingQubit) {
    }

    /***
     * Public Methods
    ***/
    void StandardOperation::dumpOpenQASM(std::ostream& of, const RegisterNames& qreg, [[maybe_unused]] const RegisterNames& creg) const {
        std::ostringstream op;
        op << std::setprecision(std::numeric_limits<dd::fp>::digits10);
        if ((controls.size() > 1 && type != X) || controls.size() > 2) {
            std::cout << "[WARNING] Multiple controlled gates are not natively supported by OpenQASM. "
                      << "However, this library can parse .qasm files with multiple controlled gates (e.g., cccx) correctly. "
                      << "Thus, while not valid vanilla OpenQASM, the dumped file will work with this library. " << std::endl;
        }

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
                op << "u(pi/2, -pi/2, pi/2)";
                break;
            case Vdag:
                op << "u(pi/2, pi/2, -pi/2)";
                break;
            case U3:
                op << "u(" << parameter[2] << "," << parameter[1] << "," << parameter[0] << ")";
                break;
            case U2:
                op << "u(pi/2, " << parameter[1] << "," << parameter[0] << ")";
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
                for (const auto& c: controls) {
                    if (c.type == dd::Control::Type::neg)
                        of << "x " << qreg[c.qubit].second << ";" << std::endl;
                }

                of << op.str() << "swap";
                for (const auto& c: controls)
                    of << " " << qreg[c.qubit].second << ",";
                of << " " << qreg[targets[0]].second << ", " << qreg[targets[1]].second << ";" << std::endl;

                for (const auto& c: controls) {
                    if (c.type == dd::Control::Type::neg)
                        of << "x " << qreg[c.qubit].second << ";" << std::endl;
                }
                return;
            case iSWAP:
                for (const auto& c: controls) {
                    if (c.type == dd::Control::Type::neg)
                        of << "x " << qreg[c.qubit].second << ";" << std::endl;
                }
                of << op.str() << "swap";
                for (const auto& c: controls)
                    of << " " << qreg[c.qubit].second << ",";
                of << " " << qreg[targets[0]].second << ", " << qreg[targets[1]].second << ";" << std::endl;

                of << op.str() << "s";
                for (const auto& c: controls)
                    of << " " << qreg[c.qubit].second << ",";
                of << " " << qreg[targets[0]].second << ";" << std::endl;

                of << op.str() << "s";
                for (const auto& c: controls)
                    of << " " << qreg[c.qubit].second << ",";
                of << " " << qreg[targets[1]].second << ";" << std::endl;

                of << op.str() << "cz";
                for (const auto& c: controls)
                    of << " " << qreg[c.qubit].second << ",";
                of << qreg[targets[0]].second << ", " << qreg[targets[1]].second << ";" << std::endl;

                for (const auto& c: controls) {
                    if (c.type == dd::Control::Type::neg)
                        of << "x " << qreg[c.qubit].second << ";" << std::endl;
                }
                return;
            case Peres:
                of << op.str() << "cx";
                for (const auto& c: controls)
                    of << " " << qreg[c.qubit].second << ",";
                of << qreg[targets[1]].second << ", " << qreg[targets[0]].second << ";" << std::endl;

                of << op.str() << "x";
                for (const auto& c: controls)
                    of << " " << qreg[c.qubit].second << ",";
                of << qreg[targets[1]].second << ";" << std::endl;
                return;
            case Peresdag:
                of << op.str() << "x";
                for (const auto& c: controls)
                    of << " " << qreg[c.qubit].second << ",";
                of << qreg[targets[1]].second << ";" << std::endl;

                of << op.str() << "cx";
                for (const auto& c: controls)
                    of << " " << qreg[c.qubit].second << ",";
                of << qreg[targets[1]].second << ", " << qreg[targets[0]].second << ";" << std::endl;
                return;
            case Teleportation:
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
                   << qreg[targets[2]].second << ";"
                   << std::endl;

                return;
            default:
                std::cerr << "gate type (index) " << static_cast<int>(type) << " could not be converted to OpenQASM" << std::endl;
        }

        for (const auto& c: controls) {
            if (c.type == dd::Control::Type::neg)
                of << "x " << qreg[c.qubit].second << ";" << std::endl;
        }
        of << op.str();
        for (const auto& c: controls) {
            of << " " << qreg[c.qubit].second << ",";
        }
        for (const auto& target: targets) {
            of << " " << qreg[target].second << ";" << std::endl;
        }
        for (const auto& c: controls) {
            if (c.type == dd::Control::Type::neg)
                of << "x " << qreg[c.qubit].second << ";" << std::endl;
        }
    }

    void StandardOperation::dumpQiskit(std::ostream& of, const RegisterNames& qreg, [[maybe_unused]] const RegisterNames& creg, const char* anc_reg_name) const {
        std::ostringstream op;
        if (targets.size() > 2 || (targets.size() > 1 && type != SWAP && type != iSWAP && type != Peres && type != Peresdag)) {
            std::cerr << "Multiple targets are not supported in general at the moment" << std::endl;
        }
        switch (type) {
            case I:
                op << "qc.iden(";
                break;
            case H:
                switch (controls.size()) {
                    case 0:
                        op << "qc.h(";
                        break;
                    case 1:
                        op << "qc.ch(" << qreg[controls.begin()->qubit].second << ", ";
                        break;
                    default:
                        std::cerr << "Multi-controlled H gate currently not supported" << std::endl;
                }
                break;
            case X:
                switch (controls.size()) {
                    case 0:
                        op << "qc.x(";
                        break;
                    case 1:
                        op << "qc.cx(" << qreg[controls.begin()->qubit].second << ", ";
                        break;
                    case 2:
                        op << "qc.ccx(" << qreg[controls.begin()->qubit].second << ", " << qreg[(++controls.begin())->qubit].second << ", ";
                        break;
                    default:
                        op << "qc.mct([";
                        for (const auto& control: controls) {
                            op << qreg[control.qubit].second << ", ";
                        }
                        op << "], " << qreg[targets[0]].second << ", " << anc_reg_name << ", mode='basic')" << std::endl;
                        of << op.str();
                        return;
                }
                break;
            case Y:
                switch (controls.size()) {
                    case 0:
                        op << "qc.y(";
                        break;
                    case 1:
                        op << "qc.cy(" << qreg[controls.begin()->qubit].second << ", ";
                        break;
                    default:
                        std::cerr << "Multi-controlled Y gate currently not supported" << std::endl;
                }
                break;
            case Z:
                if (!controls.empty()) {
                    op << "qc.mcu1(pi, [";
                    for (const auto& control: controls) {
                        op << qreg[control.qubit].second << ", ";
                    }
                    op << "], ";
                } else {
                    op << "qc.z(";
                }
                break;
            case S:
                if (!controls.empty()) {
                    op << "qc.mcu1(pi/2, [";
                    for (const auto& control: controls) {
                        op << qreg[control.qubit].second << ", ";
                    }
                    op << "], ";
                } else {
                    op << "qc.s(";
                }
                break;
            case Sdag:
                if (!controls.empty()) {
                    op << "qc.mcu1(-pi/2, [";
                    for (const auto& control: controls) {
                        op << qreg[control.qubit].second << ", ";
                    }
                    op << "], ";
                } else {
                    op << "qc.sdg(";
                }
                break;
            case T:
                if (!controls.empty()) {
                    op << "qc.mcu1(pi/4, [";
                    for (const auto& control: controls) {
                        op << qreg[control.qubit].second << ", ";
                    }
                    op << "], ";
                } else {
                    op << "qc.t(";
                }
                break;
            case Tdag:
                if (!controls.empty()) {
                    op << "qc.mcu1(-pi/4, [";
                    for (const auto& control: controls) {
                        op << qreg[control.qubit].second << ", ";
                    }
                    op << "], ";
                } else {
                    op << "qc.tdg(";
                }
                break;
            case V:
                switch (controls.size()) {
                    case 0:
                        op << "qc.u3(pi/2, -pi/2, pi/2, ";
                        break;
                    case 1:
                        op << "qc.cu3(pi/2, -pi/2, pi/2, " << qreg[controls.begin()->qubit].second << ", ";
                        break;
                    default:
                        std::cerr << "Multi-controlled V gate currently not supported" << std::endl;
                }
                break;
            case Vdag:
                switch (controls.size()) {
                    case 0:
                        op << "qc.u3(pi/2, pi/2, -pi/2, ";
                        break;
                    case 1:
                        op << "qc.cu3(pi/2, pi/2, -pi/2, " << qreg[controls.begin()->qubit].second << ", ";
                        break;
                    default:
                        std::cerr << "Multi-controlled Vdag gate currently not supported" << std::endl;
                }
                break;
            case U3:
                switch (controls.size()) {
                    case 0:
                        op << "qc.u3(" << parameter[2] << ", " << parameter[1] << ", " << parameter[0] << ", ";
                        break;
                    case 1:
                        op << "qc.cu3(" << parameter[2] << ", " << parameter[1] << ", " << parameter[0] << ", " << qreg[controls.begin()->qubit].second << ", ";
                        break;
                    default:
                        std::cerr << "Multi-controlled U3 gate currently not supported" << std::endl;
                }
                break;
            case U2:
                switch (controls.size()) {
                    case 0:
                        op << "qc.u3(pi/2, " << parameter[1] << ", " << parameter[0] << ", ";
                        break;
                    case 1:
                        op << "qc.cu3(pi/2, " << parameter[1] << ", " << parameter[0] << ", " << qreg[controls.begin()->qubit].second << ", ";
                        break;
                    default:
                        std::cerr << "Multi-controlled U2 gate currently not supported" << std::endl;
                }
                break;
            case Phase:
                if (!controls.empty()) {
                    op << "qc.mcu1(" << parameter[0] << ", [";
                    for (const auto& control: controls) {
                        op << qreg[control.qubit].second << ", ";
                    }
                    op << "], ";
                } else {
                    op << "qc.u1(" << parameter[0] << ", ";
                }
                break;
            case RX:
                if (!controls.empty()) {
                    op << "qc.mcrx(" << parameter[0] << ", [";
                    for (const auto& control: controls) {
                        op << qreg[control.qubit].second << ", ";
                    }
                    op << "], ";
                } else {
                    op << "qc.rx(" << parameter[0] << ", ";
                }
                break;
            case RY:
                if (!controls.empty()) {
                    op << "qc.mcry(" << parameter[0] << ", [";
                    for (const auto& control: controls) {
                        op << qreg[control.qubit].second << ", ";
                    }
                    op << "], ";
                } else {
                    op << "qc.ry(" << parameter[0] << ", ";
                }
                break;
            case RZ:
                if (!controls.empty()) {
                    op << "qc.mcrz(" << parameter[0] << ", [";
                    for (const auto& control: controls) {
                        op << qreg[control.qubit].second << ", ";
                    }
                    op << "], ";
                } else {
                    op << "qc.rz(" << parameter[0] << ", ";
                }
                break;
            case SWAP:
                switch (controls.size()) {
                    case 0:
                        of << "qc.swap(" << qreg[targets[0]].second << ", " << qreg[targets[1]].second << ")" << std::endl;
                        break;
                    case 1:
                        of << "qc.cswap(" << qreg[controls.begin()->qubit].second << ", " << qreg[targets[0]].second << ", " << qreg[targets[1]].second << ")" << std::endl;
                        break;
                    default:
                        of << "qc.cx(" << qreg[targets[1]].second << ", " << qreg[targets[0]].second << ")" << std::endl;
                        of << "qc.mct([";
                        for (const auto& control: controls) {
                            of << qreg[control.qubit].second << ", ";
                        }
                        of << qreg[targets[0]].second << "], " << qreg[targets[1]].second << ", " << anc_reg_name << ", mode='basic')" << std::endl;
                        of << "qc.cx(" << qreg[targets[1]].second << ", " << qreg[targets[0]].second << ")" << std::endl;
                        break;
                }
                return;
            case iSWAP:
                switch (controls.size()) {
                    case 0:
                        of << "qc.swap(" << qreg[targets[0]].second << ", " << qreg[targets[1]].second << ")" << std::endl;
                        of << "qc.s(" << qreg[targets[0]].second << ")" << std::endl;
                        of << "qc.s(" << qreg[targets[1]].second << ")" << std::endl;
                        of << "qc.cz(" << qreg[targets[0]].second << ", " << qreg[targets[1]].second << ")" << std::endl;
                        break;
                    case 1:
                        of << "qc.cswap(" << qreg[controls.begin()->qubit].second << ", " << qreg[targets[0]].second << ", " << qreg[targets[1]].second << ")" << std::endl;
                        of << "qc.cu1(pi/2, " << qreg[controls.begin()->qubit].second << ", " << qreg[targets[0]].second << ")" << std::endl;
                        of << "qc.cu1(pi/2, " << qreg[controls.begin()->qubit].second << ", " << qreg[targets[1]].second << ")" << std::endl;
                        of << "qc.mcu1(pi, [" << qreg[controls.begin()->qubit].second << ", " << qreg[targets[0]].second << "], " << qreg[targets[1]].second << ")" << std::endl;
                        break;
                    default:
                        std::cerr << "Multi-controlled iSWAP gate currently not supported" << std::endl;
                }
                return;
            case Peres:
                of << "qc.ccx(" << qreg[controls.begin()->qubit].second << ", " << qreg[targets[1]].second << ", " << qreg[targets[0]].second << ")" << std::endl;
                of << "qc.cx(" << qreg[controls.begin()->qubit].second << ", " << qreg[targets[1]].second << ")" << std::endl;
                return;
            case Peresdag:
                of << "qc.cx(" << qreg[controls.begin()->qubit].second << ", " << qreg[targets[1]].second << ")" << std::endl;
                of << "qc.ccx(" << qreg[controls.begin()->qubit].second << ", " << qreg[targets[1]].second << ", " << qreg[targets[0]].second << ")" << std::endl;
                return;
            default:
                std::cerr << "gate type (index) " << static_cast<int>(type) << " could not be converted to qiskit" << std::endl;
        }
        of << op.str() << qreg[targets[0]].second << ")" << std::endl;
    }

    MatrixDD StandardOperation::getDD(std::unique_ptr<dd::Package>& dd, Permutation& permutation) const {
        if (type == SWAP && controls.empty()) {
            auto target0 = targets.at(0);
            auto target1 = targets.at(1);
            // update permutation
            std::swap(permutation.at(target0), permutation.at(target1));
            return dd->makeIdent(nqubits);
        }
        return Operation::getDD(dd, permutation);
    }

    MatrixDD StandardOperation::getInverseDD(std::unique_ptr<dd::Package>& dd, Permutation& permutation) const {
        if (type == SWAP && controls.empty()) {
            auto target0 = targets.at(0);
            auto target1 = targets.at(1);
            // update permutation
            std::swap(permutation.at(target0), permutation.at(target1));
            return dd->makeIdent(nqubits);
        }
        return Operation::getInverseDD(dd, permutation);
    }
} // namespace qc
