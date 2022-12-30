/*
 * This file is part of MQT QFR library which is released under the MIT license.
 * See file README.md or go to https://www.cda.cit.tum.de/research/quantum/ for more information.
 */

#include "operations/Operation.hpp"

namespace qc {
    void Operation::setName() {
        switch (type) {
            case I:
                name = "I   ";
                break;
            case H:
                name = "H   ";
                break;
            case X:
                name = "X   ";
                break;
            case Y:
                name = "Y   ";
                break;
            case Z:
                name = "Z   ";
                break;
            case S:
                name = "S   ";
                break;
            case Sdag:
                name = "Sdag";
                break;
            case T:
                name = "T   ";
                break;
            case Tdag:
                name = "Tdag";
                break;
            case V:
                name = "V   ";
                break;
            case Vdag:
                name = "Vdag";
                break;
            case U3:
                name = "U   ";
                break;
            case U2:
                name = "U2  ";
                break;
            case Phase:
                name = "P   ";
                break;
            case SX:
                name = "SX  ";
                break;
            case SXdag:
                name = "SXdg";
                break;
            case RX:
                name = "RX  ";
                break;
            case RY:
                name = "RY  ";
                break;
            case RZ:
                name = "RZ  ";
                break;
            case SWAP:
                name = "SWAP";
                break;
            case iSWAP:
                name = "iSWP";
                break;
            case Peres:
                name = "Pr ";
                break;
            case Peresdag:
                name = "Prdg";
                break;
            case Compound:
                name = "Comp";
                break;
            case Measure:
                name = "Meas";
                break;
            case Teleportation:
                name = "Tele";
                break;
            case Reset:
                name = "Rst ";
                break;
            case Snapshot:
                name = "Snap";
                break;
            case ShowProbabilities:
                name = "Show probabilities";
                break;
            case Barrier:
                name = "Barr";
                break;
            case ClassicControlled:
                name = "clc_";
                break;
            default:
                throw QFRException("This constructor shall not be called for gate type (index) " + std::to_string(static_cast<int>(type)));
        }
    }

    std::ostream& Operation::printParameters(std::ostream& os) const {
        if (isClassicControlledOperation()) {
            os << "\tc[" << parameter[0];
            if (parameter[1] != 1) {
                os << " ... " << (parameter[0] + parameter[1] - 1);
            }
            os << "] == " << parameter[2];
            return os;
        }

        bool isZero = true;
        for (const auto& p: parameter) {
            if (p != static_cast<fp>(0)) {
                isZero = false;
                break;
            }
        }
        if (!isZero) {
            os << "\tp: (" << parameter[0] << ") ";
            for (size_t j = 1; j < MAX_PARAMETERS; ++j) {
                isZero = true;
                for (size_t i = j; i < MAX_PARAMETERS; ++i) {
                    if (parameter.at(i) != static_cast<fp>(0)) {
                        isZero = false;
                        break;
                    }
                }
                if (isZero) {
                    break;
                }
                os << "(" << parameter.at(j) << ") ";
            }
        }

        return os;
    }

    std::ostream& Operation::print(std::ostream& os) const {
        const auto precBefore = std::cout.precision(20);

        os << std::setw(4) << name << "\t";

        auto controlIt = controls.begin();
        auto targetIt  = targets.begin();
        for (std::size_t i = 0; i < nqubits; ++i) {
            if (targetIt != targets.end() && *targetIt == i) {
                if (type == ClassicControlled) {
                    os << "\033[1m\033[35m" << name[2] << name[3];
                } else {
                    os << "\033[1m\033[36m" << name[0] << name[1];
                }
                os << "\t\033[0m";
                ++targetIt;
            } else if (controlIt != controls.end() && controlIt->qubit == i) {
                if (controlIt->type == Control::Type::Pos) {
                    os << "\033[32m";
                } else {
                    os << "\033[31m";
                }
                os << "c\t"
                   << "\033[0m";
                ++controlIt;
            } else {
                os << "|\t";
            }
        }

        printParameters(os);

        std::cout.precision(precBefore);

        return os;
    }

    std::ostream& Operation::print(std::ostream& os, const Permutation& permutation) const {
        const auto precBefore = std::cout.precision(20);

        os << std::setw(4) << name << "\t";
        const auto& actualControls = getControls();
        const auto& actualTargets  = getTargets();
        auto        controlIt      = actualControls.cbegin();
        auto        targetIt       = actualTargets.cbegin();
        for (const auto& [physical, logical]: permutation) {
            if (targetIt != actualTargets.cend() && *targetIt == physical) {
                if (type == ClassicControlled) {
                    os << "\033[1m\033[35m" << name[2] << name[3];
                } else {
                    os << "\033[1m\033[36m" << name[0] << name[1];
                }
                os << "\t\033[0m";
                ++targetIt;
            } else if (controlIt != actualControls.cend() && controlIt->qubit == physical) {
                if (controlIt->type == Control::Type::Pos) {
                    os << "\033[32m";
                } else {
                    os << "\033[31m";
                }
                os << "c\t"
                   << "\033[0m";
                ++controlIt;
            } else {
                os << "|\t";
            }
        }

        printParameters(os);

        std::cout.precision(precBefore);

        return os;
    }

    bool Operation::equals(const Operation& op, const Permutation& perm1, const Permutation& perm2) const {
        // check type
        if (getType() != op.getType()) {
            return false;
        }

        // check number of controls
        const auto nc1 = getNcontrols();
        const auto nc2 = op.getNcontrols();
        if (nc1 != nc2) {
            return false;
        }

        // check parameters
        const auto param1 = getParameter();
        const auto param2 = op.getParameter();
        if (param1 != param2) {
            return false;
        }

        // check controls
        if (nc1 != 0U) {
            Controls controls1{};
            if (perm1.empty()) {
                controls1 = getControls();
            } else {
                for (const auto& control: getControls()) {
                    controls1.emplace(Control{perm1.at(control.qubit), control.type});
                }
            }

            Controls controls2{};
            if (perm2.empty()) {
                controls2 = op.getControls();
            } else {
                for (const auto& control: op.getControls()) {
                    controls2.emplace(Control{perm2.at(control.qubit), control.type});
                }
            }

            if (controls1 != controls2) {
                return false;
            }
        }

        // check targets
        std::set<Qubit> targets1{};
        if (perm1.empty()) {
            targets1 = {getTargets().begin(), getTargets().end()};
        } else {
            for (const auto& target: getTargets()) {
                targets1.emplace(perm1.at(target));
            }
        }

        std::set<Qubit> targets2{};
        if (perm2.empty()) {
            targets2 = {op.getTargets().begin(), op.getTargets().end()};
        } else {
            for (const auto& target: op.getTargets()) {
                targets2.emplace(perm2.at(target));
            }
        }

        return targets1 == targets2;
    }

    void Operation::addDepthContribution(std::vector<std::size_t>& depths) const {
        std::size_t maxDepth = 0;
        for (const auto& target: getTargets()) {
            maxDepth = std::max(maxDepth, depths[target]);
        }
        for (const auto& control: getControls()) {
            maxDepth = std::max(maxDepth, depths[control.qubit]);
        }
        maxDepth += 1;
        for (const auto& target: getTargets()) {
            depths[target] = maxDepth;
        }
        for (const auto& control: getControls()) {
            depths[control.qubit] = maxDepth;
        }
    }

} // namespace qc
