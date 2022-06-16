/*
 * This file is part of MQT QFR library which is released under the MIT license.
 * See file README.md or go to https://www.cda.cit.tum.de/research/quantum/ for more information.
 */

#include "operations/Operation.hpp"

namespace qc {
    void Operation::setName() {
        switch (type) {
            case I:
                strcpy(name, "I   ");
                break;
            case H:
                strcpy(name, "H   ");
                break;
            case X:
                strcpy(name, "X   ");
                break;
            case Y:
                strcpy(name, "Y   ");
                break;
            case Z:
                strcpy(name, "Z   ");
                break;
            case S:
                strcpy(name, "S   ");
                break;
            case Sdag:
                strcpy(name, "Sdag");
                break;
            case T:
                strcpy(name, "T   ");
                break;
            case Tdag:
                strcpy(name, "Tdag");
                break;
            case V:
                strcpy(name, "V   ");
                break;
            case Vdag:
                strcpy(name, "Vdag");
                break;
            case U3:
                strcpy(name, "U   ");
                break;
            case U2:
                strcpy(name, "U2  ");
                break;
            case Phase:
                strcpy(name, "P   ");
                break;
            case SX:
                strcpy(name, "SX  ");
                break;
            case SXdag:
                strcpy(name, "SXdg");
                break;
            case RX:
                strcpy(name, "RX  ");
                break;
            case RY:
                strcpy(name, "RY  ");
                break;
            case RZ:
                strcpy(name, "RZ  ");
                break;
            case SWAP:
                strcpy(name, "SWAP");
                break;
            case iSWAP:
                strcpy(name, "iSWP");
                break;
            case Peres:
                strcpy(name, "Pres");
                break;
            case Peresdag:
                strcpy(name, "Prdg");
                break;
            case Compound:
                strcpy(name, "Comp");
                break;
            case Measure:
                strcpy(name, "Meas");
                break;
            case Teleportation:
                strcpy(name, "Tele");
                break;
            case Reset:
                strcpy(name, "Rst ");
                break;
            case Snapshot:
                strcpy(name, "Snap");
                break;
            case ShowProbabilities:
                strcpy(name, "Show probabilities");
                break;
            case Barrier:
                strcpy(name, "Barr");
                break;
            case ClassicControlled:
                strcpy(name, "clc_");
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
        for (size_t i = 0; i < MAX_PARAMETERS; ++i) {
            if (parameter[i] != 0.L)
                isZero = false;
        }
        if (!isZero) {
            os << "\tp: (";
            dd::ComplexValue::printFormatted(os, parameter[0]);
            os << ") ";
            for (size_t j = 1; j < MAX_PARAMETERS; ++j) {
                isZero = true;
                for (size_t i = j; i < MAX_PARAMETERS; ++i) {
                    if (parameter[i] != 0.L)
                        isZero = false;
                }
                if (isZero) break;
                os << "(";
                dd::ComplexValue::printFormatted(os, parameter[j]);
                os << ") ";
            }
        }

        return os;
    }

    std::ostream& Operation::print(std::ostream& os) const {
        const auto prec_before = std::cout.precision(20);

        os << std::setw(4) << name << "\t";

        auto controlIt = controls.begin();
        auto targetIt  = targets.begin();
        for (dd::QubitCount i = 0; i < nqubits; ++i) {
            if (targetIt != targets.end() && *targetIt == static_cast<dd::Qubit>(i)) {
                if (type == ClassicControlled) {
                    os << "\033[1m\033[35m" << name[2] << name[3];
                } else {
                    os << "\033[1m\033[36m" << name[0] << name[1];
                }
                os << "\t\033[0m";
                ++targetIt;
            } else if (controlIt != controls.end() && controlIt->qubit == static_cast<dd::Qubit>(i)) {
                if (controlIt->type == dd::Control::Type::pos) {
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

        std::cout.precision(prec_before);

        return os;
    }

    std::ostream& Operation::print(std::ostream& os, const Permutation& permutation) const {
        const auto prec_before = std::cout.precision(20);

        os << std::setw(4) << name << "\t";
        const auto& actualControls = getControls();
        const auto& actualTargets  = getTargets();
        auto        controlIt      = actualControls.cbegin();
        auto        targetIt       = actualTargets.cbegin();
        for (const auto& [physical, logical]: permutation) {
            //            std::cout << static_cast<std::size_t>(physical) << " ";
            //            if (targetIt != targets.cend())
            //                std::cout << static_cast<std::size_t>(*targetIt) << " ";
            //            else
            //                std::cout << "x ";
            //
            //            if (controlIt != controls.cend())
            //                std::cout << static_cast<std::size_t>(controlIt->qubit) << " ";
            //            else
            //                std::cout << "x ";
            //            std::cout << std::endl;

            if (targetIt != actualTargets.cend() && *targetIt == physical) {
                if (type == ClassicControlled) {
                    os << "\033[1m\033[35m" << name[2] << name[3];
                } else {
                    os << "\033[1m\033[36m" << name[0] << name[1];
                }
                os << "\t\033[0m";
                ++targetIt;
            } else if (controlIt != actualControls.cend() && controlIt->qubit == physical) {
                if (controlIt->type == dd::Control::Type::pos) {
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

        std::cout.precision(prec_before);

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
        for (std::size_t p = 0U; p < qc::MAX_PARAMETERS; ++p) {
            // it might make sense to use fuzzy comparison here
            if (param1[p] != param2[p]) { return false; }
        }

        // check controls
        if (nc1 != 0U) {
            dd::Controls controls1{};
            if (perm1.empty()) {
                controls1 = getControls();
            } else {
                for (const auto& control: getControls()) {
                    controls1.emplace(dd::Control{perm1.at(control.qubit), control.type});
                }
            }

            dd::Controls controls2{};
            if (perm2.empty()) {
                controls2 = op.getControls();
            } else {
                for (const auto& control: op.getControls()) {
                    controls2.emplace(dd::Control{perm2.at(control.qubit), control.type});
                }
            }

            if (controls1 != controls2) { return false; }
        }

        // check targets
        std::set<dd::Qubit> targets1{};
        if (perm1.empty()) {
            targets1 = {getTargets().begin(), getTargets().end()};
        } else {
            for (const auto& target: getTargets()) {
                targets1.emplace(perm1.at(target));
            }
        }

        std::set<dd::Qubit> targets2{};
        if (perm2.empty()) {
            targets2 = {op.getTargets().begin(), op.getTargets().end()};
        } else {
            for (const auto& target: op.getTargets()) {
                targets2.emplace(perm2.at(target));
            }
        }
        if (targets1 != targets2) { return false; }

        // operations are identical
        return true;
    }

} // namespace qc
