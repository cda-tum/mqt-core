/*
 * This file is part of JKQ QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
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

        auto controlIt = controls.cbegin();
        auto targetIt  = targets.cbegin();
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

            if (targetIt != targets.cend() && *targetIt == physical) {
                if (type == ClassicControlled) {
                    os << "\033[1m\033[35m" << name[2] << name[3];
                } else {
                    os << "\033[1m\033[36m" << name[0] << name[1];
                }
                os << "\t\033[0m";
                ++targetIt;
            } else if (controlIt != controls.cend() && controlIt->qubit == physical) {
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

} // namespace qc
