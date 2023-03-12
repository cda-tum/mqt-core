/*
* This file is part of MQT QFR library which is released under the MIT license.
* See file README.md or go to https://www.cda.cit.tum.de/research/quantum/ for more information.
*/

#pragma once

#include <cstdint>
#include <functional>
#include <iostream>
#include <string>

namespace qc {
    // Natively supported operations of the QFR library
    enum OpType : std::uint8_t {
        None,
        // Standard Operations
        GPhase,
        I,
        H,
        X,
        Y,
        Z,
        S,
        Sdag,
        T,
        Tdag,
        V,
        Vdag,
        U3,
        U2,
        Phase,
        SX,
        SXdag,
        RX,
        RY,
        RZ,
        SWAP,
        iSWAP, // NOLINT (readability-identifier-naming)
        Peres,
        Peresdag,
        DCX,
        ECR,
        RXX,
        RYY,
        RZZ,
        RZX,
        XXminusYY,
        XXplusYY,
        // Compound Operation
        Compound,
        // Non Unitary Operations
        Measure,
        Reset,
        Snapshot,
        ShowProbabilities,
        Barrier,
        Teleportation,
        // Classically-controlled Operation
        ClassicControlled,
        // Noise operations
        ATrue,
        AFalse,
        MultiATrue,
        MultiAFalse,
        // Number of OpTypes
        OpCount
    };

    inline std::string toString(const OpType& opType) {
        switch (opType) {
            case None: return "none";
            case GPhase: return "gphase";
            case I: return "i";
            case H: return "h";
            case X: return "x";
            case Y: return "y";
            case Z: return "z";
            case S: return "s";
            case Sdag: return "sdg";
            case T: return "t";
            case Tdag: return "tdg";
            case V: return "v";
            case Vdag: return "vdg";
            case U3: return "u3";
            case U2: return "u2";
            case Phase: return "p";
            case SX: return "sx";
            case SXdag: return "sxdg";
            case RX: return "rx";
            case RY: return "ry";
            case RZ: return "rz";
            case SWAP: return "swap";
            case iSWAP: return "iswap";
            case Peres: return "peres";
            case Peresdag: return "peresdg";
            case DCX: return "dcx";
            case ECR: return "ecr";
            case RXX: return "rxx";
            case RYY: return "ryy";
            case RZZ: return "rzz";
            case RZX: return "rzx";
            case XXminusYY: return "xx_minus_yy";
            case XXplusYY: return "xx_plus_yy";
            case Compound: return "compound";
            case Measure: return "measure";
            case Reset: return "reset";
            case Snapshot: return "snapshot";
            case ShowProbabilities: return "show probabilities";
            case Barrier: return "barrier";
            case Teleportation: return "teleportation";
            case ClassicControlled: return "classic controlled";
            default:
                throw std::invalid_argument("Invalid OpType!");
        }
    }

    inline bool isTwoQubitGate(const OpType& opType) {
        switch (opType) {
            case SWAP:
            case iSWAP:
            case Peres:
            case Peresdag:
            case DCX:
            case ECR:
            case RXX:
            case RYY:
            case RZZ:
            case RZX:
            case XXminusYY:
            case XXplusYY:
                return true;
            default:
                return false;
        }
    }

    inline OpType opTypeFromString(const std::string& opType) {
        if (opType == "none" || opType == "0") {
            return OpType::None;
        }
        if (opType == "gphase" || opType == "1") {
            return OpType::GPhase;
        }
        if (opType == "i" || opType == "id" || opType == "2") {
            return OpType::I;
        }
        if (opType == "h" || opType == "ch" || opType == "3") {
            return OpType::H;
        }
        if (opType == "x" || opType == "cx" || opType == "cnot" || opType == "4") {
            return OpType::X;
        }
        if (opType == "y" || opType == "cy" || opType == "5") {
            return OpType::Y;
        }
        if (opType == "z" || opType == "cz" || opType == "6") {
            return OpType::Z;
        }
        if (opType == "s" || opType == "cs" || opType == "7") {
            return OpType::S;
        }
        if (opType == "sdg" || opType == "csdg" || opType == "8") {
            return OpType::Sdag;
        }
        if (opType == "t" || opType == "ct" || opType == "9") {
            return OpType::T;
        }
        if (opType == "tdg" || opType == "ctdg" || opType == "10") {
            return OpType::Tdag;
        }
        if (opType == "v" || opType == "11") {
            return OpType::V;
        }
        if (opType == "vdg" || opType == "12") {
            return OpType::Vdag;
        }
        if (opType == "u3" || opType == "cu3" || opType == "u" || opType == "cu" || opType == "13") {
            return OpType::U3;
        }
        if (opType == "u2" || opType == "cu2" || opType == "14") {
            return OpType::U2;
        }
        if (opType == "u1" || opType == "cu1" || opType == "p" || opType == "cp" || opType == "15") {
            return OpType::Phase;
        }
        if (opType == "sx" || opType == "csx" || opType == "16") {
            return OpType::SX;
        }
        if (opType == "sxdg" || opType == "csxdg" || opType == "17") {
            return OpType::SXdag;
        }
        if (opType == "rx" || opType == "crx" || opType == "18") {
            return OpType::RX;
        }
        if (opType == "ry" || opType == "cry" || opType == "19") {
            return OpType::RY;
        }
        if (opType == "rz" || opType == "crz" || opType == "20") {
            return OpType::RZ;
        }
        if (opType == "swap" || opType == "cswap" || opType == "21") {
            return OpType::SWAP;
        }
        if (opType == "iswap" || opType == "22") {
            return OpType::iSWAP;
        }
        if (opType == "peres" || opType == "23") {
            return OpType::Peres;
        }
        if (opType == "peresdg" || opType == "24") {
            return OpType::Peresdag;
        }
        if (opType == "dcx" || opType == "25") {
            return OpType::DCX;
        }
        if (opType == "ecr" || opType == "26") {
            return OpType::ECR;
        }
        if (opType == "rxx" || opType == "27") {
            return OpType::RXX;
        }
        if (opType == "ryy" || opType == "28") {
            return OpType::RYY;
        }
        if (opType == "rzz" || opType == "29") {
            return OpType::RZZ;
        }
        if (opType == "rzx" || opType == "30") {
            return OpType::RZX;
        }
        if (opType == "xx_minus_yy" || opType == "31") {
            return OpType::XXminusYY;
        }
        if (opType == "xx_plus_yy" || opType == "32") {
            return OpType::XXplusYY;
        }
        if (opType == "compound" || opType == "33") {
            return OpType::Compound;
        }
        if (opType == "measure" || opType == "34") {
            return OpType::Measure;
        }
        if (opType == "reset" || opType == "35") {
            return OpType::Reset;
        }
        if (opType == "snapshot" || opType == "36") {
            return OpType::Snapshot;
        }
        if (opType == "show probabilities" || opType == "37") {
            return OpType::ShowProbabilities;
        }
        if (opType == "barrier" || opType == "38") {
            return OpType::Barrier;
        }
        if (opType == "classic controlled" || opType == "39") {
            return OpType::ClassicControlled;
        }
        throw std::invalid_argument("Unknown operation type: " + opType);
    }

    inline std::istream& operator>>(std::istream& in, OpType& opType) {
        std::string token;
        in >> token;

        if (token.empty()) {
            in.setstate(std::istream::failbit);
            return in;
        }

        opType = opTypeFromString(token);
        return in;
    }

    inline std::ostream& operator<<(std::ostream& out, OpType& opType) {
        out << toString(opType);
        return out;
    }
} // namespace qc
