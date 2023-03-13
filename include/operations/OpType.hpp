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

    inline std::ostream& operator<<(std::ostream& out, OpType& opType) {
        out << toString(opType);
        return out;
    }
} // namespace qc
