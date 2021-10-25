/*
* This file is part of JKQ QFR library which is released under the MIT license.
* See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
*/

#ifndef QFR_OPTYPE_HPP
#define QFR_OPTYPE_HPP

#include "dd/Definitions.hpp"

#include <cstdint>
#include <functional>
#include <iostream>
#include <string>

namespace qc {
    // Natively supported operations of the QFR library
    enum OpType : std::uint8_t {
        None,
        // Standard Operations
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
        iSWAP,
        Peres,
        Peresdag,
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
        ClassicControlled
    };

    inline std::string toString(const OpType& opType) {
        switch (opType) {
            case None: return "none";
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
            case Compound: return "compound";
            case Measure: return "measure";
            case Reset: return "reset";
            case Snapshot: return "snapshot";
            case ShowProbabilities: return "show probabilities";
            case Barrier: return "barrier";
            case Teleportation: return "teleportation";
            case ClassicControlled: return "classic controlled";
            default:
                return "";
        }
    }

    inline OpType opTypeFromString(const std::string& opType) {
        if (opType == "none" || opType == "0")
            return OpType::None;
        else if (opType == "i" || opType == "1")
            return OpType::I;
        else if (opType == "h" || opType == "2")
            return OpType::H;
        else if (opType == "x" || opType == "3")
            return OpType::X;
        else if (opType == "y" || opType == "4")
            return OpType::Y;
        else if (opType == "z" || opType == "5")
            return OpType::Z;
        else if (opType == "s" || opType == "6")
            return OpType::S;
        else if (opType == "sdg" || opType == "7")
            return OpType::Sdag;
        else if (opType == "t" || opType == "8")
            return OpType::T;
        else if (opType == "tdg" || opType == "9")
            return OpType::Tdag;
        else if (opType == "v" || opType == "10")
            return OpType::V;
        else if (opType == "vdg" || opType == "11")
            return OpType::Vdag;
        else if (opType == "u3" || opType == "12")
            return OpType::U3;
        else if (opType == "u2" || opType == "13")
            return OpType::U2;
        else if (opType == "p" || opType == "14")
            return OpType::Phase;
        else if (opType == "sx" || opType == "15")
            return OpType::SX;
        else if (opType == "sxdg" || opType == "16")
            return OpType::SXdag;
        else if (opType == "rx" || opType == "17")
            return OpType::RX;
        else if (opType == "ry" || opType == "18")
            return OpType::RY;
        else if (opType == "rz" || opType == "19")
            return OpType::RZ;
        else if (opType == "swap" || opType == "20")
            return OpType::SWAP;
        else if (opType == "iswap" || opType == "21")
            return OpType::iSWAP;
        else if (opType == "peres" || opType == "22")
            return OpType::Peres;
        else if (opType == "peresdg" || opType == "23")
            return OpType::Peresdag;
        else if (opType == "compound" || opType == "24")
            return OpType::Compound;
        else if (opType == "measure" || opType == "25")
            return OpType::Measure;
        else if (opType == "reset" || opType == "26")
            return OpType::Reset;
        else if (opType == "snapshot" || opType == "27")
            return OpType::Snapshot;
        else if (opType == "show probabilities" || opType == "28")
            return OpType::ShowProbabilities;
        else if (opType == "barrier" || opType == "29")
            return OpType::Barrier;
        else if (opType == "teleportation" || opType == "30")
            return OpType::Teleportation;
        else if (opType == "classic controlled" || opType == "31")
            return OpType::ClassicControlled;
        else {
            throw std::runtime_error("Unknown operation type: " + opType);
        }
    }

    std::istream& operator>>(std::istream& in, OpType& opType) {
        std::string token;
        in >> token;
        opType = opTypeFromString(token);
        return in;
    }

    std::ostream& operator<<(std::ostream& out, OpType& opType) {
        out << toString(opType);
        return out;
    }
} // namespace qc

#endif //QFR_OPTYPE_HPP
