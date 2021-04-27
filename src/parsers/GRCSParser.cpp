/*
 * This file is part of JKQ QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#include "QuantumComputation.hpp"

void qc::QuantumComputation::importGRCS(std::istream& is) {
    std::size_t nq;
    is >> nq;
    addQubitRegister(nq);
    addClassicalRegister(nq);

    std::string line;
    std::string identifier;
    std::size_t control = 0;
    std::size_t target  = 0;
    std::size_t cycle   = 0;
    while (std::getline(is, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        ss >> cycle;
        ss >> identifier;
        if (identifier == "cz") {
            ss >> control;
            ss >> target;
            emplace_back<StandardOperation>(nqubits, dd::Control{static_cast<dd::Qubit>(control)}, target, Z);
        } else if (identifier == "is") {
            ss >> control;
            ss >> target;
            emplace_back<StandardOperation>(nqubits, dd::Controls{}, control, target, iSWAP);
        } else {
            ss >> target;
            if (identifier == "h")
                emplace_back<StandardOperation>(nqubits, target, H);
            else if (identifier == "t")
                emplace_back<StandardOperation>(nqubits, target, T);
            else if (identifier == "x_1_2")
                emplace_back<StandardOperation>(nqubits, target, RX, dd::PI_2);
            else if (identifier == "y_1_2")
                emplace_back<StandardOperation>(nqubits, target, RY, dd::PI_2);
            else {
                throw QFRException("[grcs parser] unknown gate '" + identifier + "'");
            }
        }
    }
}
