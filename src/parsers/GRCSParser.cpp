/*
 * This file is part of MQT QFR library which is released under the MIT license.
 * See file README.md or go to https://www.cda.cit.tum.de/research/quantum/ for more information.
 */

#include "QuantumComputation.hpp"

void qc::QuantumComputation::importGRCS(std::istream& is) {
    std::size_t nq{};
    is >> nq;
    addQubitRegister(nq);
    addClassicalRegister(nq);

    std::string line;
    std::string identifier;
    Qubit       control = 0;
    Qubit       target  = 0;
    std::size_t cycle   = 0;
    while (std::getline(is, line)) {
        if (line.empty()) {
            continue;
        }
        std::stringstream ss(line);
        ss >> cycle;
        ss >> identifier;
        if (identifier == "cz") {
            ss >> control;
            ss >> target;
            z(target, qc::Control{control});
        } else if (identifier == "is") {
            ss >> control;
            ss >> target;
            iswap(control, target);
        } else {
            ss >> target;
            if (identifier == "h") {
                h(target);
            } else if (identifier == "t") {
                t(target);
            } else if (identifier == "x_1_2") {
                rx(target, qc::PI_2);
            } else if (identifier == "y_1_2") {
                ry(target, qc::PI_2);
            } else {
                throw QFRException("[grcs parser] unknown gate '" + identifier + "'");
            }
        }
    }
}
