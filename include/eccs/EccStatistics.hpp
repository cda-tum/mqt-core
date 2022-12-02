/*
* This file is part of MQT QFR library which is released under the MIT license.
* See file README.md or go to https://www.cda.cit.tum.de/research/quantum/ for more information.
*/

#pragma once

#include <cstdlib>
#include <string>

class EccStatistics {
public:
    unsigned long nInputQubits;
    unsigned long nInputClassicalBits;
    unsigned long nInputGates;
    unsigned long nOutputQubits;
    unsigned long nOutputClassicalBits;
    unsigned long nOutputGates;

    std::string outputName;

    [[nodiscard]] double getGateOverhead() const {
        return (double)nOutputGates / (double)nInputGates;
    }
};
