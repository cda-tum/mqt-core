/*
 * This file is part of JKQ QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#include <stdlib.h>

#ifndef QFR_EccStatistics_HPP
#define QFR_EccStatistics_HPP

class EccStatistics {
public:
    unsigned long nInputQubits;
    unsigned long nInputGates;
    unsigned long nOutputQubits;
    unsigned long nOutputClassicalBits;
    unsigned long nOutputGates;

    std::string outputName;

    bool success;

    double getGateOverhead() {
        return (double)nOutputGates/(double)nInputGates;
    }
};

#endif // QFR_EccStatistics_HPP
