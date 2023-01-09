/*
 * This file is part of MQT QFR library which is released under the MIT license.
 * See file README.md or go to https://www.cda.cit.tum.de/research/quantum/ for more information.
 */

#include "ecc/Ecc.hpp"

namespace ecc {
    void Ecc::initMappedCircuit() {
        qcOriginal->stripIdleQubits(true, false);
        qcMapped->addQubitRegister(getNOutputQubits(qcOriginal->getNqubits()));
        auto cRegs = qcOriginal->getCregs();
        for (auto const& [regName, regBits]: cRegs) {
            qcMapped->addClassicalRegister(regBits.second, regName);
        }
        for (auto const& [regBits, regName]: ecc.classicalRegisters) {
            qcMapped->addClassicalRegister(regBits, regName);
        }
    }

    std::shared_ptr<qc::QuantumComputation> Ecc::apply() {
        initMappedCircuit();

        writeEncoding();
        isDecoded = false;

        std::size_t nInputGates = 0U;
        for (const auto& gate: *qcOriginal) {
            nInputGates++;
            mapGate(*gate);
            if (measureFrequency > 0 && nInputGates % measureFrequency == 0) {
                measureAndCorrect();
            }
        }

        //mapGate(...) can change 'isDecoded', therefore check it again
        if (!isDecoded) {
            measureAndCorrect();
            writeDecoding();
            isDecoded = true;
        }

        return qcMapped;
    }
} // namespace ecc
