/*
 * This file is part of MQT QFR library which is released under the MIT license.
 * See file README.md or go to https://www.cda.cit.tum.de/research/quantum/ for more information.
 */

#include "ecc/Q3Shor.hpp"
namespace ecc {
    void Q3Shor::writeEncoding() {
        if (!isDecoded || !gatesWritten) {
            gatesWritten = true;
            return;
        }
        isDecoded          = false;
        const auto nQubits = qcOriginal->getNqubits();

        for (std::size_t i = 0; i < nQubits; i++) {
            auto ctrl = qc::Control{static_cast<Qubit>(i)};
            qcMapped->x(static_cast<Qubit>(i + nQubits), ctrl);
            qcMapped->x(static_cast<Qubit>(i + 2 * nQubits), ctrl);
        }
    }

    void Q3Shor::measureAndCorrect() {
        if (isDecoded || !gatesWritten) {
            return;
        }
        const auto nQubits  = qcOriginal->getNqubits();
        const auto ancStart = static_cast<Qubit>(nQubits * ecc.nRedundantQubits); //measure start (index of first ancilla qubit)
        const auto clStart  = static_cast<Qubit>(qcOriginal->getNcbits());
        for (std::size_t i = 0; i < nQubits; i++) {
            qcMapped->reset(ancStart);
            qcMapped->reset(ancStart + 1);

            qcMapped->x(ancStart, qc::Control{static_cast<Qubit>(i)});
            qcMapped->x(ancStart, qc::Control{static_cast<Qubit>(i + nQubits)});
            qcMapped->x(ancStart + 1, qc::Control{static_cast<Qubit>(i + nQubits)});
            qcMapped->x(ancStart + 1, qc::Control{static_cast<Qubit>(i + 2 * nQubits)});

            qcMapped->measure(ancStart, clStart);
            qcMapped->measure(ancStart + 1, clStart + 1);

            const auto controlRegister = std::make_pair(clStart, static_cast<QubitCount>(2));
            qcMapped->classicControlled(qc::X, static_cast<Qubit>(i), controlRegister, 1U);
            qcMapped->classicControlled(qc::X, static_cast<Qubit>(i + 2 * nQubits), controlRegister, 2U);
            qcMapped->classicControlled(qc::X, static_cast<Qubit>(i + nQubits), controlRegister, 3U);
        }
    }

    void Q3Shor::writeDecoding() {
        if (isDecoded) {
            return;
        }
        const auto nQubits = qcOriginal->getNqubits();
        for (Qubit i = 0; i < nQubits; i++) {
            std::array<Qubit, N_REDUNDANT_QUBITS> qubits = {i, static_cast<Qubit>(i + nQubits), static_cast<Qubit>(i + 2 * nQubits)};
            qcMapped->x(qubits[1], qc::Control{qubits[0]});
            qcMapped->x(qubits[2], qc::Control{qubits[0]});
            qcMapped->x(qubits[0], {qc::Control{qubits[1]}, qc::Control{qubits[2]}});
        }
        isDecoded = true;
    }

    void Q3Shor::mapGate(const qc::Operation& gate) {
        if (isDecoded && gate.getType() != qc::Measure && gate.getType() != qc::H) {
            writeEncoding();
        }
        const auto nQubits = qcOriginal->getNqubits();
        switch (gate.getType()) {
            case qc::I:
            case qc::Barrier:
                break;
            case qc::X:
            case qc::Y:
            case qc::Z:
            case qc::S:
            case qc::Sdag:
            case qc::T:
            case qc::Tdag:
                for (std::size_t j = 0; j < gate.getNtargets(); j++) {
                    auto i = gate.getTargets()[j];
                    if (gate.getNcontrols() != 0U) {
                        const auto& controls = gate.getControls();
                        qcMapped->emplace_back<qc::StandardOperation>(qcMapped->getNqubits(), controls, i, gate.getType());
                        qc::Controls controls2;
                        qc::Controls controls3;
                        for (const auto& ct: controls) {
                            controls2.insert(qc::Control{static_cast<Qubit>(ct.qubit + nQubits), ct.type});
                            controls3.insert(qc::Control{static_cast<Qubit>(ct.qubit + 2 * nQubits), ct.type});
                        }
                        qcMapped->emplace_back<qc::StandardOperation>(qcMapped->getNqubits(), controls2, i + nQubits, gate.getType());
                        qcMapped->emplace_back<qc::StandardOperation>(qcMapped->getNqubits(), controls3, i + 2 * nQubits, gate.getType());
                    } else {
                        qcMapped->emplace_back<qc::StandardOperation>(qcMapped->getNqubits(), i, gate.getType());
                        qcMapped->emplace_back<qc::StandardOperation>(qcMapped->getNqubits(), i + nQubits, gate.getType());
                        qcMapped->emplace_back<qc::StandardOperation>(qcMapped->getNqubits(), i + 2 * nQubits, gate.getType());
                    }
                }
                break;
            case qc::Measure:
                if (!isDecoded) {
                    measureAndCorrect();
                    writeDecoding();
                }
                if (const auto* measureGate = dynamic_cast<const qc::NonUnitaryOperation*>(&gate)) {
                    for (std::size_t j = 0; j < measureGate->getNclassics(); j++) {
                        qcMapped->measure(measureGate->getTargets()[j], measureGate->getClassics()[j]);
                    }
                } else {
                    throw std::runtime_error("Dynamic cast to NonUnitaryOperation failed.");
                }
                break;
            default:
                gateNotAvailableError(gate);
                break;
        }
    }
} // namespace ecc
