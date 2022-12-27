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
            auto ctrl = dd::Control{static_cast<Qubit>(i), dd::Control::Type::pos};
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
            qcMapped->reset(static_cast<Qubit>(ancStart + 1));

            qcMapped->x(ancStart, dd::Control{static_cast<Qubit>(i), dd::Control::Type::pos});
            qcMapped->x(ancStart, dd::Control{static_cast<Qubit>(i + nQubits), dd::Control::Type::pos});
            qcMapped->x(static_cast<Qubit>(ancStart + 1), dd::Control{static_cast<Qubit>(i + nQubits), dd::Control::Type::pos});
            qcMapped->x(static_cast<Qubit>(ancStart + 1), dd::Control{static_cast<Qubit>(i + 2 * nQubits), dd::Control::Type::pos});

            qcMapped->measure(ancStart, clStart);
            qcMapped->measure(static_cast<Qubit>(ancStart + 1), clStart + 1);

            const auto controlRegister = std::make_pair(static_cast<Qubit>(clStart), QubitCount(2));
            classicalControl(controlRegister, 1, qc::X, static_cast<Qubit>(i));
            classicalControl(controlRegister, 2, qc::X, static_cast<Qubit>(i + 2 * nQubits));
            classicalControl(controlRegister, 3, qc::X, static_cast<Qubit>(i + nQubits));
        }
    }

    void Q3Shor::writeDecoding() {
        if (isDecoded) {
            return;
        }
        const auto nQubits = qcOriginal->getNqubits();
        for (Qubit i = 0; i < nQubits; i++) {
            auto ctrl = dd::Control{static_cast<Qubit>(i), dd::Control::Type::pos};
            qcMapped->x(static_cast<Qubit>(i + nQubits), ctrl);
            qcMapped->x(static_cast<Qubit>(i + 2 * nQubits), ctrl);
            ccx(i, static_cast<Qubit>(i + nQubits), true, static_cast<Qubit>(i + 2 * nQubits), true);
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
                    if (gate.getNcontrols()) {
                        auto& ctrls = gate.getControls();
                        qcMapped->emplace_back<qc::StandardOperation>(qcMapped->getNqubits(), ctrls, i, gate.getType());
                        dd::Controls ctrls2;
                        dd::Controls ctrls3;
                        for (const auto& ct: ctrls) {
                            ctrls2.insert(dd::Control{static_cast<Qubit>(ct.qubit + nQubits), ct.type});
                            ctrls3.insert(dd::Control{static_cast<Qubit>(ct.qubit + 2 * nQubits), ct.type});
                        }
                        qcMapped->emplace_back<qc::StandardOperation>(qcMapped->getNqubits(), ctrls2, i + nQubits, gate.getType());
                        qcMapped->emplace_back<qc::StandardOperation>(qcMapped->getNqubits(), ctrls3, i + 2 * nQubits, gate.getType());
                    } else {
                        if (gate.getType() == qc::H) {
                            qcMapped->x(static_cast<Qubit>(i + 1), dd::Control{static_cast<Qubit>(i)});
                            qcMapped->x(static_cast<Qubit>(i + 2), dd::Control{static_cast<Qubit>(i)});
                            qcMapped->h(i);
                            qcMapped->x(static_cast<Qubit>(i + 1), dd::Control{static_cast<Qubit>(i)});
                            qcMapped->x(static_cast<Qubit>(i + 2), dd::Control{static_cast<Qubit>(i)});
                        } else {
                            qcMapped->emplace_back<qc::StandardOperation>(qcMapped->getNqubits(), i, gate.getType());
                            qcMapped->emplace_back<qc::StandardOperation>(qcMapped->getNqubits(), i + nQubits, gate.getType());
                            qcMapped->emplace_back<qc::StandardOperation>(qcMapped->getNqubits(), i + 2 * nQubits, gate.getType());
                        }
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