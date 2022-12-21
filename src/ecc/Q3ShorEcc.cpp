/*
 * This file is part of MQT QFR library which is released under the MIT license.
 * See file README.md or go to https://www.cda.cit.tum.de/research/quantum/ for more information.
 */

#include "ecc/Q3ShorEcc.hpp"

void Q3ShorEcc::initMappedCircuit() {
    //method is overridden because we need 2 kinds of classical measurement output registers
    qcOriginal->stripIdleQubits(true, false);
    qcMapped->addQubitRegister(getNOutputQubits(qcOriginal->getNqubits()));
    auto cRegs = qcOriginal->getCregs();
    for (auto const& [regName, regBits]: cRegs) {
        qcMapped->addClassicalRegister(regBits.second, regName.c_str());
    }
    qcMapped->addClassicalRegister(2, "qecc");
}

void Q3ShorEcc::writeEncoding() {
    if (!isDecoded || !gatesWritten) {
        gatesWritten = true;
        return;
    }
    isDecoded          = false;
    const auto nQubits = qcOriginal->getNqubits();

    for (std::size_t i = 0; i < nQubits; i++) {
        auto ctrl = dd::Control{dd::Qubit(i), dd::Control::Type::pos};
        qcMapped->x(dd::Qubit(i + nQubits), ctrl);
        qcMapped->x(dd::Qubit(i + 2 * nQubits), ctrl);
    }
}

void Q3ShorEcc::measureAndCorrect() {
    if (isDecoded || !gatesWritten) {
        return;
    }
    const auto nQubits  = qcOriginal->getNqubits();
    const auto ancStart = static_cast<dd::Qubit>(nQubits * ecc.nRedundantQubits); //measure start (index of first ancilla qubit)
    const auto clStart  = static_cast<dd::Qubit>(qcOriginal->getNcbits());
    for (std::size_t i = 0; i < nQubits; i++) {
        qcMapped->reset(ancStart);
        qcMapped->reset(static_cast<dd::Qubit>(ancStart + 1));

        qcMapped->x(ancStart, dd::Control{dd::Qubit(i), dd::Control::Type::pos});
        qcMapped->x(ancStart, dd::Control{dd::Qubit(i + nQubits), dd::Control::Type::pos});
        qcMapped->x(dd::Qubit(ancStart + 1), dd::Control{dd::Qubit(i + nQubits), dd::Control::Type::pos});
        qcMapped->x(dd::Qubit(ancStart + 1), dd::Control{dd::Qubit(i + 2 * nQubits), dd::Control::Type::pos});

        qcMapped->measure(ancStart, clStart);
        qcMapped->measure(static_cast<dd::Qubit>(ancStart + 1), clStart + 1);

        std::unique_ptr<qc::Operation> op1   = std::make_unique<qc::StandardOperation>(qcMapped->getNqubits(),
                                                                                     dd::Qubit(i), qc::X);
        const auto                     pair1 = std::make_pair(dd::Qubit(clStart), dd::QubitCount(2));
        qcMapped->emplace_back<qc::ClassicControlledOperation>(op1, pair1, 1);

        std::unique_ptr<qc::Operation> op2   = std::make_unique<qc::StandardOperation>(qcMapped->getNqubits(),
                                                                                     dd::Qubit(i + 2 * nQubits), qc::X);
        const auto                     pair2 = std::make_pair(dd::Qubit(clStart), dd::QubitCount(2));
        qcMapped->emplace_back<qc::ClassicControlledOperation>(op2, pair2, 2);

        std::unique_ptr<qc::Operation> op3   = std::make_unique<qc::StandardOperation>(qcMapped->getNqubits(),
                                                                                     dd::Qubit(i + nQubits), qc::X);
        const auto                     pair3 = std::make_pair(dd::Qubit(clStart), dd::QubitCount(2));
        qcMapped->emplace_back<qc::ClassicControlledOperation>(op3, pair3, 3);
    }
}

void Q3ShorEcc::writeDecoding() {
    if (isDecoded) {
        return;
    }
    const auto nQubits = qcOriginal->getNqubits();
    for (dd::Qubit i = 0; i < nQubits; i++) {
        auto ctrl = dd::Control{dd::Qubit(i), dd::Control::Type::pos};
        qcMapped->x(dd::Qubit(i + nQubits), ctrl);
        qcMapped->x(dd::Qubit(i + 2 * nQubits), ctrl);
        writeToffoli(i, i + nQubits, true, i + 2 * nQubits, true);
    }
    isDecoded = true;
}

void Q3ShorEcc::mapGate(const qc::Operation& gate) {
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
                        ctrls2.insert(dd::Control{dd::Qubit(ct.qubit + nQubits), ct.type});
                        ctrls3.insert(dd::Control{dd::Qubit(ct.qubit + 2 * nQubits), ct.type});
                    }
                    qcMapped->emplace_back<qc::StandardOperation>(qcMapped->getNqubits(), ctrls2, i + nQubits, gate.getType());
                    qcMapped->emplace_back<qc::StandardOperation>(qcMapped->getNqubits(), ctrls3, i + 2 * nQubits, gate.getType());
                } else {
                    if (gate.getType() == qc::H) {
                        qcMapped->x(dd::Qubit(i + 1), dd::Control{dd::Qubit(i)});
                        qcMapped->x(dd::Qubit(i + 2), dd::Control{dd::Qubit(i)});
                        qcMapped->h(i);
                        qcMapped->x(dd::Qubit(i + 1), dd::Control{dd::Qubit(i)});
                        qcMapped->x(dd::Qubit(i + 2), dd::Control{dd::Qubit(i)});
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
            if (auto measureGate = dynamic_cast<const qc::NonUnitaryOperation*>(&gate)) {
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
