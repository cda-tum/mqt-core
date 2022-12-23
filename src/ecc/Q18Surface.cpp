/*
* This file is part of MQT QFR library which is released under the MIT license.
* See file README.md or go to https://www.cda.cit.tum.de/research/quantum/ for more information.
*/

#include "ecc/Q18Surface.hpp"

void Q18Surface::initMappedCircuit() {
    //method is overridden because we need 2 kinds of classical measurement output registers
    qcOriginal->stripIdleQubits(true, false);
    qcMapped->addQubitRegister(getNOutputQubits(qcOriginal->getNqubits()));
    auto cRegs = qcOriginal->getCregs();
    for (auto const& [regName, regBits]: cRegs) {
        qcMapped->addClassicalRegister(regBits.second, regName.c_str());
    }
    qcMapped->addClassicalRegister(ancillaWidth, "qeccX");
    qcMapped->addClassicalRegister(ancillaWidth, "qeccZ");
}

void Q18Surface::measureAndCorrect() {
    if (isDecoded) {
        return;
    }
    const auto nQubits    = qcOriginal->getNqubits();
    const auto clAncStart = qcOriginal->getNcbits();

    std::map<std::size_t, std::size_t> xCheckMasks;
    std::map<std::size_t, std::size_t> zCheckMasks;
    for (std::size_t j = 0; j < ancillaWidth; j++) {
        xCheckMasks[xChecks.at(j)] = 1 << j;
        zCheckMasks[zChecks.at(j)] = 1 << j;
    }

    for (dd::Qubit i = 0; i < nQubits; i++) {
        std::array<dd::Qubit, 36>   qubits        = {};
        std::array<dd::Control, 36> controlQubits = {};
        for (std::size_t j = 0; j < qubits.size(); j++) {
            qubits.at(j) = static_cast<dd::Qubit>(i + j * nQubits);
        }
        for (std::size_t j = 0; j < controlQubits.size(); j++) {
            controlQubits.at(j) = dd::Control{static_cast<dd::Qubit>(qubits.at(j)), dd::Control::Type::pos};
        }

        if (gatesWritten) {
            for (dd::Qubit const ai: ancillaIndices) {
                qcMapped->reset(qubits.at(ai));
            }
        }

        //initialize ancillas: Z-check
        for (const auto& pair: qubitCorrectionX) {
            for (auto ancilla: pair.second) {
                qcMapped->x(qubits[ancilla], controlQubits[pair.first]);
            }
        }

        //initialize ancillas: X-check

        for (std::size_t const xc: zChecks) {
            qcMapped->h(qubits.at(xc));
        }
        for (const auto& pair: qubitCorrectionZ) {
            for (auto ancilla: pair.second) {
                qcMapped->x(qubits[pair.first], controlQubits[ancilla]);
            }
        }
        for (std::size_t const xc: zChecks) {
            qcMapped->h(qubits.at(xc));
        }

        //map ancillas to classical bit result
        for (std::size_t j = 0; j < xChecks.size(); j++) {
            qcMapped->measure(qubits[xChecks.at(j)], clAncStart + j);
        }
        for (std::size_t j = 0; j < zChecks.size(); j++) {
            qcMapped->measure(qubits[zChecks.at(j)], clAncStart + ancillaWidth + j);
        }

        std::set<std::size_t> unmeasuredQubits = {3, 6, 17, 18, 32, 34};

        //logic: classical control

        for (const auto& pair: qubitCorrectionX) {
            if (unmeasuredQubits.count(pair.first) > 0) {
                std::size_t mask = 0;
                for (std::size_t value: pair.second) {
                    mask |= value;
                }
                writeClassicalControl(static_cast<dd::Qubit>(clAncStart), ancillaWidth, mask, qc::X, qubits[pair.first]);
            }
        }

        for (const auto& pair: qubitCorrectionZ) {
            if (unmeasuredQubits.count(pair.first) > 0) {
                std::size_t mask = 0;
                for (std::size_t value: pair.second) {
                    mask |= value;
                }
                writeClassicalControl(static_cast<dd::Qubit>(clAncStart + ancillaWidth), ancillaWidth, mask, qc::Z, qubits[pair.first]);
            }
        }

        gatesWritten = true;
    }
}

void Q18Surface::writeDecoding() {
    if (isDecoded) {
        return;
    }
    const auto nQubits = qcOriginal->getNqubits();
    for (dd::Qubit i = 0; i < nQubits; i++) {
        qcMapped->x(static_cast<dd::Qubit>(i + 14 * nQubits), dd::Control{static_cast<dd::Qubit>(i + 8 * nQubits), dd::Control::Type::pos});
        qcMapped->x(static_cast<dd::Qubit>(i + 14 * nQubits), dd::Control{static_cast<dd::Qubit>(i + 13 * nQubits), dd::Control::Type::pos});
        qcMapped->x(static_cast<dd::Qubit>(i + 14 * nQubits), dd::Control{static_cast<dd::Qubit>(i + 15 * nQubits), dd::Control::Type::pos});
        qcMapped->x(static_cast<dd::Qubit>(i + 14 * nQubits), dd::Control{static_cast<dd::Qubit>(i + 20 * nQubits), dd::Control::Type::pos});
        qcMapped->measure(static_cast<dd::Qubit>(i + 14 * nQubits), i);
        qcMapped->reset(static_cast<dd::Qubit>(i));
        qcMapped->x(static_cast<dd::Qubit>(i), dd::Control{static_cast<dd::Qubit>(i + 14 * nQubits), dd::Control::Type::pos});
    }
    isDecoded = true;
}

void Q18Surface::mapGate(const qc::Operation& gate) {
    if (isDecoded && gate.getType() != qc::Measure) {
        writeEncoding();
    }
    const auto nQubits = qcOriginal->getNqubits();

    //no control gate decomposition is supported
    if (gate.isControlled() && gate.getType() != qc::Measure) {
        //multi-qubit gates are currently not supported
        gateNotAvailableError(gate);
    } else {
        static constexpr std::array<std::pair<int, int>, 6> swapQubitIndices = {std::make_pair(1, 29), std::make_pair(3, 17), std::make_pair(6, 34), std::make_pair(8, 22), std::make_pair(13, 27), std::make_pair(18, 32)};

        switch (gate.getType()) {
            case qc::I:
                break;
            case qc::X:
                for (auto i: gate.getTargets()) {
                    qcMapped->x(static_cast<dd::Qubit>(i + 15 * nQubits));
                    qcMapped->x(static_cast<dd::Qubit>(i + 17 * nQubits));
                }
                break;
            case qc::H:
                //apply H gate to every data qubit
                //swap circuit along '/' axis
                for (auto i: gate.getTargets()) {
                    for (const auto j: dataQubits) {
                        qcMapped->h(static_cast<dd::Qubit>(i + j * nQubits));
                    }
                    for (auto pair: swapQubitIndices) {
                        qcMapped->swap(static_cast<dd::Qubit>(i + pair.first * nQubits), static_cast<dd::Qubit>(i + pair.second * nQubits));
                    }
                    //qubits 5, 10, 15, 20, 25, 30 are along axis
                }
                break;
            case qc::Y:
                //Y = Z X
                for (auto i: gate.getTargets()) {
                    qcMapped->z(static_cast<dd::Qubit>(i + 18 * nQubits));
                    qcMapped->z(static_cast<dd::Qubit>(i + 20 * nQubits));
                    qcMapped->x(static_cast<dd::Qubit>(i + 15 * nQubits));
                    qcMapped->x(static_cast<dd::Qubit>(i + 17 * nQubits));
                }
                break;
            case qc::Z:
                for (auto i: gate.getTargets()) {
                    qcMapped->z(static_cast<dd::Qubit>(i + 18 * nQubits));
                    qcMapped->z(static_cast<dd::Qubit>(i + 20 * nQubits));
                }
                break;
            case qc::Measure:
                if (!isDecoded) {
                    measureAndCorrect();
                    writeDecoding();
                }
                if (const auto* measureGate = dynamic_cast<const qc::NonUnitaryOperation*>(&gate)) {
                    const auto& classics = measureGate->getClassics();
                    const auto& targets  = measureGate->getTargets();
                    for (std::size_t j = 0; j < classics.size(); j++) {
                        qcMapped->measure(targets.at(j), classics.at(j));
                    }
                } else {
                    throw std::runtime_error("Dynamic cast to NonUnitaryOperation failed.");
                }
                break;
            default:
                gateNotAvailableError(gate);
        }
    }
}
