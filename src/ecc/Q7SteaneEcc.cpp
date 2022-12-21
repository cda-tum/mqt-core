/*
* This file is part of MQT QFR library which is released under the MIT license.
* See file README.md or go to https://www.cda.cit.tum.de/research/quantum/ for more information.
*/

#include "ecc/Q7SteaneEcc.hpp"

void Q7SteaneEcc::initMappedCircuit() {
    //method is overridden because we need 2 kinds of classical measurement output registers
    qcOriginal->stripIdleQubits(true, false);
    qcMapped->addQubitRegister(getNOutputQubits(qcOriginal->getNqubits()));
    auto cRegs = qcOriginal->getCregs();
    for (auto const& [regName, regBits]: cRegs) {
        qcMapped->addClassicalRegister(regBits.second, regName.c_str());
    }
    qcMapped->addClassicalRegister(3, "qecc");
}

void Q7SteaneEcc::writeEncoding() {
    if (!isDecoded) {
        return;
    }
    isDecoded          = false;
    const auto nQubits = qcOriginal->getNqubits();
    //reset data qubits if necessary
    if (gatesWritten) {
        for (std::size_t i = 0; i < nQubits; i++) {
            for (std::size_t j = 1; j < 7; j++) {
                qcMapped->reset(dd::Qubit(i + j * nQubits));
            }
        }
    }
    measureAndCorrectSingle(true);
}

void Q7SteaneEcc::measureAndCorrect() {
    if (isDecoded) {
        return;
    }
    measureAndCorrectSingle(true);
    measureAndCorrectSingle(false);
}

void Q7SteaneEcc::measureAndCorrectSingle(bool xSyndrome) {
    const auto nQubits    = qcOriginal->getNqubits();
    const auto ancStart   = nQubits * ecc.nRedundantQubits;
    const auto clAncStart = static_cast<int>(qcOriginal->getNcbits());

    for (dd::Qubit i = 0; i < nQubits; i++) {
        if (gatesWritten) {
            for (std::size_t i = 0; i < nQubits; i++) {
                qcMapped->reset(static_cast<dd::Qubit>(ancStart));
                qcMapped->reset(static_cast<dd::Qubit>(ancStart + 1));
                qcMapped->reset(static_cast<dd::Qubit>(ancStart + 2));
            }
        }

        qcMapped->h(static_cast<dd::Qubit>(ancStart));
        qcMapped->h(static_cast<dd::Qubit>(ancStart + 1));
        qcMapped->h(static_cast<dd::Qubit>(ancStart + 2));

        auto c0 = dd::Control{dd::Qubit(ancStart), dd::Control::Type::pos};
        auto c1 = dd::Control{dd::Qubit(ancStart + 1), dd::Control::Type::pos};
        auto c2 = dd::Control{dd::Qubit(ancStart + 2), dd::Control::Type::pos};

        void (*writeXZ)(dd::Qubit, dd::Control, const std::shared_ptr<qc::QuantumComputation>&) = xSyndrome ? writeXstatic : writeZstatic;

        //K1: UIUIUIU
        writeXZ(static_cast<dd::Qubit>(i + nQubits * 0), c0, qcMapped);
        writeXZ(static_cast<dd::Qubit>(i + nQubits * 2), c0, qcMapped);
        writeXZ(static_cast<dd::Qubit>(i + nQubits * 4), c0, qcMapped);
        writeXZ(static_cast<dd::Qubit>(i + nQubits * 6), c0, qcMapped);

        //K2: IUUIIUU
        writeXZ(static_cast<dd::Qubit>(i + nQubits * 1), c1, qcMapped);
        writeXZ(static_cast<dd::Qubit>(i + nQubits * 2), c1, qcMapped);
        writeXZ(static_cast<dd::Qubit>(i + nQubits * 5), c1, qcMapped);
        writeXZ(static_cast<dd::Qubit>(i + nQubits * 6), c1, qcMapped);

        //K3: IIIUUUU
        writeXZ(static_cast<dd::Qubit>(i + nQubits * 3), c2, qcMapped);
        writeXZ(static_cast<dd::Qubit>(i + nQubits * 4), c2, qcMapped);
        writeXZ(static_cast<dd::Qubit>(i + nQubits * 5), c2, qcMapped);
        writeXZ(static_cast<dd::Qubit>(i + nQubits * 6), c2, qcMapped);

        qcMapped->h(static_cast<dd::Qubit>(ancStart));
        qcMapped->h(static_cast<dd::Qubit>(ancStart + 1));
        qcMapped->h(static_cast<dd::Qubit>(ancStart + 2));

        qcMapped->measure(static_cast<dd::Qubit>(ancStart), clAncStart);
        qcMapped->measure(static_cast<dd::Qubit>(ancStart + 1), clAncStart + 1);
        qcMapped->measure(static_cast<dd::Qubit>(ancStart + 2), clAncStart + 2);

        //correct Z_i for i+1 = c0*1+c1*2+c2*4
        //correct X_i for i+1 = c3*1+c4*2+c5*4
        for (std::size_t j = 0; j < 7; j++) {
            writeClassicalControl(dd::Qubit(clAncStart), dd::QubitCount(3), j + 1U, xSyndrome ? qc::Z : qc::X, i + j * nQubits);
        }
        gatesWritten = true;
    }
}

void Q7SteaneEcc::writeDecoding() {
    if (isDecoded) {
        return;
    }
    const auto               nQubits           = qcOriginal->getNqubits();
    const auto               clAncStart        = static_cast<int>(qcOriginal->getNcbits());
    std::array<dd::Qubit, 4> correction_needed = {1, 2, 4, 7}; //values with odd amount of '1' bits
    //use exiting registers qeccX and qeccZ for decoding

    for (dd::Qubit i = 0; i < nQubits; i++) {
        //#|###|###
        //0|111|111
        //odd amount of 1's -> x[0] = 1
        //measure from index 1 (not 0) to 6, =qubit 2 to 7

        qcMapped->measure(static_cast<dd::Qubit>(i + 1 * nQubits), clAncStart);
        qcMapped->measure(static_cast<dd::Qubit>(i + 2 * nQubits), clAncStart + 1);
        qcMapped->measure(static_cast<dd::Qubit>(i + 3 * nQubits), clAncStart + 2);
        for (auto value: correction_needed) {
            writeClassicalControl(dd::Qubit(clAncStart), dd::QubitCount(3), value, qc::X, i);
        }
        qcMapped->measure(static_cast<dd::Qubit>(i + 4 * nQubits), clAncStart);
        qcMapped->measure(static_cast<dd::Qubit>(i + 5 * nQubits), clAncStart + 1);
        qcMapped->measure(static_cast<dd::Qubit>(i + 6 * nQubits), clAncStart + 2);
        for (auto value: correction_needed) {
            writeClassicalControl(dd::Qubit(clAncStart), dd::QubitCount(3), value, qc::X, i);
        }
    }
    isDecoded = true;
}

void Q7SteaneEcc::mapGate(const qc::Operation& gate) {
    if (isDecoded && gate.getType() != qc::Measure) {
        writeEncoding();
    }
    const int nQubits = qcOriginal->getNqubits();
    switch (gate.getType()) {
        case qc::I:
            break;
        case qc::X:
        case qc::H:
        case qc::Y:
        case qc::Z:
            for (auto i: gate.getTargets()) {
                if (gate.getNcontrols()) {
                    auto& ctrls = gate.getControls();
                    for (std::size_t j = 0; j < 7; j++) {
                        dd::Controls ctrls2;
                        for (const auto& ct: ctrls) {
                            ctrls2.insert(dd::Control{dd::Qubit(ct.qubit + j * nQubits), ct.type});
                        }
                        qcMapped->emplace_back<qc::StandardOperation>(qcMapped->getNqubits(), ctrls2, static_cast<dd::Qubit>(i + j * nQubits), gate.getType());
                    }
                } else {
                    for (std::size_t j = 0; j < 7; j++) {
                        qcMapped->emplace_back<qc::StandardOperation>(qcMapped->getNqubits(), static_cast<dd::Qubit>(i + j * nQubits), gate.getType());
                    }
                }
            }
            break;
            //locigal S = 3 physical S's
        case qc::S:
        case qc::Sdag:
            for (auto i: gate.getTargets()) {
                if (gate.getNcontrols()) {
                    auto& ctrls = gate.getControls();
                    for (std::size_t j = 0; j < 7; j++) {
                        dd::Controls ctrls2;
                        for (const auto& ct: ctrls) {
                            ctrls2.insert(dd::Control{dd::Qubit(ct.qubit + j * nQubits), ct.type});
                        }
                        qcMapped->emplace_back<qc::StandardOperation>(qcMapped->getNqubits(), ctrls2, static_cast<dd::Qubit>(i + j * nQubits), gate.getType());
                        qcMapped->emplace_back<qc::StandardOperation>(qcMapped->getNqubits(), ctrls2, static_cast<dd::Qubit>(i + j * nQubits), gate.getType());
                        qcMapped->emplace_back<qc::StandardOperation>(qcMapped->getNqubits(), ctrls2, static_cast<dd::Qubit>(i + j * nQubits), gate.getType());
                    }
                } else {
                    for (std::size_t j = 0; j < 7; j++) {
                        qcMapped->emplace_back<qc::StandardOperation>(qcMapped->getNqubits(), static_cast<dd::Qubit>(i + j * nQubits), gate.getType());
                        qcMapped->emplace_back<qc::StandardOperation>(qcMapped->getNqubits(), static_cast<dd::Qubit>(i + j * nQubits), gate.getType());
                        qcMapped->emplace_back<qc::StandardOperation>(qcMapped->getNqubits(), static_cast<dd::Qubit>(i + j * nQubits), gate.getType());
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
                    auto classicalRegisterName = qcOriginal->getClassicalRegister(measureGate->getTargets()[j]);
                    if (!classicalRegisterName.empty()) {
                        qcMapped->measure(static_cast<dd::Qubit>(measureGate->getClassics()[j]), {classicalRegisterName, measureGate->getTargets()[j]});
                    } else {
                        qcMapped->measure(static_cast<dd::Qubit>(measureGate->getClassics()[j]), measureGate->getTargets()[j]);
                    }
                }
            } else {
                throw std::runtime_error("Dynamic cast to NonUnitaryOperation failed.");
            }

            break;
        case qc::T:
        case qc::Tdag:
            for (auto i: gate.getTargets()) {
                if (gate.getControls().empty()) {
                    qcMapped->x(dd::Qubit(i + 5 * nQubits), dd::Control{dd::Qubit(i + 6 * nQubits), dd::Control::Type::pos});
                    qcMapped->x(dd::Qubit(i + 0 * nQubits), dd::Control{dd::Qubit(i + 5 * nQubits), dd::Control::Type::pos});
                    if (gate.getType() == qc::T) {
                        qcMapped->t(dd::Qubit(i + 0 * nQubits));
                    } else {
                        qcMapped->tdag(dd::Qubit(i + 0 * nQubits));
                    }
                    qcMapped->x(dd::Qubit(i + 0 * nQubits), dd::Control{dd::Qubit(i + 5 * nQubits), dd::Control::Type::pos});
                    qcMapped->x(dd::Qubit(i + 5 * nQubits), dd::Control{dd::Qubit(i + 6 * nQubits), dd::Control::Type::pos});
                } else {
                    gateNotAvailableError(gate);
                }
            }
            break;
        default:
            gateNotAvailableError(gate);
    }
}
