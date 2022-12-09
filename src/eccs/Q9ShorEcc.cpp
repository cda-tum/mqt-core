/*
 * This file is part of JKQ QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#include "eccs/Q9ShorEcc.hpp"

void Q9ShorEcc::initMappedCircuit() {
    //method is overridden because we need 2 kinds of classical measurement output registers
    qcOriginal->stripIdleQubits(true, false);
    qcMapped->addQubitRegister(getNOutputQubits(qcOriginal->getNqubits()));
    auto cRegs = qcOriginal->getCregs();
    for (auto const& [regName, regBits]: cRegs) {
        qcMapped->addClassicalRegister(regBits.second, regName.c_str());
    }
    qcMapped->addClassicalRegister(2, "qeccX1");
    qcMapped->addClassicalRegister(2, "qeccX2");
    qcMapped->addClassicalRegister(2, "qeccX3");
    qcMapped->addClassicalRegister(2, "qeccZ");
}

void Q9ShorEcc::writeEncoding() {
    if (!isDecoded) {
        return;
    }
    isDecoded          = false;
    const auto nQubits = qcOriginal->getNqubits();
    for (int i = 0; i < nQubits; i++) {
        dd::Control ci = {dd::Qubit(i), dd::Control::Type::pos};
        qcMapped->x(dd::Qubit(i + 3 * nQubits), ci);
        qcMapped->x(dd::Qubit(i + 6 * nQubits), ci);

        qcMapped->h(dd::Qubit(i));
        qcMapped->h(dd::Qubit(i + 3 * nQubits));
        qcMapped->h(dd::Qubit(i + 6 * nQubits));

        dd::Control ci3 = {dd::Qubit(i + 3 * nQubits), dd::Control::Type::pos};
        dd::Control ci6 = {dd::Qubit(i + 6 * nQubits), dd::Control::Type::pos};
        qcMapped->x(dd::Qubit(i + nQubits), ci);
        qcMapped->x(dd::Qubit(i + 2 * nQubits), ci);
        qcMapped->x(dd::Qubit(i + 4 * nQubits), ci3);
        qcMapped->x(dd::Qubit(i + 5 * nQubits), ci3);
        qcMapped->x(dd::Qubit(i + 7 * nQubits), ci6);
        qcMapped->x(dd::Qubit(i + 8 * nQubits), ci6);
    }
    gatesWritten = true;
}

void Q9ShorEcc::measureAndCorrect() {
    if (isDecoded) {
        return;
    }
    const auto nQubits = qcOriginal->getNqubits();
    const auto clStart = qcOriginal->getNcbits();
    for (dd::Qubit i = 0; i < nQubits; i++) {
        //syntactic sugar for qubit indices
        std::array<dd::Qubit, 9>   qubits                  = {};
        std::array<dd::Qubit, 8>   ancillaQubits           = {};
        std::array<dd::Control, 8> ancillaControls         = {};
        std::array<dd::Control, 8> negativeAncillaControls = {};
        for (std::size_t j = 0; j < 9; j++) {
            qubits[j] = dd::Qubit(i + j * nQubits);
        }
        for (std::size_t j = 0; j < 8; j++) {
            ancillaQubits[j] = dd::Qubit(ecc.nRedundantQubits * nQubits + j);
            qcMapped->reset(ancillaQubits[j]);
        }
        for (std::size_t j = 0; j < 8; j++) {
            ancillaControls[j] = dd::Control{dd::Qubit(ancillaQubits[j]), dd::Control::Type::pos};
        }
        for (std::size_t j = 0; j < 8; j++) {
            negativeAncillaControls[j] = dd::Control{dd::Qubit(ancillaQubits[j]), dd::Control::Type::neg};
        }

        // PREPARE measurements --------------------------------------------------------
        for (dd::Qubit j: ancillaQubits) {
            qcMapped->h(j);
        }
        //x errors = indirectly via controlled z
        qcMapped->z(qubits[0], ancillaControls[0]);
        qcMapped->z(qubits[1], ancillaControls[0]);
        qcMapped->z(qubits[1], ancillaControls[1]);
        qcMapped->z(qubits[2], ancillaControls[1]);

        qcMapped->z(qubits[3], ancillaControls[2]);
        qcMapped->z(qubits[4], ancillaControls[2]);
        qcMapped->z(qubits[4], ancillaControls[3]);
        qcMapped->z(qubits[5], ancillaControls[3]);

        qcMapped->z(qubits[6], ancillaControls[4]);
        qcMapped->z(qubits[7], ancillaControls[4]);
        qcMapped->z(qubits[7], ancillaControls[5]);
        qcMapped->z(qubits[8], ancillaControls[5]);

        //z errors = indirectly via controlled x/CNOT
        qcMapped->x(qubits[0], ancillaControls[6]);
        qcMapped->x(qubits[1], ancillaControls[6]);
        qcMapped->x(qubits[2], ancillaControls[6]);
        qcMapped->x(qubits[3], ancillaControls[6]);
        qcMapped->x(qubits[4], ancillaControls[6]);
        qcMapped->x(qubits[5], ancillaControls[6]);

        qcMapped->x(qubits[3], ancillaControls[7]);
        qcMapped->x(qubits[4], ancillaControls[7]);
        qcMapped->x(qubits[5], ancillaControls[7]);
        qcMapped->x(qubits[6], ancillaControls[7]);
        qcMapped->x(qubits[7], ancillaControls[7]);
        qcMapped->x(qubits[8], ancillaControls[7]);

        for (dd::Qubit j: ancillaQubits) {
            qcMapped->h(j);
        }

        //MEASURE ancilla qubits
        for (std::size_t j = 0; j < 8; j++) {
            qcMapped->measure(ancillaQubits[j], clStart + j);
        }

        //CORRECT
        //x, i.e. bit flip errors
        writeClassicalControl(dd::Qubit(clStart), 2, 1, qc::X, i);
        writeClassicalControl(dd::Qubit(clStart), 2, 2, qc::X, i + 2 * nQubits);
        writeClassicalControl(dd::Qubit(clStart), 2, 3, qc::X, i + nQubits);

        writeClassicalControl(dd::Qubit(clStart + 2), 2, 1, qc::X, i + 3 * nQubits);
        writeClassicalControl(dd::Qubit(clStart + 2), 2, 2, qc::X, i + 5 * nQubits);
        writeClassicalControl(dd::Qubit(clStart + 2), 2, 3, qc::X, i + 4 * nQubits);

        writeClassicalControl(dd::Qubit(clStart + 4), 2, 1, qc::X, i + 6 * nQubits);
        writeClassicalControl(dd::Qubit(clStart + 4), 2, 2, qc::X, i + 8 * nQubits);
        writeClassicalControl(dd::Qubit(clStart + 4), 2, 3, qc::X, i + 7 * nQubits);

        //z, i.e. phase flip errors
        writeClassicalControl(dd::Qubit(clStart + 6), 2, 1, qc::Z, i);
        writeClassicalControl(dd::Qubit(clStart + 6), 2, 2, qc::Z, i + 6 * nQubits);
        writeClassicalControl(dd::Qubit(clStart + 6), 2, 3, qc::Z, i + 3 * nQubits);
    }
}

void Q9ShorEcc::writeDecoding() {
    if (isDecoded) {
        return;
    }
    const auto nQubits = qcOriginal->getNqubits();
    for (dd::Qubit i = 0; i < nQubits; i++) {
        std::array<dd::Control, 9> ci;
        for (dd::Qubit j = 0; j < 9; j++) {
            ci[j] = dd::Control{dd::Qubit(i + j * nQubits), dd::Control::Type::pos};
        }

        qcMapped->x(i + nQubits, ci[0]);
        qcMapped->x(i + 2 * nQubits, ci[0]);

        qcMapped->x(i + 4 * nQubits, ci[3]);
        qcMapped->x(i + 5 * nQubits, ci[3]);

        qcMapped->x(i + 7 * nQubits, ci[6]);
        qcMapped->x(i + 8 * nQubits, ci[6]);

        writeToffoli(i, i + nQubits, true, i + 2 * nQubits, true);
        writeToffoli(i + 3 * nQubits, i + 4 * nQubits, true, i + 5 * nQubits, true);
        writeToffoli(i + 6 * nQubits, i + 7 * nQubits, true, i + 8 * nQubits, true);

        qcMapped->h(i);
        qcMapped->h(static_cast<dd::Qubit>(i + 3 * nQubits));
        qcMapped->h(static_cast<dd::Qubit>(i + 6 * nQubits));

        qcMapped->x(i + 3 * nQubits, ci[0]);
        qcMapped->x(i + 6 * nQubits, ci[0]);
        writeToffoli(i, i + 3 * nQubits, true, i + 6 * nQubits, true);
    }
    isDecoded = true;
}

void Q9ShorEcc::mapGate(const qc::Operation& gate) {
    if (isDecoded && gate.getType() != qc::Measure && gate.getType() != qc::H) {
        writeEncoding();
    }
    const int nQubits = qcOriginal->getNqubits();
    auto      type    = qc::I;
    switch (gate.getType()) {
        case qc::I: break;
        case qc::X:
            type = qc::Z;
            break;
        case qc::Y:
            type = qc::Y;
            break;
        case qc::Z:
            type = qc::X;
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
            return;
        default:
            gateNotAvailableError(gate);
    }
    for (std::size_t t = 0; t < gate.getNtargets(); t++) {
        auto i = gate.getTargets()[t];

        if (gate.getNcontrols()) {
            //Q9Shor code: put H gate before and after each control point, i.e. "cx 0,1" becomes "h0; cz 0,1; h0"
            auto& ctrls = gate.getControls();
            for (int j = 0; j < 9; j++) {
                dd::Controls ctrls2;
                for (const auto& ct: ctrls) {
                    ctrls2.insert(dd::Control{dd::Qubit(ct.qubit + j * nQubits), ct.type});
                    qcMapped->h(static_cast<dd::Qubit>(ct.qubit + j * nQubits));
                }
                qcMapped->emplace_back<qc::StandardOperation>(qcMapped->getNqubits(), ctrls2, i + j * nQubits, type);
                for (const auto& ct: ctrls) {
                    qcMapped->h(static_cast<dd::Qubit>(ct.qubit + j * nQubits));
                }
            }
        } else {
            for (int j = 0; j < 9; j++) {
                qcMapped->emplace_back<qc::StandardOperation>(qcMapped->getNqubits(), i + j * nQubits, type);
            }
        }
    }
}
