/*
 * This file is part of MQT QFR library which is released under the MIT license.
 * See file README.md or go to https://www.cda.cit.tum.de/research/quantum/ for more information.
 */

#include "ecc/Q9Shor.hpp"

void Q9Shor::initMappedCircuit() {
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

void Q9Shor::writeEncoding() {
    if (!isDecoded) {
        return;
    }
    isDecoded          = false;
    const auto nQubits = qcOriginal->getNqubits();
    for (dd::Qubit i = 0; i < nQubits; i++) {
        dd::Control const ci = {static_cast<dd::Qubit>(i), dd::Control::Type::pos};
        qcMapped->x(static_cast<dd::Qubit>(i + 3 * nQubits), ci);
        qcMapped->x(static_cast<dd::Qubit>(i + 6 * nQubits), ci);

        qcMapped->h(static_cast<dd::Qubit>(i));
        qcMapped->h(static_cast<dd::Qubit>(i + 3 * nQubits));
        qcMapped->h(static_cast<dd::Qubit>(i + 6 * nQubits));

        dd::Control const ci3 = {static_cast<dd::Qubit>(i + 3 * nQubits), dd::Control::Type::pos};
        dd::Control const ci6 = {static_cast<dd::Qubit>(i + 6 * nQubits), dd::Control::Type::pos};
        qcMapped->x(static_cast<dd::Qubit>(i + nQubits), ci);
        qcMapped->x(static_cast<dd::Qubit>(i + 2 * nQubits), ci);
        qcMapped->x(static_cast<dd::Qubit>(i + 4 * nQubits), ci3);
        qcMapped->x(static_cast<dd::Qubit>(i + 5 * nQubits), ci3);
        qcMapped->x(static_cast<dd::Qubit>(i + 7 * nQubits), ci6);
        qcMapped->x(static_cast<dd::Qubit>(i + 8 * nQubits), ci6);
    }
    gatesWritten = true;
}

void Q9Shor::measureAndCorrect() {
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
            qubits.at(j) = static_cast<dd::Qubit>(i + j * nQubits);
        }
        for (std::size_t j = 0; j < 8; j++) {
            ancillaQubits.at(j) = static_cast<dd::Qubit>(ecc.nRedundantQubits * nQubits + j);
            qcMapped->reset(ancillaQubits.at(j));
        }
        for (std::size_t j = 0; j < 8; j++) {
            ancillaControls.at(j) = dd::Control{static_cast<dd::Qubit>(ancillaQubits.at(j)), dd::Control::Type::pos};
        }
        for (std::size_t j = 0; j < 8; j++) {
            negativeAncillaControls.at(j) = dd::Control{static_cast<dd::Qubit>(ancillaQubits.at(j)), dd::Control::Type::neg};
        }

        // PREPARE measurements --------------------------------------------------------
        for (dd::Qubit const j: ancillaQubits) {
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

        for (dd::Qubit const j: ancillaQubits) {
            qcMapped->h(j);
        }

        //MEASURE ancilla qubits
        for (std::size_t j = 0; j < 8; j++) {
            qcMapped->measure(ancillaQubits.at(j), clStart + j);
        }

        //CORRECT
        //x, i.e. bit flip errors
        writeClassicalControl(static_cast<dd::Qubit>(clStart), 2, 1, qc::X, i);
        writeClassicalControl(static_cast<dd::Qubit>(clStart), 2, 2, qc::X, static_cast<dd::Qubit>(i + 2 * nQubits));
        writeClassicalControl(static_cast<dd::Qubit>(clStart), 2, 3, qc::X, static_cast<dd::Qubit>(i + nQubits));

        writeClassicalControl(static_cast<dd::Qubit>(clStart + 2), 2, 1, qc::X, static_cast<dd::Qubit>(i + 3 * nQubits));
        writeClassicalControl(static_cast<dd::Qubit>(clStart + 2), 2, 2, qc::X, static_cast<dd::Qubit>(i + 5 * nQubits));
        writeClassicalControl(static_cast<dd::Qubit>(clStart + 2), 2, 3, qc::X, static_cast<dd::Qubit>(i + 4 * nQubits));

        writeClassicalControl(static_cast<dd::Qubit>(clStart + 4), 2, 1, qc::X, static_cast<dd::Qubit>(i + 6 * nQubits));
        writeClassicalControl(static_cast<dd::Qubit>(clStart + 4), 2, 2, qc::X, static_cast<dd::Qubit>(i + 8 * nQubits));
        writeClassicalControl(static_cast<dd::Qubit>(clStart + 4), 2, 3, qc::X, static_cast<dd::Qubit>(i + 7 * nQubits));

        //z, i.e. phase flip errors
        writeClassicalControl(static_cast<dd::Qubit>(clStart + 6), 2, 1, qc::Z, i);
        writeClassicalControl(static_cast<dd::Qubit>(clStart + 6), 2, 2, qc::Z, static_cast<dd::Qubit>(i + 6 * nQubits));
        writeClassicalControl(static_cast<dd::Qubit>(clStart + 6), 2, 3, qc::Z, static_cast<dd::Qubit>(i + 3 * nQubits));
    }
}

void Q9Shor::writeDecoding() {
    if (isDecoded) {
        return;
    }
    const auto nQubits = qcOriginal->getNqubits();
    for (dd::Qubit i = 0; i < nQubits; i++) {
        std::array<dd::Control, 9> ci;
        for (dd::Qubit j = 0; j < 9; j++) {
            ci.at(j) = dd::Control{static_cast<dd::Qubit>(i + j * nQubits), dd::Control::Type::pos};
        }

        qcMapped->x(static_cast<dd::Qubit>(i + nQubits), ci[0]);
        qcMapped->x(static_cast<dd::Qubit>(i + 2 * nQubits), ci[0]);

        qcMapped->x(static_cast<dd::Qubit>(i + 4 * nQubits), ci[3]);
        qcMapped->x(static_cast<dd::Qubit>(i + 5 * nQubits), ci[3]);

        qcMapped->x(static_cast<dd::Qubit>(i + 7 * nQubits), ci[6]);
        qcMapped->x(static_cast<dd::Qubit>(i + 8 * nQubits), ci[6]);

        ccx(i, static_cast<dd::Qubit>(i + nQubits), true, static_cast<dd::Qubit>(i + 2 * nQubits), true);
        ccx(static_cast<dd::Qubit>(i + 3 * nQubits), static_cast<dd::Qubit>(i + 4 * nQubits), true, static_cast<dd::Qubit>(i + 5 * nQubits), true);
        ccx(static_cast<dd::Qubit>(i + 6 * nQubits), static_cast<dd::Qubit>(i + 7 * nQubits), true, static_cast<dd::Qubit>(i + 8 * nQubits), true);

        qcMapped->h(i);
        qcMapped->h(static_cast<dd::Qubit>(i + 3 * nQubits));
        qcMapped->h(static_cast<dd::Qubit>(i + 6 * nQubits));

        qcMapped->x(static_cast<dd::Qubit>(i + 3 * nQubits), ci[0]);
        qcMapped->x(static_cast<dd::Qubit>(i + 6 * nQubits), ci[0]);
        ccx(i, static_cast<dd::Qubit>(i + 3 * nQubits), true, static_cast<dd::Qubit>(i + 6 * nQubits), true);
    }
    isDecoded = true;
}

void Q9Shor::mapGate(const qc::Operation& gate) {
    if (isDecoded && gate.getType() != qc::Measure && gate.getType() != qc::H) {
        writeEncoding();
    }
    const auto nQubits = qcOriginal->getNqubits();
    auto       type    = qc::I;
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
            if (const auto* measureGate = dynamic_cast<const qc::NonUnitaryOperation*>(&gate)) {
                for (std::size_t j = 0; j < measureGate->getNclassics(); j++) {
                    qcMapped->measure(measureGate->getTargets().at(j), measureGate->getClassics().at(j));
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

        if (gate.getNcontrols() != 0U) {
            //Q9Shor code: put H gate before and after each control point, i.e. "cx 0,1" becomes "h0; cz 0,1; h0"
            const auto& ctrls = gate.getControls();
            for (size_t j = 0; j < 9; j++) {
                dd::Controls ctrls2;
                for (const auto& ct: ctrls) {
                    ctrls2.insert(dd::Control{static_cast<dd::Qubit>(ct.qubit + j * nQubits), ct.type});
                    qcMapped->h(static_cast<dd::Qubit>(ct.qubit + j * nQubits));
                }
                qcMapped->emplace_back<qc::StandardOperation>(qcMapped->getNqubits(), ctrls2, i + j * nQubits, type);
                for (const auto& ct: ctrls) {
                    qcMapped->h(static_cast<dd::Qubit>(ct.qubit + j * nQubits));
                }
            }
        } else {
            for (size_t j = 0; j < 9; j++) {
                qcMapped->emplace_back<qc::StandardOperation>(qcMapped->getNqubits(), i + j * nQubits, type);
            }
        }
    }
}
