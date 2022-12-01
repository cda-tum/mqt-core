/*
* This file is part of JKQ QFR library which is released under the MIT license.
* See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
*/

#include "eccs/Q9SurfaceEcc.hpp"

void Q9SurfaceEcc::initMappedCircuit() {
    //method is overridden because we need 2 kinds of classical measurement output registers
    qcOriginal.stripIdleQubits(true, false);
    statistics.nInputQubits         = qcOriginal.getNqubits();
    statistics.nInputClassicalBits  = (int)qcOriginal.getNcbits();
    statistics.nOutputQubits        = qcOriginal.getNqubits() * ecc.nRedundantQubits + ecc.nCorrectingBits;
    statistics.nOutputClassicalBits = statistics.nInputClassicalBits + ecc.nCorrectingBits;
    qcMapped.addQubitRegister(statistics.nOutputQubits);
    auto cRegs = qcOriginal.getCregs();
    for (auto const& [regName, regBits]: cRegs) {
        qcMapped.addClassicalRegister(regBits.second, regName);
    }
    qcMapped.addClassicalRegister(4, "qeccX");
    qcMapped.addClassicalRegister(4, "qeccZ");
}

void Q9SurfaceEcc::writeEncoding() {
    if (!isDecoded) {
        return;
    }
    isDecoded = false;
    measureAndCorrect();
}

void Q9SurfaceEcc::measureAndCorrect() {
    if (isDecoded) {
        return;
    }
    const int  nQubits    = qcOriginal.getNqubits();
    const int  ancStart   = qcOriginal.getNqubits() * ecc.nRedundantQubits;
    const auto clAncStart = qcOriginal.getNcbits();
    for (int i = 0; i < nQubits; i++) {
        std::array<dd::Qubit, 9>   qubits          = {};
        std::array<dd::Qubit, 8>   ancillaQubits   = {};
        std::array<dd::Control, 8> ancillaControls = {};
        std::array<dd::Control, 9> controlQubits   = {};
        for (int j = 0; j < 9; j++) {
            qubits[j] = dd::Qubit(i + j * nQubits);
        }
        for (int j = 0; j < 8; j++) {
            ancillaQubits[j] = dd::Qubit(ancStart + j);
        }
        if (gatesWritten) {
            for (int j = 0; j < 8; j++) {
                qcMapped.reset(ancillaQubits[j]);
            }
        }
        for (int j = 0; j < 8; j++) {
            ancillaControls[j] = dd::Control{dd::Qubit(ancillaQubits[j]), dd::Control::Type::pos};
        }
        for (int j = 0; j < 9; j++) {
            controlQubits[j] = dd::Control{dd::Qubit(qubits[j]), dd::Control::Type::pos};
        }

        //X-type check on a0, a2, a5, a7: cx ancillaQubits->qubits
        //Z-type check on a1, a3, a4, a6: cz ancillaQubits->qubits = cx qubits->ancillaQubits, no hadamard gate
        qcMapped.h(ancillaQubits[0]);
        qcMapped.h(ancillaQubits[2]);
        qcMapped.h(ancillaQubits[5]);
        qcMapped.h(ancillaQubits[7]);

        qcMapped.x(ancillaQubits[6], controlQubits[8]);
        qcMapped.x(qubits[7], ancillaControls[5]);
        qcMapped.x(ancillaQubits[4], controlQubits[6]);
        qcMapped.x(ancillaQubits[3], controlQubits[4]);
        qcMapped.x(qubits[3], ancillaControls[2]);
        qcMapped.x(qubits[1], ancillaControls[0]);

        qcMapped.x(ancillaQubits[6], controlQubits[5]);
        qcMapped.x(ancillaQubits[4], controlQubits[3]);
        qcMapped.x(qubits[8], ancillaControls[5]);
        qcMapped.x(ancillaQubits[3], controlQubits[1]);
        qcMapped.x(qubits[4], ancillaControls[2]);
        qcMapped.x(qubits[2], ancillaControls[0]);

        qcMapped.x(qubits[6], ancillaControls[7]);
        qcMapped.x(qubits[4], ancillaControls[5]);
        qcMapped.x(ancillaQubits[4], controlQubits[7]);
        qcMapped.x(ancillaQubits[3], controlQubits[5]);
        qcMapped.x(qubits[0], ancillaControls[2]);
        qcMapped.x(ancillaQubits[1], controlQubits[3]);

        qcMapped.x(qubits[7], ancillaControls[7]);
        qcMapped.x(qubits[5], ancillaControls[5]);
        qcMapped.x(ancillaQubits[4], controlQubits[4]);
        qcMapped.x(ancillaQubits[3], controlQubits[2]);
        qcMapped.x(qubits[1], ancillaControls[2]);
        qcMapped.x(ancillaQubits[1], controlQubits[0]);

        qcMapped.h(ancillaQubits[0]);
        qcMapped.h(ancillaQubits[2]);
        qcMapped.h(ancillaQubits[5]);
        qcMapped.h(ancillaQubits[7]);

        qcMapped.measure(ancillaQubits[0], clAncStart);
        qcMapped.measure(ancillaQubits[2], clAncStart + 1);
        qcMapped.measure(ancillaQubits[5], clAncStart + 2);
        qcMapped.measure(ancillaQubits[7], clAncStart + 3);

        qcMapped.measure(ancillaQubits[1], clAncStart + 4);
        qcMapped.measure(ancillaQubits[3], clAncStart + 5);
        qcMapped.measure(ancillaQubits[4], clAncStart + 6);
        qcMapped.measure(ancillaQubits[6], clAncStart + 7);

        //logic
        writeClassicalControl(dd::Qubit(clAncStart), 4, 1, qc::Z, qubits[2]);  //ancillaQubits[0]
        writeClassicalControl(dd::Qubit(clAncStart), 4, 2, qc::Z, qubits[3]);  //ancillaQubits[2] (or qubits[0])
        writeClassicalControl(dd::Qubit(clAncStart), 4, 3, qc::Z, qubits[1]);  //ancillaQubits[0,2]
        writeClassicalControl(dd::Qubit(clAncStart), 4, 4, qc::Z, qubits[5]);  //ancillaQubits[5] (or qubits[8])
        writeClassicalControl(dd::Qubit(clAncStart), 4, 6, qc::Z, qubits[4]);  //ancillaQubits[2,5]
        writeClassicalControl(dd::Qubit(clAncStart), 4, 8, qc::Z, qubits[6]);  //ancillaQubits[7]
        writeClassicalControl(dd::Qubit(clAncStart), 4, 12, qc::Z, qubits[7]); //ancillaQubits[5,7]

        writeClassicalControl(dd::Qubit(clAncStart + 4), 4, 1, qc::X, qubits[0]);  //ancillaQubits[1]
        writeClassicalControl(dd::Qubit(clAncStart + 4), 4, 2, qc::X, qubits[1]);  //ancillaQubits[3] (or qubits[2])
        writeClassicalControl(dd::Qubit(clAncStart + 4), 4, 4, qc::X, qubits[7]);  //ancillaQubits[4] (or qubits[6])
        writeClassicalControl(dd::Qubit(clAncStart + 4), 4, 5, qc::X, qubits[3]);  //ancillaQubits[1,4]
        writeClassicalControl(dd::Qubit(clAncStart + 4), 4, 6, qc::X, qubits[4]);  //ancillaQubits[3,4]
        writeClassicalControl(dd::Qubit(clAncStart + 4), 4, 8, qc::X, qubits[8]);  //ancillaQubits[6]
        writeClassicalControl(dd::Qubit(clAncStart + 4), 4, 10, qc::X, qubits[5]); //ancillaQubits[3,6]
    }
    gatesWritten = true;
}

void Q9SurfaceEcc::writeDecoding() {
    if (isDecoded) {
        return;
    }
    const int nQubits = qcOriginal.getNqubits();
    for (int i = 0; i < nQubits; i++) {
        //measure 0, 4, 8. state = m0*m4*m8
        qcMapped.measure(dd::Qubit(i), i);
        qcMapped.measure(dd::Qubit(i + 4 * nQubits), i);
        qcMapped.measure(dd::Qubit(i + 8 * nQubits), i);
        qcMapped.x(dd::Qubit(i), dd::Control{dd::Qubit(i + 4 * nQubits), dd::Control::Type::pos});
        qcMapped.x(dd::Qubit(i), dd::Control{dd::Qubit(i + 8 * nQubits), dd::Control::Type::pos});
        qcMapped.measure(dd::Qubit(i), i);
    }
    isDecoded = true;
}

void Q9SurfaceEcc::mapGate(const qc::Operation& gate) {
    if (isDecoded && gate.getType() != qc::Measure) {
        writeEncoding();
    }
    const int nQubits = qcOriginal.getNqubits();
    dd::Qubit i;

    if (gate.getNcontrols() && gate.getType() != qc::Measure) {
        auto&        ctrls = gate.getControls();
        dd::Controls ctrls2;
        dd::Controls ctrls3;
        for (const auto& ct: ctrls) {
            ctrls2.insert(dd::Control{dd::Qubit(ct.qubit + 4 * nQubits), ct.type});
            ctrls3.insert(dd::Control{dd::Qubit(ct.qubit + 8 * nQubits), ct.type});
        }
        switch (gate.getType()) {
            case qc::I:
                break;
            case qc::X:
                for (std::size_t t = 0; t < gate.getNtargets(); t++) {
                    i = gate.getTargets()[t];
                    qcMapped.x(dd::Qubit(i + 2 * nQubits), ctrls);
                    qcMapped.x(dd::Qubit(i + 4 * nQubits), ctrls);
                    qcMapped.x(dd::Qubit(i + 6 * nQubits), ctrls);
                    qcMapped.x(dd::Qubit(i + 2 * nQubits), ctrls2);
                    qcMapped.x(dd::Qubit(i + 4 * nQubits), ctrls2);
                    qcMapped.x(dd::Qubit(i + 6 * nQubits), ctrls2);
                    qcMapped.x(dd::Qubit(i + 2 * nQubits), ctrls3);
                    qcMapped.x(dd::Qubit(i + 4 * nQubits), ctrls3);
                    qcMapped.x(dd::Qubit(i + 6 * nQubits), ctrls3);
                }
                break;
            case qc::Y:
                //Y = Z X
                for (std::size_t t = 0; t < gate.getNtargets(); t++) {
                    i = gate.getTargets()[t];
                    qcMapped.z(dd::Qubit(i), ctrls);
                    qcMapped.z(dd::Qubit(i + 4 * nQubits), ctrls);
                    qcMapped.z(dd::Qubit(i + 8 * nQubits), ctrls);
                    qcMapped.x(dd::Qubit(i + 2 * nQubits), ctrls);
                    qcMapped.x(dd::Qubit(i + 4 * nQubits), ctrls);
                    qcMapped.x(dd::Qubit(i + 6 * nQubits), ctrls);
                    qcMapped.z(dd::Qubit(i), ctrls2);
                    qcMapped.z(dd::Qubit(i + 4 * nQubits), ctrls2);
                    qcMapped.z(dd::Qubit(i + 8 * nQubits), ctrls2);
                    qcMapped.x(dd::Qubit(i + 2 * nQubits), ctrls2);
                    qcMapped.x(dd::Qubit(i + 4 * nQubits), ctrls2);
                    qcMapped.x(dd::Qubit(i + 6 * nQubits), ctrls2);
                    qcMapped.z(dd::Qubit(i), ctrls3);
                    qcMapped.z(dd::Qubit(i + 4 * nQubits), ctrls3);
                    qcMapped.z(dd::Qubit(i + 8 * nQubits), ctrls3);
                    qcMapped.x(dd::Qubit(i + 2 * nQubits), ctrls3);
                    qcMapped.x(dd::Qubit(i + 4 * nQubits), ctrls3);
                    qcMapped.x(dd::Qubit(i + 6 * nQubits), ctrls3);
                }
                break;
            case qc::Z:
                for (std::size_t t = 0; t < gate.getNtargets(); t++) {
                    i = gate.getTargets()[t];
                    qcMapped.z(dd::Qubit(i), ctrls);
                    qcMapped.z(dd::Qubit(i + 4 * nQubits), ctrls);
                    qcMapped.z(dd::Qubit(i + 8 * nQubits), ctrls);
                    qcMapped.z(dd::Qubit(i), ctrls2);
                    qcMapped.z(dd::Qubit(i + 4 * nQubits), ctrls2);
                    qcMapped.z(dd::Qubit(i + 8 * nQubits), ctrls2);
                    qcMapped.z(dd::Qubit(i), ctrls3);
                    qcMapped.z(dd::Qubit(i + 4 * nQubits), ctrls3);
                    qcMapped.z(dd::Qubit(i + 8 * nQubits), ctrls3);
                }
                break;
            default:
                gateNotAvailableError(gate);
        }
    } else {
        switch (gate.getType()) {
            case qc::I:
                break;
            case qc::X:
                for (std::size_t t = 0; t < gate.getNtargets(); t++) {
                    i = gate.getTargets()[t];
                    qcMapped.x(dd::Qubit(i + 2 * nQubits));
                    qcMapped.x(dd::Qubit(i + 4 * nQubits));
                    qcMapped.x(dd::Qubit(i + 6 * nQubits));
                }
                break;
            case qc::H:
                for (std::size_t t = 0; t < gate.getNtargets(); t++) {
                    i = gate.getTargets()[t];
                    for (int j = 0; j < 9; j++) {
                        qcMapped.h(dd::Qubit(i + j * nQubits));
                    }

                    qcMapped.swap(dd::Qubit(i), dd::Qubit(i + 6 * nQubits));
                    qcMapped.swap(dd::Qubit(i + 3 * nQubits), dd::Qubit(i + 7 * nQubits));
                    qcMapped.swap(dd::Qubit(i + 2 * nQubits), dd::Qubit(i + 8 * nQubits));
                    qcMapped.swap(dd::Qubit(i + nQubits), dd::Qubit(i + 5 * nQubits));
                }
                break;
            case qc::Y:
                //Y = Z X
                for (std::size_t t = 0; t < gate.getNtargets(); t++) {
                    i = gate.getTargets()[t];
                    qcMapped.z(dd::Qubit(i));
                    qcMapped.z(dd::Qubit(i + 4 * nQubits));
                    qcMapped.z(dd::Qubit(i + 8 * nQubits));
                    qcMapped.x(dd::Qubit(i + 2 * nQubits));
                    qcMapped.x(dd::Qubit(i + 4 * nQubits));
                    qcMapped.x(dd::Qubit(i + 6 * nQubits));
                }
                break;
            case qc::Z:
                for (std::size_t t = 0; t < gate.getNtargets(); t++) {
                    i = gate.getTargets()[t];
                    qcMapped.z(dd::Qubit(i));
                    qcMapped.z(dd::Qubit(i + 4 * nQubits));
                    qcMapped.z(dd::Qubit(i + 8 * nQubits));
                }
                break;
            case qc::Measure:
                if (!isDecoded) {
                    measureAndCorrect();
                    writeDecoding();
                }
                if (auto measureGate = dynamic_cast<const qc::NonUnitaryOperation*>(&gate)) {
                    for (std::size_t j = 0; j < measureGate->getNclassics(); j++) {
                        qcMapped.measure(measureGate->getTargets()[j], measureGate->getClassics()[j]);
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
