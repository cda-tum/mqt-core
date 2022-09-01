/*
 * This file is part of JKQ QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#include "eccs/Q5LaflammeEcc.hpp"

//5 data qubits, 4 for measuring
Q5LaflammeEcc::Q5LaflammeEcc(qc::QuantumComputation& qc, int measureFq, bool decomposeMC, bool cliffOnly):
    Ecc({ID::Q5Laflamme, 5, 4, Q5LaflammeEcc::getName()}, qc, measureFq, decomposeMC, cliffOnly) {}

void Q5LaflammeEcc::initMappedCircuit() {
    //method is overridden because we need 2 kinds of classical measurement output registers
    qc.stripIdleQubits(true, false);
    statistics.nInputQubits         = qc.getNqubits();
    statistics.nInputClassicalBits  = (int)qc.getNcbits();
    statistics.nOutputQubits        = qc.getNqubits() * ecc.nRedundantQubits + ecc.nCorrectingBits;
    statistics.nOutputClassicalBits = statistics.nInputClassicalBits + ecc.nCorrectingBits;
    qcMapped.addQubitRegister(statistics.nOutputQubits);
    qcMapped.addClassicalRegister(statistics.nInputClassicalBits);
    qcMapped.addClassicalRegister(4, "qecc");
    qcMapped.addClassicalRegister(1, "encode");
}

void Q5LaflammeEcc::writeEncoding() {
    if (!isDecoded) {
        return;
    }
    isDecoded = false;
    measureAndCorrect();
    const int nQubits  = qc.getNqubits();
    const int ancStart = nQubits * ecc.nRedundantQubits;
    const int clEncode = qc.getNcbits() + 4; //encode
    for (int i = 0; i < nQubits; i++) {
        qcMapped.reset(dd::Qubit(ancStart));
        qcMapped.h(dd::Qubit(ancStart));
        writeZ(dd::Qubit(i), dd::Control{dd::Qubit(ancStart), dd::Control::Type::pos});
        writeZ(dd::Qubit(i + nQubits), dd::Control{dd::Qubit(ancStart), dd::Control::Type::pos});
        writeZ(dd::Qubit(i + 2 * nQubits), dd::Control{dd::Qubit(ancStart), dd::Control::Type::pos});
        writeZ(dd::Qubit(i + 3 * nQubits), dd::Control{dd::Qubit(ancStart), dd::Control::Type::pos});
        writeZ(dd::Qubit(i + 4 * nQubits), dd::Control{dd::Qubit(ancStart), dd::Control::Type::pos});
        qcMapped.h(dd::Qubit(ancStart));
        qcMapped.measure(dd::Qubit(ancStart), clEncode);

        writeClassicalControlled(1, i, qc::OpType::X, dd::Qubit(clEncode), dd::QubitCount(1));
        writeClassicalControlled(1, i + nQubits, qc::OpType::X, dd::Qubit(clEncode), dd::QubitCount(1));
        writeClassicalControlled(1, i + 2 * nQubits, qc::OpType::X, dd::Qubit(clEncode), dd::QubitCount(1));
        writeClassicalControlled(1, i + 3 * nQubits, qc::OpType::X, dd::Qubit(clEncode), dd::QubitCount(1));
        writeClassicalControlled(1, i + 4 * nQubits, qc::OpType::X, dd::Qubit(clEncode), dd::QubitCount(1));
    }
}

void Q5LaflammeEcc::measureAndCorrect() {
    if (isDecoded) {
        return;
    }
    const int nQubits    = qc.getNqubits();
    const int ancStart   = nQubits * ecc.nRedundantQubits;
    const int clAncStart = qc.getNcbits();

    for (int i = 0; i < nQubits; i++) {
        int q[5] = {};
        for (int j = 0; j < 5; j++) { q[j] = i + j * nQubits; }

        qcMapped.reset(ancStart);
        qcMapped.reset(ancStart + 1);
        qcMapped.reset(ancStart + 2);
        qcMapped.reset(ancStart + 3);

        qcMapped.h(ancStart);
        qcMapped.h(ancStart + 1);
        qcMapped.h(ancStart + 2);
        qcMapped.h(ancStart + 3);

        auto c0 = dd::Control{dd::Qubit(ancStart), dd::Control::Type::pos};
        auto c1 = dd::Control{dd::Qubit(ancStart + 1), dd::Control::Type::pos};
        auto c2 = dd::Control{dd::Qubit(ancStart + 2), dd::Control::Type::pos};
        auto c3 = dd::Control{dd::Qubit(ancStart + 3), dd::Control::Type::pos};

        //traversal of matrix: "/"
        //K1: XZZXI
        //K2: IXZZX
        //K3: XIXZZ
        //K4: ZXIXZ

        writeX(q[0], c0);

        writeZ(q[1], c0);
        //controlled-id(i, c1)

        writeZ(q[2], c0);
        writeX(q[1], c1);
        writeX(q[0], c2);

        writeX(q[3], c0);
        writeZ(q[2], c1);
        //controlled-id(i+1, c2)
        writeZ(q[0], c3);

        //controlled-id(i+4, c0)
        writeZ(q[3], c1);
        writeX(q[2], c2);
        writeX(q[1], c3);

        writeX(q[4], c1);
        writeZ(q[3], c2);
        //controlled-id(i+2, c3)

        writeZ(q[4], c2);
        writeX(q[3], c3);

        writeZ(q[4], c3);

        qcMapped.h(ancStart);
        qcMapped.h(ancStart + 1);
        qcMapped.h(ancStart + 2);
        qcMapped.h(ancStart + 3);

        qcMapped.measure(ancStart, clAncStart);
        qcMapped.measure(ancStart + 1, clAncStart + 1);
        qcMapped.measure(ancStart + 2, clAncStart + 2);
        qcMapped.measure(ancStart + 3, clAncStart + 3);

        writeClassicalControlledCorrect(1, q[1], qc::X);
        writeClassicalControlledCorrect(2, q[4], qc::Z);
        writeClassicalControlledCorrect(3, q[2], qc::X);
        writeClassicalControlledCorrect(4, q[2], qc::Z);
        writeClassicalControlledCorrect(5, q[0], qc::Z);
        writeClassicalControlledCorrect(6, q[3], qc::X);
        writeClassicalControlledCorrect(7, q[2], qc::Y);
        writeClassicalControlledCorrect(8, q[0], qc::X);
        writeClassicalControlledCorrect(9, q[3], qc::Z);
        writeClassicalControlledCorrect(10, q[1], qc::Z);
        writeClassicalControlledCorrect(11, q[1], qc::Y);
        writeClassicalControlledCorrect(12, q[4], qc::X);
        writeClassicalControlledCorrect(13, q[0], qc::Y);
        writeClassicalControlledCorrect(14, q[4], qc::Y);
        writeClassicalControlledCorrect(15, q[3], qc::Y);
    }
}

void Q5LaflammeEcc::writeClassicalControlledCorrect(const unsigned int value, int target, qc::OpType optype) {
    writeClassicalControlled(value, target, optype, dd::Qubit(statistics.nInputClassicalBits), dd::QubitCount(4));
}

void Q5LaflammeEcc::writeClassicalControlled(const unsigned int value, int target, qc::OpType optype, dd::Qubit clStart, dd::QubitCount clCount) {
    std::unique_ptr<qc::Operation> op    = std::make_unique<qc::StandardOperation>(qcMapped.getNqubits(), target, optype);
    const auto                     pair_ = std::make_pair(clStart, clCount);
    qcMapped.emplace_back<qc::ClassicControlledOperation>(op, pair_, value);
}

void Q5LaflammeEcc::writeDecoding() {
    if (isDecoded) {
        return;
    }
    const int    nQubits             = qc.getNqubits();
    const int    clAncStart          = static_cast<int>(qc.getNcbits());
    unsigned int correction_needed[] = {1, 2, 4, 7, 8, 11, 13, 14}; //values with odd amount of '1' bits

    for (int i = 0; i < nQubits; i++) {
        //#|####
        //0|1111
        //odd amount of 1's -> x[0] = 1
        //measure from index 1 (not 0) to 4, =qubit 2 to 5

        qcMapped.measure(static_cast<dd::Qubit>(i + 1 * nQubits), clAncStart);
        qcMapped.measure(static_cast<dd::Qubit>(i + 2 * nQubits), clAncStart + 1);
        qcMapped.measure(static_cast<dd::Qubit>(i + 3 * nQubits), clAncStart + 2);
        qcMapped.measure(static_cast<dd::Qubit>(i + 4 * nQubits), clAncStart + 3);
        for (unsigned int value: correction_needed) {
            writeClassicalControl(dd::Qubit(clAncStart), dd::QubitCount(4), value, qc::X, i);
        }
    }
    isDecoded = true;
}

void Q5LaflammeEcc::mapGate(const std::unique_ptr<qc::Operation>& gate, qc::QuantumComputation& qc) {
    if (isDecoded && gate.get()->getType() != qc::Measure && gate.get()->getType() != qc::H) {
        writeEncoding();
    }
    const int                nQubits     = qc.getNqubits();
    qc::NonUnitaryOperation* measureGate = nullptr;
    switch (gate.get()->getType()) {
        case qc::I: break;
        case qc::X:
        case qc::H:
        case qc::Y:
        case qc::Z:
            for (std::size_t t = 0; t < gate.get()->getNtargets(); t++) {
                int i = gate.get()->getTargets()[t];
                if (gate.get()->getNcontrols()) {
                    gateNotAvailableError(gate);
                } else {
                    for (int j = 0; j < 5; j++) {
                        writeGeneric(i + j * nQubits, gate.get()->getType());
                    }
                }
            }
            break;
        case qc::Measure:
            if (!isDecoded) {
                measureAndCorrect();
                writeDecoding();
            }
            measureGate = (qc::NonUnitaryOperation*)gate.get();
            for (std::size_t j = 0; j < measureGate->getNclassics(); j++) {
                qcMapped.measure(measureGate->getClassics()[j], measureGate->getTargets()[j]);
            }
            break;
        case qc::S:
        case qc::Sdag:
        case qc::T:
        case qc::Tdag:
        case qc::V:
        case qc::Vdag:
        case qc::U3:
        case qc::U2:
        case qc::Phase:
        case qc::SX:
        case qc::SXdag:
        case qc::RX:
        case qc::RY:
        case qc::RZ:
        case qc::SWAP:
        case qc::iSWAP:
        case qc::Peres:
        case qc::Peresdag:
        case qc::Compound:
        case qc::ClassicControlled:
        default:
            gateNotAvailableError(gate);
    }
}
