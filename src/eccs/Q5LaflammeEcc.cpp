/*
 * This file is part of JKQ QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#include "eccs/Q5LaflammeEcc.hpp"

void Q5LaflammeEcc::initMappedCircuit() {
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
    qcMapped.addClassicalRegister(4, "qecc");
    qcMapped.addClassicalRegister(1, "encode");
}

void Q5LaflammeEcc::writeEncoding() {
    if (!isDecoded) {
        return;
    }
    isDecoded = false;
    measureAndCorrect();
    const auto nQubits  = qcOriginal.getNqubits();
    const auto ancStart = nQubits * ecc.nRedundantQubits;
    const auto clEncode = qcOriginal.getNcbits() + 4; //encode

    for (int i = 0; i < nQubits; i++) {
        qcMapped.reset(dd::Qubit(ancStart));
    }

    for (int i = 0; i < nQubits; i++) {
        qcMapped.h(dd::Qubit(ancStart));
        qcMapped.z(dd::Qubit(i), dd::Control{dd::Qubit(ancStart), dd::Control::Type::pos});
        qcMapped.z(dd::Qubit(i + nQubits), dd::Control{dd::Qubit(ancStart), dd::Control::Type::pos});
        qcMapped.z(dd::Qubit(i + 2 * nQubits), dd::Control{dd::Qubit(ancStart), dd::Control::Type::pos});
        qcMapped.z(dd::Qubit(i + 3 * nQubits), dd::Control{dd::Qubit(ancStart), dd::Control::Type::pos});
        qcMapped.z(dd::Qubit(i + 4 * nQubits), dd::Control{dd::Qubit(ancStart), dd::Control::Type::pos});
        qcMapped.h(dd::Qubit(ancStart));
        qcMapped.measure(dd::Qubit(ancStart), clEncode);

        writeClassicalControlled(1, i, qc::OpType::X, dd::Qubit(clEncode), dd::QubitCount(1));
        writeClassicalControlled(1, i + nQubits, qc::OpType::X, dd::Qubit(clEncode), dd::QubitCount(1));
        writeClassicalControlled(1, i + 2 * nQubits, qc::OpType::X, dd::Qubit(clEncode), dd::QubitCount(1));
        writeClassicalControlled(1, i + 3 * nQubits, qc::OpType::X, dd::Qubit(clEncode), dd::QubitCount(1));
        writeClassicalControlled(1, i + 4 * nQubits, qc::OpType::X, dd::Qubit(clEncode), dd::QubitCount(1));
    }
    gatesWritten = true;
}

void Q5LaflammeEcc::measureAndCorrect() {
    if (isDecoded) {
        return;
    }
    const auto nQubits    = static_cast<dd::Qubit>(qcOriginal.getNqubits());
    const auto ancStart   = static_cast<dd::Qubit>(nQubits * ecc.nRedundantQubits);
    const auto clAncStart = static_cast<dd::Qubit>(qcOriginal.getNcbits());

    for (dd::Qubit i = 0; i < nQubits; i++) {
        std::array<dd::Qubit, 5> qubits = {};
        for (dd::Qubit j = 0; j < 5; j++) {
            qubits[j] = static_cast<dd::Qubit>(i + j * nQubits);
        }

        qcMapped.reset(ancStart);
        qcMapped.reset(static_cast<dd::Qubit>(ancStart + 1));
        qcMapped.reset(static_cast<dd::Qubit>(ancStart + 2));
        qcMapped.reset(static_cast<dd::Qubit>(ancStart + 3));

        qcMapped.h(ancStart);
        qcMapped.h(static_cast<dd::Qubit>(ancStart + 1));
        qcMapped.h(static_cast<dd::Qubit>(ancStart + 2));
        qcMapped.h(static_cast<dd::Qubit>(ancStart + 3));

        auto c0 = dd::Control{dd::Qubit(ancStart), dd::Control::Type::pos};
        auto c1 = dd::Control{dd::Qubit(ancStart + 1), dd::Control::Type::pos};
        auto c2 = dd::Control{dd::Qubit(ancStart + 2), dd::Control::Type::pos};
        auto c3 = dd::Control{dd::Qubit(ancStart + 3), dd::Control::Type::pos};

        //traversal of matrix: "/"
        //K1: XZZXI
        //K2: IXZZX
        //K3: XIXZZ
        //K4: ZXIXZ

        qcMapped.x(qubits[0], c0);

        qcMapped.z(qubits[1], c0);
        //controlled-id(i, c1)

        qcMapped.z(qubits[2], c0);
        qcMapped.x(qubits[1], c1);
        qcMapped.x(qubits[0], c2);

        qcMapped.x(qubits[3], c0);
        qcMapped.z(qubits[2], c1);
        //controlled-id(i+1, c2)
        qcMapped.z(qubits[0], c3);

        //controlled-id(i+4, c0)
        qcMapped.z(qubits[3], c1);
        qcMapped.x(qubits[2], c2);
        qcMapped.x(qubits[1], c3);

        qcMapped.x(qubits[4], c1);
        qcMapped.z(qubits[3], c2);
        //controlled-id(i+2, c3)

        qcMapped.z(qubits[4], c2);
        qcMapped.x(qubits[3], c3);

        qcMapped.z(qubits[4], c3);

        qcMapped.h(ancStart);
        qcMapped.h(static_cast<dd::Qubit>(ancStart + 1));
        qcMapped.h(static_cast<dd::Qubit>(ancStart + 2));
        qcMapped.h(static_cast<dd::Qubit>(ancStart + 3));

        qcMapped.measure(ancStart, clAncStart);
        qcMapped.measure(static_cast<dd::Qubit>(ancStart + 1), clAncStart + 1);
        qcMapped.measure(static_cast<dd::Qubit>(ancStart + 2), clAncStart + 2);
        qcMapped.measure(static_cast<dd::Qubit>(ancStart + 3), clAncStart + 3);

        writeClassicalControlledCorrect(1, qubits[1], qc::X);
        writeClassicalControlledCorrect(2, qubits[4], qc::Z);
        writeClassicalControlledCorrect(3, qubits[2], qc::X);
        writeClassicalControlledCorrect(4, qubits[2], qc::Z);
        writeClassicalControlledCorrect(5, qubits[0], qc::Z);
        writeClassicalControlledCorrect(6, qubits[3], qc::X);
        writeClassicalControlledCorrect(7, qubits[2], qc::Y);
        writeClassicalControlledCorrect(8, qubits[0], qc::X);
        writeClassicalControlledCorrect(9, qubits[3], qc::Z);
        writeClassicalControlledCorrect(10, qubits[1], qc::Z);
        writeClassicalControlledCorrect(11, qubits[1], qc::Y);
        writeClassicalControlledCorrect(12, qubits[4], qc::X);
        writeClassicalControlledCorrect(13, qubits[0], qc::Y);
        writeClassicalControlledCorrect(14, qubits[4], qc::Y);
        writeClassicalControlledCorrect(15, qubits[3], qc::Y);
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
    const int          nQubits           = qcOriginal.getNqubits();
    const auto         clAncStart        = static_cast<int>(qcOriginal.getNcbits());
    std::array<int, 8> correction_needed = {1, 2, 4, 7, 8, 11, 13, 14}; //values with odd amount of '1' bits

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

void Q5LaflammeEcc::mapGate(const qc::Operation& gate) {
    if (isDecoded && gate.getType() != qc::Measure && gate.getType() != qc::H) {
        writeEncoding();
    }
    const auto nQubits = qcOriginal.getNqubits();
    switch (gate.getType()) {
        case qc::I: break;
        case qc::X:
        case qc::Y:
        case qc::Z:
            for (std::size_t t = 0; t < gate.getNtargets(); t++) {
                auto i = gate.getTargets()[t];
                if (gate.getNcontrols()) {
                    gateNotAvailableError(gate);
                } else {
                    for (int j = 0; j < 5; j++) {
                        qcMapped.emplace_back<qc::StandardOperation>(qcMapped.getNqubits(), static_cast<dd::Qubit>(i + j * nQubits), gate.getType());
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
                    qcMapped.measure(static_cast<dd::Qubit>(measureGate->getClassics()[j]), measureGate->getTargets()[j]);
                }
            } else {
                throw std::runtime_error("Dynamic cast to NonUnitaryOperation failed.");
            }

            break;
        default:
            gateNotAvailableError(gate);
    }
}
