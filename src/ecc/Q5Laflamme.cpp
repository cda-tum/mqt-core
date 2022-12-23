/*
 * This file is part of MQT QFR library which is released under the MIT license.
 * See file README.md or go to https://www.cda.cit.tum.de/research/quantum/ for more information.
 */

#include "ecc/Q5Laflamme.hpp"

void Q5Laflamme::initMappedCircuit() {
    //method is overridden because we need 2 kinds of classical measurement output registers
    qcOriginal->stripIdleQubits(true, false);
    qcMapped->addQubitRegister(getNOutputQubits(qcOriginal->getNqubits()));
    auto cRegs = qcOriginal->getCregs();
    for (auto const& [regName, regBits]: cRegs) {
        qcMapped->addClassicalRegister(regBits.second, regName.c_str());
    }
    qcMapped->addClassicalRegister(4, "qecc");
    qcMapped->addClassicalRegister(1, "encode");
}

void Q5Laflamme::writeEncoding() {
    Ecc::writeEncoding();

    const auto nQubits  = qcOriginal->getNqubits();
    const auto ancStart = nQubits * ecc.nRedundantQubits;
    const auto clEncode = qcOriginal->getNcbits() + 4; //encode

    for (dd::Qubit i = 0; i < nQubits; i++) {
        qcMapped->reset(static_cast<dd::Qubit>(ancStart));
    }

    for (dd::Qubit i = 0; i < nQubits; i++) {
        qcMapped->h(static_cast<dd::Qubit>(ancStart));
        qcMapped->z((i), dd::Control{static_cast<dd::Qubit>(ancStart), dd::Control::Type::pos});
        qcMapped->z(static_cast<dd::Qubit>(i + nQubits), dd::Control{static_cast<dd::Qubit>(ancStart), dd::Control::Type::pos});
        qcMapped->z(static_cast<dd::Qubit>(i + 2 * nQubits), dd::Control{static_cast<dd::Qubit>(ancStart), dd::Control::Type::pos});
        qcMapped->z(static_cast<dd::Qubit>(i + 3 * nQubits), dd::Control{static_cast<dd::Qubit>(ancStart), dd::Control::Type::pos});
        qcMapped->z(static_cast<dd::Qubit>(i + 4 * nQubits), dd::Control{static_cast<dd::Qubit>(ancStart), dd::Control::Type::pos});
        qcMapped->h(static_cast<dd::Qubit>(ancStart));
        qcMapped->measure(static_cast<dd::Qubit>(ancStart), clEncode);

        writeClassicalControlled(1, i, qc::OpType::X, static_cast<dd::Qubit>(clEncode), static_cast<dd::QubitCount>(1));
        writeClassicalControlled(1, i + nQubits, qc::OpType::X, static_cast<dd::Qubit>(clEncode), static_cast<dd::QubitCount>(1));
        writeClassicalControlled(1, i + 2 * nQubits, qc::OpType::X, static_cast<dd::Qubit>(clEncode), static_cast<dd::QubitCount>(1));
        writeClassicalControlled(1, i + 3 * nQubits, qc::OpType::X, static_cast<dd::Qubit>(clEncode), static_cast<dd::QubitCount>(1));
        writeClassicalControlled(1, i + 4 * nQubits, qc::OpType::X, static_cast<dd::Qubit>(clEncode), static_cast<dd::QubitCount>(1));
    }
    gatesWritten = true;
}

void Q5Laflamme::measureAndCorrect() {
    if (isDecoded) {
        return;
    }
    const auto nQubits    = static_cast<dd::Qubit>(qcOriginal->getNqubits());
    const auto ancStart   = static_cast<dd::Qubit>(nQubits * ecc.nRedundantQubits);
    const auto clAncStart = static_cast<dd::Qubit>(qcOriginal->getNcbits());

    for (dd::Qubit i = 0; i < nQubits; i++) {
        std::array<dd::Qubit, 5> qubits = {};
        for (dd::Qubit j = 0; j < 5; j++) {
            qubits[j] = static_cast<dd::Qubit>(i + j * nQubits);
        }

        qcMapped->reset(ancStart);
        qcMapped->reset(static_cast<dd::Qubit>(ancStart + 1));
        qcMapped->reset(static_cast<dd::Qubit>(ancStart + 2));
        qcMapped->reset(static_cast<dd::Qubit>(ancStart + 3));

        qcMapped->h(ancStart);
        qcMapped->h(static_cast<dd::Qubit>(ancStart + 1));
        qcMapped->h(static_cast<dd::Qubit>(ancStart + 2));
        qcMapped->h(static_cast<dd::Qubit>(ancStart + 3));

        auto c0 = dd::Control{static_cast<dd::Qubit>(ancStart), dd::Control::Type::pos};
        auto c1 = dd::Control{static_cast<dd::Qubit>(ancStart + 1), dd::Control::Type::pos};
        auto c2 = dd::Control{static_cast<dd::Qubit>(ancStart + 2), dd::Control::Type::pos};
        auto c3 = dd::Control{static_cast<dd::Qubit>(ancStart + 3), dd::Control::Type::pos};

        //traversal of matrix: "/"
        //K1: XZZXI
        //K2: IXZZX
        //K3: XIXZZ
        //K4: ZXIXZ

        qcMapped->x(qubits[0], c0);

        qcMapped->z(qubits[1], c0);
        //controlled-id(i, c1)

        qcMapped->z(qubits[2], c0);
        qcMapped->x(qubits[1], c1);
        qcMapped->x(qubits[0], c2);

        qcMapped->x(qubits[3], c0);
        qcMapped->z(qubits[2], c1);
        //controlled-id(i+1, c2)
        qcMapped->z(qubits[0], c3);

        //controlled-id(i+4, c0)
        qcMapped->z(qubits[3], c1);
        qcMapped->x(qubits[2], c2);
        qcMapped->x(qubits[1], c3);

        qcMapped->x(qubits[4], c1);
        qcMapped->z(qubits[3], c2);
        //controlled-id(i+2, c3)

        qcMapped->z(qubits[4], c2);
        qcMapped->x(qubits[3], c3);

        qcMapped->z(qubits[4], c3);

        qcMapped->h(ancStart);
        qcMapped->h(static_cast<dd::Qubit>(ancStart + 1));
        qcMapped->h(static_cast<dd::Qubit>(ancStart + 2));
        qcMapped->h(static_cast<dd::Qubit>(ancStart + 3));

        qcMapped->measure(ancStart, clAncStart);
        qcMapped->measure(static_cast<dd::Qubit>(ancStart + 1), clAncStart + 1);
        qcMapped->measure(static_cast<dd::Qubit>(ancStart + 2), clAncStart + 2);
        qcMapped->measure(static_cast<dd::Qubit>(ancStart + 3), clAncStart + 3);

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

void Q5Laflamme::writeClassicalControlledCorrect(dd::QubitCount value, dd::Qubit target, qc::OpType operationType) {
    writeClassicalControlled(value, target, operationType, static_cast<dd::Qubit>(qcOriginal->getNcbits()), static_cast<dd::QubitCount>(4));
}

void Q5Laflamme::writeClassicalControlled(dd::QubitCount value, dd::Qubit target, qc::OpType opType, dd::Qubit clStart, dd::QubitCount clCount) {
    std::unique_ptr<qc::Operation> op    = std::make_unique<qc::StandardOperation>(qcMapped->getNqubits(), target, opType);
    qcMapped->emplace_back<qc::ClassicControlledOperation>(op, std::make_pair(clStart, clCount), value);
}

void Q5Laflamme::writeDecoding() {
    if (isDecoded) {
        return;
    }
    const dd::QubitCount                      nQubits           = qcOriginal->getNqubits();
    const size_t                              clAncStart        = qcOriginal->getNcbits();
    static constexpr std::array<dd::Qubit, 8> correctionNeeded  = {1, 2, 4, 7, 8, 11, 13, 14}; //values with odd amount of '1' bits

    for (std::size_t i = 0; i < nQubits; i++) {
        //#|####
        //0|1111
        //odd amount of 1's -> x[0] = 1
        //measure from index 1 (not 0) to 4, =qubit 2 to 5

        qcMapped->measure(static_cast<dd::Qubit>(i + 1 * nQubits), clAncStart);
        qcMapped->measure(static_cast<dd::Qubit>(i + 2 * nQubits), clAncStart + 1);
        qcMapped->measure(static_cast<dd::Qubit>(i + 3 * nQubits), clAncStart + 2);
        qcMapped->measure(static_cast<dd::Qubit>(i + 4 * nQubits), clAncStart + 3);
        for (dd::Qubit const value: correctionNeeded) {
            writeClassicalControl(static_cast<dd::Qubit>(clAncStart), 4, value, qc::X, static_cast<dd::Qubit>(i));
        }
    }
    isDecoded = true;
}

void Q5Laflamme::mapGate(const qc::Operation& gate) {
    if (isDecoded && gate.getType() != qc::Measure && gate.getType() != qc::H) {
        writeEncoding();
    }
    const auto nQubits = qcOriginal->getNqubits();
    switch (gate.getType()) {
        case qc::I: break;
        case qc::X:
        case qc::Y:
        case qc::Z:
            for (std::size_t t = 0; t < gate.getNtargets(); t++) {
                auto i = gate.getTargets()[t];
                if (gate.getNcontrols() != 0U) {
                    gateNotAvailableError(gate);
                } else {
                    for (dd::Qubit j = 0; j < 5; j++) {
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
            if (const auto* measureGate = dynamic_cast<const qc::NonUnitaryOperation*>(&gate)) {
                for (std::size_t j = 0; j < measureGate->getNclassics(); j++) {
                    qcMapped->measure(static_cast<dd::Qubit>(measureGate->getClassics()[j]), measureGate->getTargets()[j]);
                }
            } else {
                throw std::runtime_error("Dynamic cast to NonUnitaryOperation failed.");
            }

            break;
        default:
            gateNotAvailableError(gate);
    }
}
