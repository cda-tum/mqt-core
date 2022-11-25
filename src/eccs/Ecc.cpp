/*
 * This file is part of JKQ QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#include "eccs/Ecc.hpp"

Ecc::Ecc(Info ecc, qc::QuantumComputation& qc, int measureFrequency):
    ecc(std::move(ecc)),
    qcOriginal(qc), measureFrequency(measureFrequency) {
}

void Ecc::initMappedCircuit() {
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

    if (ecc.nCorrectingBits > 0) {
        qcMapped.addClassicalRegister(ecc.nCorrectingBits, "qecc");
    }
}

qc::QuantumComputation& Ecc::apply() {
    initMappedCircuit();

    writeEncoding();
    isDecoded = false;

    long nInputGates = 0;
    for (const auto& gate: qcOriginal) {
        nInputGates++;
        mapGate(*gate);
        if (measureFrequency > 0 && nInputGates % measureFrequency == 0) {
            measureAndCorrect();
        }
    }
    statistics.nInputGates = nInputGates;

    if (!isDecoded) {
        measureAndCorrect();
        writeDecoding();
        isDecoded = true;
    }

    statistics.nOutputGates = qcMapped.getNindividualOps();

    return qcMapped;
}

void Ecc::swap(dd::Qubit target1, dd::Qubit target2) {
    qcMapped.swap(target1, target2);
}

void Ecc::writeToffoli(int target, int c1, bool p1, int c2, bool p2) {
    dd::Controls ctrls;
    ctrls.insert(dd::Control{dd::Qubit(c1), p1 ? dd::Control::Type::pos : dd::Control::Type::neg});
    ctrls.insert(dd::Control{dd::Qubit(c2), p2 ? dd::Control::Type::pos : dd::Control::Type::neg});
    writeX(static_cast<dd::Qubit>(target), ctrls);
}

void Ecc::writeGeneric(dd::Qubit target, qc::OpType type) {
    switch (type) {
        case qc::I:
            return;
        case qc::H:
            qcMapped.h(target);
            return;
        case qc::S:
            qcMapped.s(target);
            return;
        case qc::Sdag:
            writeSdag(target);
            return;
        case qc::X:
            writeX(target);
            return;
        case qc::Y:
            writeY(target);
            return;
        case qc::Z:
            writeZ(target);
            return;
        default:
            int nQubits = qcOriginal.getNqubits();
            qcMapped.emplace_back<qc::StandardOperation>(nQubits * ecc.nRedundantQubits, target + 2 * nQubits, type);
    }
}

[[maybe_unused]] [[maybe_unused]] void Ecc::writeGeneric(dd::Qubit target, const dd::Control& control, qc::OpType type) {
    switch (type) {
        case qc::I:
            return;
        case qc::X:
            writeX(target, control);
            return;
        case qc::Y:
            writeY(target, control);
            return;
        case qc::Z:
            writeZ(target, control);
            return;
        default:
            qcMapped.emplace_back<qc::StandardOperation>(qcOriginal.getNqubits() * ecc.nRedundantQubits, control, target, type);
            return;
    }
}

void Ecc::writeGeneric(dd::Qubit target, const dd::Controls& controls, qc::OpType type) {
    switch (type) {
        case qc::I:
            return;
        case qc::X:
            writeX(target, controls);
            return;
        case qc::Y:
            writeY(target, controls);
            return;
        case qc::Z:
            writeZ(target, controls);
            return;
        default:
            qcMapped.emplace_back<qc::StandardOperation>(qcOriginal.getNqubits() * ecc.nRedundantQubits, controls, target, type);
            return;
    }
}

void Ecc::writeX(dd::Qubit target) {
    qcMapped.x(target);
}

void Ecc::writeX(int target, const dd::Control& control) {
    writeX(static_cast<dd::Qubit>(target), control);
}

void Ecc::writeX(dd::Qubit target, const dd::Control& control) {
    qcMapped.x(target, control);
}

/*method has to have same signature as "writeZstatic" (as it is stored in the same function pointer in certain codes), thus bool parameter is kept */
void Ecc::writeXstatic(dd::Qubit target, dd::Control control, qc::QuantumComputation* qcMapped) {
    qcMapped->x(target, control);
}

void Ecc::writeX(dd::Qubit target, const dd::Controls& controls) {
    qcMapped.x(target, controls);
}

void Ecc::writeY(dd::Qubit target) {
    qcMapped.y(target);
}

void Ecc::writeY(dd::Qubit target, const dd::Control& control) {
    qcMapped.y(target, control);
}

void Ecc::writeY(dd::Qubit target, const dd::Controls& controls) {
    gatesWritten = true;
    qcMapped.y(target, controls);
}

void Ecc::writeZ(dd::Qubit target) {
    qcMapped.z(target);
}

void Ecc::writeZ(dd::Qubit target, const dd::Control& control) {
    writeZstatic(target, control, &qcMapped);
}

void Ecc::writeZstatic(dd::Qubit target, dd::Control control, qc::QuantumComputation* qcMapped) {
    qcMapped->z(target, control);
}

void Ecc::writeZ(dd::Qubit target, const dd::Controls& controls) {
    qcMapped.z(target, controls);
}

void Ecc::writeSdag(dd::Qubit target) {
    qcMapped.sdag(target);
}

void Ecc::writeClassicalControl(dd::Qubit control, int qubitCount, unsigned int value, qc::OpType opType, int target) {
    std::unique_ptr<qc::Operation> op    = std::make_unique<qc::StandardOperation>(qcMapped.getNqubits(), dd::Qubit(target), opType);
    const auto                     pair_ = std::make_pair(control, dd::QubitCount(qubitCount));
    qcMapped.emplace_back<qc::ClassicControlledOperation>(op, pair_, value);
}
