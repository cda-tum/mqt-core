/*
 * This file is part of JKQ QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#include "eccs/Ecc.hpp"

void Ecc::initMappedCircuit() {
    qcOriginal.stripIdleQubits(true, false);
    qcMapped.addQubitRegister(getNOutputQubits(qcOriginal.getNqubits()));
    for (const auto& cRegs = qcOriginal.getCregs(); auto const& [regName, regBits]: cRegs) {
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

    std::size_t nInputGates = 0U;
    for (const auto& gate: qcOriginal) {
        nInputGates++;
        mapGate(*gate);
        if (measureFrequency > 0 && nInputGates % measureFrequency == 0) {
            measureAndCorrect();
        }
    }

    //mapGate(...) can change 'isDecoded', therefore check it again
    if (!isDecoded) {
        measureAndCorrect();
        writeDecoding();
        isDecoded = true;
    }

    return qcMapped;
}

void Ecc::writeToffoli(int target, int c1, bool p1, int c2, bool p2) {
    dd::Controls ctrls;
    ctrls.insert(dd::Control{dd::Qubit(c1), p1 ? dd::Control::Type::pos : dd::Control::Type::neg});
    ctrls.insert(dd::Control{dd::Qubit(c2), p2 ? dd::Control::Type::pos : dd::Control::Type::neg});
    qcMapped.x(static_cast<dd::Qubit>(target), ctrls);
}

/*method has to have same signature as the static "writeZ" (as it is stored in the same function pointer in certain codes)*/
void Ecc::writeXstatic(dd::Qubit target, dd::Control control, qc::QuantumComputation* qcMapped) {
    qcMapped->x(target, control);
}

void Ecc::writeZstatic(dd::Qubit target, dd::Control control, qc::QuantumComputation* qcMapped) {
    qcMapped->z(target, control);
}

void Ecc::writeClassicalControl(dd::Qubit control, int qubitCount, unsigned int value, qc::OpType opType, int target) {
    std::unique_ptr<qc::Operation> op    = std::make_unique<qc::StandardOperation>(qcMapped.getNqubits(), dd::Qubit(target), opType);
    const auto                     pair_ = std::make_pair(control, dd::QubitCount(qubitCount));
    qcMapped.emplace_back<qc::ClassicControlledOperation>(op, pair_, value);
}
