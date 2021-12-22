/*
 * This file is part of JKQ QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#include "eccs/Ecc.hpp"

//#include <utility>

Ecc::Ecc(struct Info eccInfo, qc::QuantumComputation &quantumcomputation, int measFreq, bool decomposeMC,
         bool cliffOnly) : ecc(std::move(eccInfo)), qc(quantumcomputation), measureFrequency(measFreq),
                           decomposeMultiControlledGates(decomposeMC), cliffordGatesOnly(cliffOnly) {
    decodingDone = false;
}

void Ecc::initMappedCircuit() {
    qc.stripIdleQubits(true, false);
    statistics.nInputQubits = qc.getNqubits();
    statistics.nInputClassicalBits = (int) qc.getNcbits();
    statistics.nOutputQubits = qc.getNqubits() * ecc.nRedundantQubits + ecc.nCorrectingBits;
    statistics.nOutputClassicalBits = statistics.nInputClassicalBits + ecc.nCorrectingBits;
    qcMapped.addQubitRegister(statistics.nOutputQubits);
    qcMapped.addClassicalRegister(statistics.nInputClassicalBits);
    qcMapped.addClassicalRegister(ecc.nCorrectingBits, "qecc");
}

qc::QuantumComputation &Ecc::apply() {
    initMappedCircuit();

    writeEncoding();

    long nInputGates = 0;
    for (const auto &gate: qc) {
        nInputGates++;
        mapGate(gate);
        if (measureFrequency > 0 && nInputGates % measureFrequency == 0) {
            measureAndCorrect();
        }
    }
    statistics.nInputGates = nInputGates;

    if (!decodingDone) {
        measureAndCorrect();
        writeDecoding();
        decodingDone = true;
    }


    statistics.nOutputGates = qcMapped.getNindividualOps();

    return qcMapped;
}

[[maybe_unused]] std::ostream &Ecc::printResult(std::ostream &out) {
    out << "\tused error correcting code: " << ecc.name << std::endl;
    out << "\tgate overhead: " << statistics.getGateOverhead() << std::endl;
    out << "\tinput qubits: " << statistics.nInputQubits << std::endl;
    out << "\tinput gates: " << statistics.nInputGates << std::endl;
    out << "\toutput qubits: " << statistics.nOutputQubits << std::endl;
    out << "\toutput gates: " << statistics.nOutputGates << std::endl;
    return out;
}

void Ecc::dumpResult(const std::string &outputFilename) {
    if (qcMapped.empty()) {
        std::cerr << "Mapped circuit is empty." << std::endl;
        return;
    }

    size_t dot = outputFilename.find_last_of('.');
    std::string extension = outputFilename.substr(dot + 1);
    std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char c) { return ::tolower(c); });
    if (extension == "qasm") {
        dumpResult(outputFilename, qc::OpenQASM);
    } else {
        throw qc::QFRException("[dump] Extension " + extension + " not recognized/supported for dumping.");
    }
}

void Ecc::gateNotAvailableError(const std::unique_ptr<qc::Operation> &gate) {
    throw qc::QFRException(
            std::string("Gate ") + gate->getName() + " not supported to encode in error code " + ecc.name + "!");
}

void Ecc::writeToffoli(int target, int c1, bool p1, int c2, bool p2) {
    if (decomposeMultiControlledGates && cliffordGatesOnly) {
        throw qc::QFRException(std::string("Gate t not possible to encode with clifford gates only!"));
    }
    if (decomposeMultiControlledGates) {

        if (!p1) { writeX(static_cast<dd::Qubit>(c1)); }
        if (!p2) { writeX(static_cast<dd::Qubit>(c2)); }

        qcMapped.h(static_cast<dd::Qubit>(target));
        writeX(static_cast<dd::Qubit>(target), dd::Control{dd::Qubit(c2)});
        qcMapped.tdag(static_cast<dd::Qubit>(target));
        writeX(static_cast<dd::Qubit>(target), dd::Control{dd::Qubit(c1)});
        qcMapped.t(static_cast<dd::Qubit>(target));
        writeX(static_cast<dd::Qubit>(target), dd::Control{dd::Qubit(c2)});
        qcMapped.tdag(static_cast<dd::Qubit>(target));
        writeX(static_cast<dd::Qubit>(target), dd::Control{dd::Qubit(c1)});
        qcMapped.t(static_cast<dd::Qubit>(target));
        qcMapped.t(static_cast<dd::Qubit>(c2));
        qcMapped.h(static_cast<dd::Qubit>(target));
        writeX(static_cast<dd::Qubit>(c2), dd::Control{dd::Qubit(c1)});
        qcMapped.t(static_cast<dd::Qubit>(c1));
        qcMapped.tdag(static_cast<dd::Qubit>(c2));
        writeX(static_cast<dd::Qubit>(c2), dd::Control{dd::Qubit(c1)});

        if (!p1) { writeX(static_cast<dd::Qubit>(c1)); }
        if (!p2) { writeX(static_cast<dd::Qubit>(c2)); }
    } else {
        dd::Controls ctrls;
        ctrls.insert(dd::Control{dd::Qubit(c1), p1 ? dd::Control::Type::pos : dd::Control::Type::neg});
        ctrls.insert(dd::Control{dd::Qubit(c2), p2 ? dd::Control::Type::pos : dd::Control::Type::neg});
        writeX(static_cast<dd::Qubit>(target), ctrls);
    }
}

[[maybe_unused]] void Ecc::writeZToffoli(int target, int c1, bool p1, int c2, bool p2) {
    qcMapped.h(static_cast<dd::Qubit>(target));
    writeToffoli(static_cast<dd::Qubit>(target), c1, p1, c2, p2);
    qcMapped.h(static_cast<dd::Qubit>(target));
}

void Ecc::writeGeneric(dd::Qubit target, qc::OpType type) {
    switch (type) {
        case qc::I:
            return;
        case qc::H:
            qcMapped.h(target);
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
            throw qc::QFRException(std::string("Gate not possible to encode!"));
    }
}

[[maybe_unused]] void Ecc::writeGeneric(dd::Qubit target, const dd::Control &control, qc::OpType type) {
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
            throw qc::QFRException(std::string("Gate not possible to encode!"));
    }
}

void Ecc::writeGeneric(dd::Qubit target, const dd::Controls &controls, qc::OpType type) {
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
            throw qc::QFRException(std::string("Gate not possible to encode!"));
    }
}

void Ecc::writeX(dd::Qubit target) {
    if (cliffordGatesOnly) {
        qcMapped.h(target);
        writeZ(target);
        qcMapped.h(target);
    } else { qcMapped.x(target); }
}

void Ecc::writeX(dd::Qubit target, const dd::Control &control) {
    qcMapped.x(target, control);
}

void Ecc::writeX(dd::Qubit target, const dd::Controls &controls) {
    if (controls.empty()) {
        writeX(target);
    } else if (controls.empty()) {
        auto el(controls.begin());
        qcMapped.x(target, *el);
    } else if (decomposeMultiControlledGates || cliffordGatesOnly) {
        throw qc::QFRException(std::string("Multi-controlled X gate not possible to encode!"));
    } else {
        qcMapped.x(target, controls);
    }

    if (cliffordGatesOnly) {
        qcMapped.h(target);
        writeZ(target, controls);
        qcMapped.h(target);
    } else { qcMapped.x(target, controls); }
}

void Ecc::writeY(dd::Qubit target) {
    if (cliffordGatesOnly) {
        writeZ(target);
        writeX(target);
    } else { qcMapped.y(target); }
}

void Ecc::writeY(dd::Qubit target, const dd::Control &control) {
    if (cliffordGatesOnly) {
        writeZ(target, control);
        writeX(target, control);
    } else { qcMapped.y(target, control); }
}

void Ecc::writeY(dd::Qubit target, const dd::Controls &controls) {
    if (cliffordGatesOnly) {
        writeZ(target, controls);
        writeX(target, controls);
    } else { qcMapped.y(target, controls); }
}

void Ecc::writeZ(dd::Qubit target) {
    if (cliffordGatesOnly) {
        qcMapped.s(target);
        qcMapped.s(target);
    } else { qcMapped.z(target); }
}

void Ecc::writeZ(dd::Qubit target, const dd::Control &control) {
    if (cliffordGatesOnly) {
        qcMapped.h(target);
        qcMapped.x(target, control);
        qcMapped.h(target);
    } else { qcMapped.z(target, control); }
}

void Ecc::writeZ(dd::Qubit target, const dd::Controls &controls) {
    if (cliffordGatesOnly) {
        qcMapped.s(target, controls);
        qcMapped.s(target, controls);
    } else { qcMapped.z(target, controls); }
}

void Ecc::writeSdag(dd::Qubit target) {
    if (cliffordGatesOnly) {
        qcMapped.s(target);
        qcMapped.s(target);
        qcMapped.s(target);
    } else { qcMapped.sdag(target); }
}


