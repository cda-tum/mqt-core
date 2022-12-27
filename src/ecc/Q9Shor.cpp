/*
 * This file is part of MQT QFR library which is released under the MIT license.
 * See file README.md or go to https://www.cda.cit.tum.de/research/quantum/ for more information.
 */

#include "ecc/Q9Shor.hpp"
namespace ecc {
    void Q9Shor::writeEncoding() {
        if (!isDecoded) {
            return;
        }
        isDecoded          = false;
        const auto nQubits = qcOriginal->getNqubits();
        for (Qubit i = 0; i < nQubits; i++) {
            std::array<dd::Control, 3> controls = {};
            for (std::size_t j = 0; j < controls.size(); j++) {
                controls[j] = {static_cast<Qubit>(i + 3 * j * nQubits), dd::Control::Type::pos};
                if (j > 0) {
                    qcMapped->x(static_cast<Qubit>(i + 3 * j * nQubits), controls[0]);
                }
            }
            for (std::size_t j = 0; j < controls.size(); j++) {
                qcMapped->h(static_cast<Qubit>(i + 3 * j * nQubits));
                qcMapped->x(static_cast<Qubit>(i + (3 * j + 1) * nQubits), controls[j]);
                qcMapped->x(static_cast<Qubit>(i + (3 * j + 2) * nQubits), controls[j]);
            }
        }
        gatesWritten = true;
    }

    void Q9Shor::measureAndCorrect() {
        if (isDecoded) {
            return;
        }
        const auto nQubits = qcOriginal->getNqubits();
        const auto clStart = qcOriginal->getNcbits();
        for (Qubit i = 0; i < nQubits; i++) {
            //syntactic sugar for qubit indices
            std::array<Qubit, 9>       qubits                  = {};
            std::array<Qubit, 8>       ancillaQubits           = {};
            std::array<dd::Control, 8> ancillaControls         = {};
            std::array<dd::Control, 8> negativeAncillaControls = {};
            for (std::size_t j = 0; j < 9; j++) {
                qubits.at(j) = static_cast<Qubit>(i + j * nQubits);
            }
            for (std::size_t j = 0; j < 8; j++) {
                ancillaQubits.at(j) = static_cast<Qubit>(ecc.nRedundantQubits * nQubits + j);
                qcMapped->reset(ancillaQubits.at(j));
            }
            for (std::size_t j = 0; j < 8; j++) {
                ancillaControls.at(j) = dd::Control{static_cast<Qubit>(ancillaQubits.at(j)), dd::Control::Type::pos};
            }
            for (std::size_t j = 0; j < 8; j++) {
                negativeAncillaControls.at(j) = dd::Control{static_cast<Qubit>(ancillaQubits.at(j)), dd::Control::Type::neg};
            }

            // PREPARE measurements --------------------------------------------------------
            for (Qubit const j: ancillaQubits) {
                qcMapped->h(j);
            }
            //x errors = indirectly via controlled z
            for (std::size_t j = 0; j < 3; j++) {
                qcMapped->z(qubits[3 * j], ancillaControls[2 * j]);
                qcMapped->z(qubits[3 * j + 1], ancillaControls[2 * j]);
                qcMapped->z(qubits[3 * j + 1], ancillaControls[2 * j + 1]);
                qcMapped->z(qubits[3 * j + 2], ancillaControls[2 * j + 1]);
            }

            //z errors = indirectly via controlled x/CNOT
            for (std::size_t j = 0; j < 6; j++) {
                qcMapped->x(qubits[j], ancillaControls[6]);
                qcMapped->x(qubits[3 + j], ancillaControls[7]);
            }

            for (Qubit const j: ancillaQubits) {
                qcMapped->h(j);
            }

            //MEASURE ancilla qubits
            for (std::size_t j = 0; j < 8; j++) {
                qcMapped->measure(ancillaQubits.at(j), clStart + j);
            }

            //CORRECT
            //x, i.e. bit flip errors
            for (std::size_t j = 0; j < 3; j++) {
                const auto controlRegister = std::make_pair(static_cast<Qubit>(clStart + 2 * j), 2);
                classicalControl(controlRegister, 1, qc::X, static_cast<Qubit>(i + 3 * j * nQubits));
                classicalControl(controlRegister, 2, qc::X, static_cast<Qubit>(i + (3 * j + 2) * nQubits));
                classicalControl(controlRegister, 3, qc::X, static_cast<Qubit>(i + (3 * j + 1) * nQubits));
            }

            //z, i.e. phase flip errors
            const auto controlRegister = std::make_pair(static_cast<Qubit>(clStart + 6), 2);
            classicalControl(controlRegister, 1, qc::Z, i);
            classicalControl(controlRegister, 2, qc::Z, static_cast<Qubit>(i + 6 * nQubits));
            classicalControl(controlRegister, 3, qc::Z, static_cast<Qubit>(i + 3 * nQubits));
        }
    }

    void Q9Shor::writeDecoding() {
        if (isDecoded) {
            return;
        }
        const auto nQubits = qcOriginal->getNqubits();
        for (Qubit i = 0; i < nQubits; i++) {
            std::array<dd::Control, 9> ci;
            for (Qubit j = 0; j < 9; j++) {
                ci.at(j) = dd::Control{static_cast<Qubit>(i + j * nQubits), dd::Control::Type::pos};
            }

            for (std::size_t j = 0; j < 3; j++) {
                qcMapped->x(static_cast<Qubit>(i + (3 * j + 1) * nQubits), ci[3 * j]);
                qcMapped->x(static_cast<Qubit>(i + (3 * j + 2) * nQubits), ci[3 * j]);
                ccx(static_cast<Qubit>(i + 3 * j * nQubits), static_cast<Qubit>(i + (3 * j + 1) * nQubits), true, static_cast<Qubit>(i + (3 * j + 2) * nQubits), true);
                qcMapped->h(static_cast<Qubit>(i + 3 * j * nQubits));
            }

            qcMapped->x(static_cast<Qubit>(i + 3 * nQubits), ci[0]);
            qcMapped->x(static_cast<Qubit>(i + 6 * nQubits), ci[0]);
            ccx(i, static_cast<Qubit>(i + 3 * nQubits), true, static_cast<Qubit>(i + 6 * nQubits), true);
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
                        ctrls2.insert(dd::Control{static_cast<Qubit>(ct.qubit + j * nQubits), ct.type});
                        qcMapped->h(static_cast<Qubit>(ct.qubit + j * nQubits));
                    }
                    qcMapped->emplace_back<qc::StandardOperation>(qcMapped->getNqubits(), ctrls2, i + j * nQubits, type);
                    for (const auto& ct: ctrls) {
                        qcMapped->h(static_cast<Qubit>(ct.qubit + j * nQubits));
                    }
                }
            } else {
                for (size_t j = 0; j < 9; j++) {
                    qcMapped->emplace_back<qc::StandardOperation>(qcMapped->getNqubits(), i + j * nQubits, type);
                }
            }
        }
    }
} // namespace ecc
