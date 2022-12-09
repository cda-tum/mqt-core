/*
 * This file is part of JKQ QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#include "eccs/Ecc.hpp"

#include <dd/Simulation.hpp>

void Ecc::initMappedCircuit() {
    qcOriginal->stripIdleQubits(true, false);
    qcMapped->addQubitRegister(getNOutputQubits(qcOriginal->getNqubits()));
    for (const auto& cRegs = qcOriginal->getCregs(); auto const& [regName, regBits]: cRegs) {
        qcMapped->addClassicalRegister(regBits.second, regName.c_str());
    }

    if (ecc.nCorrectingBits > 0) {
        qcMapped->addClassicalRegister(ecc.nCorrectingBits, "qecc");
    }
}

std::shared_ptr<qc::QuantumComputation> Ecc::apply() {
    initMappedCircuit();

    writeEncoding();
    isDecoded = false;

    std::size_t nInputGates = 0U;
    for (const auto& gate: *qcOriginal) {
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
    qcMapped->x(static_cast<dd::Qubit>(target), ctrls);
}

/*method has to have same signature as the static "writeZ" (as it is stored in the same function pointer in certain codes)*/
void Ecc::writeXstatic(dd::Qubit target, dd::Control control, const std::shared_ptr<qc::QuantumComputation>& qcMapped) {
    qcMapped->x(target, control);
}

void Ecc::writeZstatic(dd::Qubit target, dd::Control control, const std::shared_ptr<qc::QuantumComputation>& qcMapped) {
    qcMapped->z(target, control);
}

void Ecc::writeClassicalControl(dd::Qubit control, int qubitCount, unsigned int value, qc::OpType opType, int target) {
    std::unique_ptr<qc::Operation> op    = std::make_unique<qc::StandardOperation>(qcMapped->getNqubits(), dd::Qubit(target), opType);
    const auto                     pair_ = std::make_pair(control, dd::QubitCount(qubitCount));
    qcMapped->emplace_back<qc::ClassicControlledOperation>(op, pair_, value);
}

bool Ecc::verifyExecution(bool simulateWithErrors, const std::vector<dd::Qubit>& dataQubits, int insertErrorAfterNGates) const {
    auto   shots             = 50;
    double tolerance         = 0.2;
    size_t seed              = 1;
    auto   toleranceAbsolute = (shots / 100.0) * (tolerance * 100.0);

    auto ddOriginal       = std::make_unique<dd::Package<>>(qcOriginal->getNqubits());
    auto originalRootEdge = ddOriginal->makeZeroState(qcOriginal->getNqubits());
    ddOriginal->incRef(originalRootEdge);

    auto ddEcc       = std::make_unique<dd::Package<>>(qcMapped->getNqubits());
    auto eccRootEdge = ddEcc->makeZeroState(qcMapped->getNqubits());
    ddEcc->incRef(eccRootEdge);

    auto measurementsOriginal = simulate(qcOriginal.get(), originalRootEdge, ddOriginal, shots, seed);

    if (!simulateWithErrors) {
        auto measurementsProtected = simulate(qcMapped.get(), eccRootEdge, ddEcc, shots, seed);
        for (auto const& [cBitsOriginal, cHitsOriginal]: measurementsOriginal) {
            // Count the cHitsOriginal in the register with error correction
            size_t cHitsProtected = 0;
            for (auto const& [cBitsProtected, cHitsProtectedTemp]: measurementsProtected) {
                if (0 == cBitsProtected.compare(cBitsProtected.length() - cBitsOriginal.length(), cBitsOriginal.length(), cBitsOriginal)) cHitsProtected += cHitsProtectedTemp;
            }
            auto difference = std::max(cHitsProtected, cHitsOriginal) - std::min(cHitsProtected, cHitsOriginal);
            if (static_cast<double>(difference) > toleranceAbsolute) {
                return false;
            }
        }
    } else {
        for (auto const& qubit: dataQubits) {
            auto measurementsProtected = simulate(qcMapped.get(), eccRootEdge, ddEcc, shots, seed, true, qubit, insertErrorAfterNGates);
            for (auto const& [classicalBit, hits]: measurementsOriginal) {
                // Since the result is stored as one bit string. I have to count the relevant classical bits.
                size_t eccHits = 0;
                for (auto const& [eccMeasure, tempHits]: measurementsProtected) {
                    if (0 == eccMeasure.compare(eccMeasure.length() - classicalBit.length(), classicalBit.length(), classicalBit)) eccHits += tempHits;
                }
                auto difference = std::max(eccHits, hits) - std::min(eccHits, hits);
                if (static_cast<double>(difference) > toleranceAbsolute) {
                    std::cout << "Simulation failed when applying error to qubit " << static_cast<unsigned>(qubit) << " after " << insertErrorAfterNGates << " gates.\n";
                    std::cout << "Error in bit " << classicalBit << " original register: " << hits << " ecc register: " << eccHits << std::endl;
                    return false;
                } else {
                    std::cout << "Diff/tolerance " << difference << "/" << toleranceAbsolute << " Original register: " << hits << " ecc register: " << eccHits;
                    std::cout << " Error at qubit " << static_cast<unsigned>(qubit) << " after " << insertErrorAfterNGates << " gates." << std::endl;
                }
            }
        }
    }
    return true;
}
