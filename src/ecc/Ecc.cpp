/*
 * This file is part of MQT QFR library which is released under the MIT license.
 * See file README.md or go to https://www.cda.cit.tum.de/research/quantum/ for more information.
 */

#include "ecc/Ecc.hpp"

#include <dd/Simulation.hpp>

void Ecc::initMappedCircuit() {
    qcOriginal->stripIdleQubits(true, false);
    qcMapped->addQubitRegister(getNOutputQubits(qcOriginal->getNqubits()));
    auto cRegs = qcOriginal->getCregs();
    for (auto const& [regName, regBits]: cRegs) {
        qcMapped->addClassicalRegister(regBits.second, regName.c_str());
    }
    for (auto cr: ecc.classicalRegisters) {
        qcMapped->addClassicalRegister(cr.first, cr.second);
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

bool Ecc::verifyExecution(bool simulateWithErrors) const {
    auto toleranceAbsolute = (shots / 100.0) * (tolerance * 100.0);

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
                if (0 == cBitsProtected.compare(cBitsProtected.length() - cBitsOriginal.length(), cBitsOriginal.length(), cBitsOriginal)) {
                    cHitsProtected += cHitsProtectedTemp;
                }
            }
            auto difference = std::max(cHitsProtected, cHitsOriginal) - std::min(cHitsProtected, cHitsOriginal);
            if (static_cast<double>(difference) > toleranceAbsolute) {
                return false;
            }
        }
    } else {
        for (dd::Qubit qubit = 0; qubit < static_cast<dd::Qubit>(this->qcMapped->getNqubits()); qubit++) {
            auto measurementsProtected = simulate(qcMapped.get(), eccRootEdge, ddEcc, shots, seed, true, qubit, this->insertErrorAfterNGates);
            for (auto const& [classicalBit, hits]: measurementsOriginal) {
                // Since the result is stored as one bit string. I have to count the relevant classical bits.
                size_t eccHits = 0;
                for (auto const& [eccMeasure, tempHits]: measurementsProtected) {
                    if (0 == eccMeasure.compare(eccMeasure.length() - classicalBit.length(), classicalBit.length(), classicalBit)) {
                        eccHits += tempHits;
                    }
                }
                auto difference = std::max(eccHits, hits) - std::min(eccHits, hits);
                std::cout << "Diff/tolerance " << difference << "/" << toleranceAbsolute << " Original register: " << hits << " ecc register: " << eccHits;
                std::cout << " Simulating an error in qubit " << static_cast<unsigned>(qubit) << " after " << this->insertErrorAfterNGates << " gates." << std::endl;
                if (static_cast<double>(difference) > toleranceAbsolute) {
                    std::cout << "Error is too large!" << std::endl;
                    return false;
                }
            }
        }
    }
    return true;
}

bool Ecc::commutative(qc::OpType op1, qc::OpType op2) {
    return op1 == op2 || op1 == qc::I || op2 == qc::I;
}
