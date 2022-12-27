/*
* This file is part of MQT QFR library which is released under the MIT license.
* See file README.md or go to https://www.cda.cit.tum.de/research/quantum/ for more information.
*/

#pragma once

#include "QuantumComputation.hpp"

namespace ecc {
    using Qubit      = dd::Qubit;
    using QubitCount = std::make_unsigned_t<Qubit>;

    class Ecc {
    public:
        enum class ID {
            Id,
            Q3Shor,
            Q9Shor,
            Q5Laflamme,
            Q7Steane,
            Q9Surface,
            Q18Surface
        };
        struct Info {
            ID                                               id;
            std::size_t                                      nRedundantQubits; //usually number of physical qubits per (encoded) logical qubit
            std::size_t                                      nCorrectingBits;  //usually number of classical bits needed for correcting one qubit
            std::string                                      name;
            std::vector<std::pair<std::size_t, const char*>> classicalRegisters;
        };

        Ecc(Info ecc, std::shared_ptr<qc::QuantumComputation> qc, std::size_t measureFrequency):
            qcOriginal(std::move(qc)), measureFrequency(measureFrequency), ecc(std::move(ecc)) {
            qcMapped = std::make_shared<qc::QuantumComputation>();
        }
        virtual ~Ecc() = default;

        std::shared_ptr<qc::QuantumComputation> apply();

        virtual std::string getName() {
            return ecc.name;
        }

        virtual std::size_t getNOutputQubits(std::size_t nInputQubits) {
            return nInputQubits * ecc.nRedundantQubits + ecc.nCorrectingBits;
        }

        [[nodiscard]] bool verifyExecution(bool simulateWithErrors, const std::vector<Qubit>& dataQubits, size_t insertErrorAfterNGates) const;

    protected:
        std::shared_ptr<qc::QuantumComputation> qcOriginal;
        std::shared_ptr<qc::QuantumComputation> qcMapped;
        std::size_t                             measureFrequency;
        bool                                    isDecoded    = true;
        bool                                    gatesWritten = false;
        Info                                    ecc;

        // Set parameters for verifying the eccs
        const size_t            shots     = 50;
        constexpr static double tolerance = 0.2;
        const size_t            seed      = 1;

        void initMappedCircuit();

        /**
     * prepares an encoded logical |0> state in the qcMapped circuit.
     * May, but does not have to be overridden by subclasses.
     * */
        virtual void writeEncoding() {
            if (!isDecoded) {
                return;
            }
            isDecoded = false;
            measureAndCorrect();
        }

        /**
     * in case of an error, calling this function creates a 'clean' state again. Usual structure:
     *
     * for each logical qubit i:
     * -- reset ancilla qubits
     * -- measure physical data qubits of logical qubit[i] onto ancilla qubits
     * -- correct data qubits based on measurement results of ancilla qubits
     * */
        virtual void measureAndCorrect() = 0;

        /**
     * moves encoded state information back to original qubits (i.e. 1 qubit per logical qubit)
     * */
        virtual void writeDecoding() = 0;

        virtual void mapGate(const qc::Operation& gate) = 0;

        void gateNotAvailableError(const qc::Operation& gate) const {
            throw qc::QFRException(std::string("Gate ") + gate.getName() + " not supported to encode in error code " + ecc.name + "!");
        }

        void ccx(Qubit target, Qubit c1, bool p1, Qubit c2, bool p2) {
            dd::Controls controls;
            controls.insert(dd::Control{(c1), p1 ? dd::Control::Type::pos : dd::Control::Type::neg});
            controls.insert(dd::Control{(c2), p2 ? dd::Control::Type::pos : dd::Control::Type::neg});
            qcMapped->x(static_cast<Qubit>(target), controls);
        }

        void classicalControl(const std::pair<Qubit, std::size_t>& controlRegister, std::size_t value, qc::OpType opType, Qubit target) {
            std::unique_ptr<qc::Operation> op = std::make_unique<qc::StandardOperation>(qcMapped->getNqubits(), target, opType);
            qcMapped->emplace_back<qc::ClassicControlledOperation>(op, controlRegister, value);
        }

        //static, since some codes need to store those functions into function pointers
        using staticWriteFunctionType = void (*)(Qubit, dd::Control, const std::shared_ptr<qc::QuantumComputation>&);
        static void x(Qubit target, dd::Control control, const std::shared_ptr<qc::QuantumComputation>& qcMapped) {
            qcMapped->x(target, control);
        }
        static void z(Qubit target, dd::Control control, const std::shared_ptr<qc::QuantumComputation>& qcMapped) {
            qcMapped->z(target, control);
        }

        /**
     * returns if op1 and op2 are commutative,
     * i.e. if for all qubit states s: op1(op2(s)) == op2(op1(s))
     * */
        static bool commutative(qc::OpType op1, qc::OpType op2) {
            return op1 == op2 || op1 == qc::I || op2 == qc::I;
        }
    };
} // namespace ecc
