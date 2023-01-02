/*
 * This file is part of MQT QFR library which is released under the MIT license.
 * See file README.md or go to https://www.cda.cit.tum.de/research/quantum/ for more information.
 */

#ifndef QFR_QASMQOBJEXPERIMENT_HPP
#define QFR_QASMQOBJEXPERIMENT_HPP

#include "pybind11/pybind11.h"

namespace py = pybind11;

#include "QuantumComputation.hpp"

namespace qc::qiskit {
    using namespace pybind11::literals;

    class QasmQobjExperiment {
    public:
        static void import(QuantumComputation& qc, const py::object& circ) {
            qc.reset();

            const py::object pyQasmQobjExperiment = py::module::import("qiskit.qobj").attr("QasmQobjExperiment");

            if (!py::isinstance(circ, pyQasmQobjExperiment)) {
                throw QFRException("[import] Python object needs to be a Qiskit QasmQobjExperiment");
            }

            auto&& header       = circ.attr("header");
            auto&& instructions = circ.attr("instructions");

            auto&& circQregs = header.attr("qreg_sizes");
            for (const auto qreg: circQregs) {
                qc.addQubitRegister(qreg.cast<py::list>()[1].cast<std::size_t>(), qreg.cast<py::list>()[0].cast<std::string>());
            }

            auto&& circCregs = header.attr("creg_sizes");
            for (const auto creg: circCregs) {
                qc.addClassicalRegister(creg.cast<py::list>()[1].cast<std::size_t>(), creg.cast<py::list>()[0].cast<std::string>());
            }

            for (const auto instruction: instructions) {
                emplaceInstruction(qc, instruction.cast<py::object>());
            }
            qc.initializeIOMapping();
        }

    protected:
        static void emplaceInstruction(QuantumComputation& qc, const py::object& instruction) {
            static const auto NATIVELY_SUPPORTED_GATES = std::set<std::string>{"i", "id", "iden", "x", "y", "z", "h", "s", "sdg", "t", "tdg", "p", "u1", "rx", "ry", "rz", "u2", "u", "u3", "cx", "cy", "cz", "cp", "cu1", "ch", "crx", "cry", "crz", "cu3", "ccx", "swap", "cswap", "iswap", "sx", "sxdg", "csx", "mcx", "mcx_gray", "mcx_recursive", "mcx_vchain", "mcphase", "mcrx", "mcry", "mcrz"};

            auto instructionName = instruction.attr("name").cast<std::string>();
            if (instructionName == "measure") {
                auto qubit = instruction.attr("qubits").cast<py::list>()[0].cast<Qubit>();
                auto clbit = instruction.attr("memory").cast<py::list>()[0].cast<std::size_t>();
                qc.emplace_back<NonUnitaryOperation>(qc.getNqubits(), qubit, clbit);
            } else if (instructionName == "barrier") {
                Targets targets{};
                for (const auto qubit: instruction.attr("qubits")) {
                    auto target = qubit.cast<Qubit>();
                    targets.emplace_back(target);
                }
                qc.emplace_back<NonUnitaryOperation>(qc.getNqubits(), targets, Barrier);
            } else if (instructionName == "reset") {
                Targets targets{};
                for (const auto qubit: instruction.attr("qubits")) {
                    auto target = qubit.cast<Qubit>();
                    targets.emplace_back(target);
                }
                qc.reset(targets);
            } else if (NATIVELY_SUPPORTED_GATES.count(instructionName) != 0) {
                auto&&   qubits = instruction.attr("qubits").cast<py::list>();
                py::list params{};
                // natively supported operations
                if (instructionName == "i" || instructionName == "id" || instructionName == "iden") {
                    addOperation(qc, I, qubits, params);
                } else if (instructionName == "x" || instructionName == "cx" || instructionName == "ccx" || instructionName == "mcx_gray" || instructionName == "mcx") {
                    addOperation(qc, X, qubits, params);
                } else if (instructionName == "y" || instructionName == "cy") {
                    addOperation(qc, Y, qubits, params);
                } else if (instructionName == "z" || instructionName == "cz") {
                    addOperation(qc, Z, qubits, params);
                } else if (instructionName == "h" || instructionName == "ch") {
                    addOperation(qc, H, qubits, params);
                } else if (instructionName == "s") {
                    addOperation(qc, S, qubits, params);
                } else if (instructionName == "sdg") {
                    addOperation(qc, Sdag, qubits, params);
                } else if (instructionName == "t") {
                    addOperation(qc, T, qubits, params);
                } else if (instructionName == "tdg") {
                    addOperation(qc, Tdag, qubits, params);
                } else if (instructionName == "rx" || instructionName == "crx" || instructionName == "mcrx") {
                    params = instruction.attr("params").cast<py::list>();
                    addOperation(qc, RX, qubits, params);
                } else if (instructionName == "ry" || instructionName == "cry" || instructionName == "mcry") {
                    params = instruction.attr("params").cast<py::list>();
                    addOperation(qc, RY, qubits, params);
                } else if (instructionName == "rz" || instructionName == "crz" || instructionName == "mcrz") {
                    params = instruction.attr("params").cast<py::list>();
                    addOperation(qc, RZ, qubits, params);
                } else if (instructionName == "p" || instructionName == "u1" || instructionName == "cp" || instructionName == "cu1" || instructionName == "mcphase") {
                    params = instruction.attr("params").cast<py::list>();
                    addOperation(qc, Phase, qubits, params);
                } else if (instructionName == "sx" || instructionName == "csx") {
                    addOperation(qc, SX, qubits, params);
                } else if (instructionName == "sxdg") {
                    addOperation(qc, SXdag, qubits, params);
                } else if (instructionName == "u2") {
                    params = instruction.attr("params").cast<py::list>();
                    addOperation(qc, U2, qubits, params);
                } else if (instructionName == "u" || instructionName == "u3" || instructionName == "cu3") {
                    params = instruction.attr("params").cast<py::list>();
                    addOperation(qc, U3, qubits, params);
                } else if (instructionName == "swap" || instructionName == "cswap") {
                    addTwoTargetOperation(qc, SWAP, qubits, params);
                } else if (instructionName == "iswap") {
                    addTwoTargetOperation(qc, iSWAP, qubits, params);
                } else if (instructionName == "mcx_recursive") {
                    if (qubits.size() <= 5) {
                        addOperation(qc, X, qubits, params);
                    } else {
                        auto qubitsCopy = qubits.attr("copy")();
                        qubitsCopy.attr("pop")(); // discard ancillaries
                        addOperation(qc, X, qubitsCopy, params);
                    }
                } else if (instructionName == "mcx_vchain") {
                    auto              size       = qubits.size();
                    const std::size_t ncontrols  = (size + 1) / 2;
                    auto              qubitsCopy = qubits.attr("copy")();
                    // discard ancillaries
                    for (std::size_t i = 0; i < ncontrols - 2; ++i) {
                        qubitsCopy.attr("pop")();
                    }
                    addOperation(qc, X, qubitsCopy, params);
                }
            } else {
                std::cerr << "Failed to import instruction " << instructionName << " from Qiskit QasmQobjExperiment" << std::endl;
            }
        }

        static void addOperation(QuantumComputation& qc, OpType type, const py::list& qubits, const py::list& params) {
            std::vector<Control> qargs{};
            for (const auto qubit: qubits) {
                auto target = qubit.cast<Qubit>();
                qargs.emplace_back(Control{target});
            }
            auto target = qargs.back().qubit;
            qargs.pop_back();

            fp theta  = 0.;
            fp phi    = 0.;
            fp lambda = 0.;
            if (params.size() == 1) {
                lambda = params[0].cast<fp>();
            } else if (params.size() == 2) {
                phi    = params[0].cast<fp>();
                lambda = params[1].cast<fp>();
            } else if (params.size() == 3) {
                theta  = params[0].cast<fp>();
                phi    = params[1].cast<fp>();
                lambda = params[2].cast<fp>();
            }
            const Controls controls(qargs.cbegin(), qargs.cend());
            qc.emplace_back<StandardOperation>(qc.getNqubits(), controls, target, type, lambda, phi, theta);
        }

        static void addTwoTargetOperation(QuantumComputation& qc, OpType type, const py::list& qubits, const py::list& params) {
            std::vector<Control> qargs{};
            for (const auto qubit: qubits) {
                auto target = qubit.cast<Qubit>();
                qargs.emplace_back(Control{target});
            }
            auto target1 = qargs.back().qubit;
            qargs.pop_back();
            auto target0 = qargs.back().qubit;
            qargs.pop_back();
            fp theta  = 0.;
            fp phi    = 0.;
            fp lambda = 0.;
            if (params.size() == 1) {
                lambda = params[0].cast<fp>();
            } else if (params.size() == 2) {
                phi    = params[0].cast<fp>();
                lambda = params[1].cast<fp>();
            } else if (params.size() == 3) {
                theta  = params[0].cast<fp>();
                phi    = params[1].cast<fp>();
                lambda = params[2].cast<fp>();
            }
            const Controls controls(qargs.cbegin(), qargs.cend());
            qc.emplace_back<StandardOperation>(qc.getNqubits(), controls, target0, target1, type, lambda, phi, theta);
        }
    };
} // namespace qc::qiskit
#endif //QFR_QASMQOBJEXPERIMENT_HPP
