/*
 * This file is part of JKQ QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#ifndef QFR_QUANTUMCIRCUIT_HPP
#define QFR_QUANTUMCIRCUIT_HPP

#include "pybind11/pybind11.h"

namespace py = pybind11;
using namespace pybind11::literals;

#include "QuantumComputation.hpp"

namespace qiskit {
    class QuantumCircuit {
    public:
        static void import(qc::QuantumComputation& qc, const py::object& circ) {
            qc.reset();

            py::object QuantumCircuit = py::module::import("qiskit").attr("QuantumCircuit");

            if (!py::isinstance(circ, QuantumCircuit)) {
                throw qc::QFRException("[import] Python object needs to be a Qiskit QuantumCircuit");
            }

            // import initial layout in case it is available
            if (!circ.attr("_layout").is_none()) {
                importInitialLayout(qc, circ);
            }

            // handle qubit registers
            py::object Qubit      = py::module::import("qiskit.circuit").attr("Qubit");
            int        qubitIndex = 0;
            py::dict   qubitMap{};
            auto&&     circQregs = circ.attr("qregs");
            for (const auto qreg: circQregs) {
                // create corresponding register in quantum computation
                auto size = qreg.attr("size").cast<dd::QubitCount>();
                auto name = qreg.attr("name").cast<std::string>();
                qc.addQubitRegister(size, name.c_str());

                // add qubits to qubit map
                for (int i = 0; i < size; ++i) {
                    qubitMap[Qubit(qreg, i)] = qubitIndex;
                    qubitIndex++;
                }
            }

            // handle classical registers
            py::object Clbit      = py::module::import("qiskit.circuit").attr("Clbit");
            int        clbitIndex = 0;
            py::dict   clbitMap{};
            auto&&     circCregs = circ.attr("cregs");
            for (const auto creg: circCregs) {
                // create corresponding register in quantum computation
                auto size = creg.attr("size").cast<std::size_t>();
                auto name = creg.attr("name").cast<std::string>();
                qc.addClassicalRegister(size, name.c_str());

                // add clbits to clbit map
                for (std::size_t i = 0; i < size; ++i) {
                    clbitMap[Clbit(creg, i)] = clbitIndex;
                    clbitIndex++;
                }
            }

            // iterate over instructions
            auto&& data = circ.attr("data");
            for (const auto pyinst: data) {
                auto&& inst        = pyinst.cast<std::tuple<py::object, py::list, py::list>>();
                auto&& instruction = std::get<0>(inst);
                auto&& qargs       = std::get<1>(inst);
                auto&& cargs       = std::get<2>(inst);
                auto&& params      = instruction.attr("params");

                emplaceOperation(qc, instruction, qargs, cargs, params, qubitMap, clbitMap);
            }
            qc.initializeIOMapping();
        }

    protected:
        static void emplaceOperation(qc::QuantumComputation& qc, const py::object& instruction, const py::list& qargs, const py::list& cargs, const py::list& params, const py::dict& qubitMap, const py::dict& clbitMap) {
            static const auto nativelySupportedGates = std::set<std::string>{"i", "id", "iden", "x", "y", "z", "h", "s", "sdg", "t", "tdg", "p", "u1", "rx", "ry", "rz", "u2", "u", "u3", "cx", "cy", "cz", "cp", "cu1", "ch", "crx", "cry", "crz", "cu3", "ccx", "swap", "cswap", "iswap", "sx", "sxdg", "csx", "mcx_gray", "mcx_recursive", "mcx_vchain", "mcphase", "mcrx", "mcry", "mcrz"};

            auto instructionName = instruction.attr("name").cast<std::string>();
            if (instructionName == "measure") {
                auto control = qubitMap[qargs[0]].cast<dd::Qubit>();
                auto target  = clbitMap[cargs[0]].cast<std::size_t>();
                qc.emplace_back<qc::NonUnitaryOperation>(qc.getNqubits(), control, target);
            } else if (instructionName == "barrier") {
                qc::Targets targets{};
                for (const auto qubit: qargs) {
                    auto target = qubitMap[qubit].cast<dd::Qubit>();
                    targets.emplace_back(target);
                }
                qc.emplace_back<qc::NonUnitaryOperation>(qc.getNqubits(), targets, qc::Barrier);
            } else if (nativelySupportedGates.count(instructionName)) {
                // natively supported operations
                if (instructionName == "i" || instructionName == "id" || instructionName == "iden") {
                    addOperation(qc, qc::I, qargs, params, qubitMap);
                } else if (instructionName == "x" || instructionName == "cx" || instructionName == "ccx" || instructionName == "mcx_gray") {
                    addOperation(qc, qc::X, qargs, params, qubitMap);
                } else if (instructionName == "y" || instructionName == "cy") {
                    addOperation(qc, qc::Y, qargs, params, qubitMap);
                } else if (instructionName == "z" || instructionName == "cz") {
                    addOperation(qc, qc::Z, qargs, params, qubitMap);
                } else if (instructionName == "h" || instructionName == "ch") {
                    addOperation(qc, qc::H, qargs, params, qubitMap);
                } else if (instructionName == "s") {
                    addOperation(qc, qc::S, qargs, params, qubitMap);
                } else if (instructionName == "sdg") {
                    addOperation(qc, qc::Sdag, qargs, params, qubitMap);
                } else if (instructionName == "t") {
                    addOperation(qc, qc::T, qargs, params, qubitMap);
                } else if (instructionName == "tdg") {
                    addOperation(qc, qc::Tdag, qargs, params, qubitMap);
                } else if (instructionName == "rx" || instructionName == "crx" || instructionName == "mcrx") {
                    addOperation(qc, qc::RX, qargs, params, qubitMap);
                } else if (instructionName == "ry" || instructionName == "cry" || instructionName == "mcry") {
                    addOperation(qc, qc::RY, qargs, params, qubitMap);
                } else if (instructionName == "rz" || instructionName == "crz" || instructionName == "mcrz") {
                    addOperation(qc, qc::RZ, qargs, params, qubitMap);
                } else if (instructionName == "p" || instructionName == "u1" || instructionName == "cp" || instructionName == "cu1" || instructionName == "mcphase") {
                    addOperation(qc, qc::Phase, qargs, params, qubitMap);
                } else if (instructionName == "sx" || instructionName == "csx") {
                    addOperation(qc, qc::SX, qargs, params, qubitMap);
                } else if (instructionName == "sxdg") {
                    addOperation(qc, qc::SXdag, qargs, params, qubitMap);
                } else if (instructionName == "u2") {
                    addOperation(qc, qc::U2, qargs, params, qubitMap);
                } else if (instructionName == "u" || instructionName == "u3" || instructionName == "cu3") {
                    addOperation(qc, qc::U3, qargs, params, qubitMap);
                } else if (instructionName == "swap" || instructionName == "cswap") {
                    addTwoTargetOperation(qc, qc::SWAP, qargs, params, qubitMap);
                } else if (instructionName == "iswap") {
                    addTwoTargetOperation(qc, qc::iSWAP, qargs, params, qubitMap);
                } else if (instructionName == "mcx_recursive") {
                    if (qargs.size() <= 5) {
                        addOperation(qc, qc::X, qargs, params, qubitMap);
                    } else {
                        auto qargs_copy = qargs.attr("copy")();
                        qargs_copy.attr("pop")(); // discard ancillaries
                        addOperation(qc, qc::X, qargs_copy, params, qubitMap);
                    }
                } else if (instructionName == "mcx_vchain") {
                    auto        size       = qargs.size();
                    std::size_t ncontrols  = (size + 1) / 2;
                    auto        qargs_copy = qargs.attr("copy")();
                    // discard ancillaries
                    for (std::size_t i = 0; i < ncontrols - 2; ++i) {
                        qargs_copy.attr("pop")();
                    }
                    addOperation(qc, qc::X, qargs_copy, params, qubitMap);
                }
            } else {
                try {
                    importDefinition(qc, instruction.attr("definition"), qargs, cargs, qubitMap, clbitMap);
                } catch (py::error_already_set& e) {
                    std::cerr << "Failed to import instruction " << instructionName << " from Qiskit QuantumCircuit" << std::endl;
                    std::cerr << e.what() << std::endl;
                }
            }
        }

        static void addOperation(qc::QuantumComputation& qc, qc::OpType type, const py::list& qargs, const py::list& params, const py::dict& qubitMap) {
            std::vector<dd::Control> qubits{};
            for (const auto qubit: qargs) {
                auto target = qubitMap[qubit].cast<dd::Qubit>();
                qubits.emplace_back(dd::Control{target});
            }
            auto target = qubits.back().qubit;
            qubits.pop_back();
            dd::fp theta = 0., phi = 0., lambda = 0.;
            if (params.size() == 1) {
                lambda = params[0].cast<dd::fp>();
            } else if (params.size() == 2) {
                phi    = params[0].cast<dd::fp>();
                lambda = params[1].cast<dd::fp>();
            } else if (params.size() == 3) {
                theta  = params[0].cast<dd::fp>();
                phi    = params[1].cast<dd::fp>();
                lambda = params[2].cast<dd::fp>();
            }
            dd::Controls controls(qubits.cbegin(), qubits.cend());
            qc.emplace_back<qc::StandardOperation>(qc.getNqubits(), controls, target, type, lambda, phi, theta);
        }

        static void addTwoTargetOperation(qc::QuantumComputation& qc, qc::OpType type, const py::list& qargs, const py::list& params, const py::dict& qubitMap) {
            std::vector<dd::Control> qubits{};
            for (const auto qubit: qargs) {
                auto target = qubitMap[qubit].cast<dd::Qubit>();
                qubits.emplace_back(dd::Control{target});
            }
            auto target1 = qubits.back().qubit;
            qubits.pop_back();
            auto target0 = qubits.back().qubit;
            qubits.pop_back();
            dd::fp theta = 0., phi = 0., lambda = 0.;
            if (params.size() == 1) {
                lambda = params[0].cast<dd::fp>();
            } else if (params.size() == 2) {
                phi    = params[0].cast<dd::fp>();
                lambda = params[1].cast<dd::fp>();
            } else if (params.size() == 3) {
                theta  = params[0].cast<dd::fp>();
                phi    = params[1].cast<dd::fp>();
                lambda = params[2].cast<dd::fp>();
            }
            dd::Controls controls(qubits.cbegin(), qubits.cend());
            qc.emplace_back<qc::StandardOperation>(qc.getNqubits(), controls, target0, target1, type, lambda, phi, theta);
        }

        static void importDefinition(qc::QuantumComputation& qc, const py::object& circ, const py::list& qargs, const py::list& cargs, const py::dict& qubitMap, const py::dict& clbitMap) {
            py::dict   qargMap{};
            py::list&& def_qubits = circ.attr("qubits");
            for (size_t i = 0; i < qargs.size(); ++i) {
                qargMap[def_qubits[i]] = qargs[i];
            }

            py::dict   cargMap{};
            py::list&& def_clbits = circ.attr("clbits");
            for (size_t i = 0; i < cargs.size(); ++i) {
                cargMap[def_clbits[i]] = cargs[i];
            }

            auto&& data = circ.attr("data");
            for (const auto pyinst: data) {
                auto&& inst        = pyinst.cast<std::tuple<py::object, py::list, py::list>>();
                auto&& instruction = std::get<0>(inst);

                py::list& inst_qargs = std::get<1>(inst);
                py::list  mapped_qargs{};
                for (auto&& inst_qarg: inst_qargs) {
                    mapped_qargs.append(qargMap[inst_qarg]);
                }

                py::list inst_cargs = std::get<2>(inst);
                py::list mapped_cargs{};
                for (auto&& inst_carg: inst_cargs) {
                    mapped_cargs.append(cargMap[inst_carg]);
                }

                auto&& inst_params = instruction.attr("params");

                emplaceOperation(qc, instruction, mapped_qargs, mapped_cargs, inst_params, qubitMap, clbitMap);
            }
        }

        static void importInitialLayout(qc::QuantumComputation& qc, const py::object& circ) {
            py::object Qubit = py::module::import("qiskit.circuit").attr("Qubit");

            // get layout
            auto&& layout = circ.attr("_layout");

            // create map between registers used in the layout and logical qubit indices
            // NOTE: this only works correctly if the registers were originally declared in alphabetical order!
            auto&&   registers         = layout.attr("get_registers")().cast<py::set>();
            int      logicalQubitIndex = 0;
            py::dict logicalQubitIndices{};
            for (const auto& qreg: registers) {
                auto qregName = qreg.attr("name").cast<std::string>();
                // skip ancillary register
                if (qregName == "ancilla")
                    continue;

                auto size = qreg.attr("size").cast<dd::QubitCount>();
                for (int i = 0; i < size; ++i) {
                    logicalQubitIndices[Qubit(qreg, i)] = logicalQubitIndex;
                    logicalQubitIndex++;
                }
            }

            // get a map of physical to logical qubits
            auto&& physicalQubits = layout.attr("get_physical_bits")().cast<py::dict>();

            // create initial layout
            for (auto qubit: physicalQubits) {
                auto&& physicalQubit = qubit.first.cast<dd::Qubit>();
                auto&& logicalQubit  = qubit.second;
                if (logicalQubitIndices.contains(logicalQubit)) {
                    qc.initialLayout[physicalQubit] = logicalQubitIndices[logicalQubit].cast<dd::Qubit>();
                }
            }
        }
    };
} // namespace qiskit
#endif //QFR_QUANTUMCIRCUIT_HPP
