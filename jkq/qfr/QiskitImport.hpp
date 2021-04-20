/*
 * This file is part of JKQ QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#ifndef QFR_QISKITIMPORT_HPP
#define QFR_QISKITIMPORT_HPP

#include "pybind11/pybind11.h"

namespace py = pybind11;
using namespace pybind11::literals;

#include "QuantumComputation.hpp"

namespace qc {
    void emplaceQiskitOperation(QuantumComputation& qc, const py::object& instruction, const py::list& qargs, const py::list& cargs, const py::list& params);

    void importQiskitDefinition(QuantumComputation& qc, const py::object& circ, const py::list& qargs, const py::list& cargs) {
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

            emplaceQiskitOperation(qc, instruction, mapped_qargs, mapped_cargs, inst_params);
        }
    }

    void addQiskitOperation(QuantumComputation& qc, qc::OpType type, const py::list& qargs, const py::list& params) {
        std::vector<dd::Control> qubits{};
        for (const auto qubit: qargs) {
            auto&&    qreg   = qubit.attr("register").attr("name").cast<std::string>();
            auto&&    qidx   = qubit.attr("index").cast<dd::Qubit>();
            dd::Qubit target = qc.getIndexFromQubitRegister({qreg, qidx});
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

    void addTwoTargetQiskitOperation(QuantumComputation& qc, qc::OpType type, const py::list& qargs, const py::list& params) {
        std::vector<dd::Control> qubits{};
        for (const auto qubit: qargs) {
            auto&&    qreg   = qubit.attr("register").attr("name").cast<std::string>();
            auto&&    qidx   = qubit.attr("index").cast<dd::Qubit>();
            dd::Qubit target = qc.getIndexFromQubitRegister({qreg, qidx});
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

    void emplaceQiskitOperation(QuantumComputation& qc, const py::object& instruction, const py::list& qargs, const py::list& cargs, const py::list& params) {
        static const auto nativelySupportedGates = std::set<std::string>{"i", "id", "iden", "x", "y", "z", "h", "s", "sdg", "t", "tdg", "p", "u1", "rx", "ry", "rz", "u2", "u", "u3", "cx", "cy", "cz", "cp", "cu1", "ch", "crx", "cry", "crz", "cu3", "ccx", "swap", "cswap", "iswap", "sx", "sxdg", "csx", "mcx_gray", "mcx_recursive", "mcx_vchain", "mcphase", "mcrx", "mcry", "mcrz"};

        auto instructionName = instruction.attr("name").cast<std::string>();
        if (instructionName == "measure") {
            auto&&      qubit   = qargs[0];
            auto&&      clbit   = cargs[0];
            auto&&      qreg    = qubit.attr("register").attr("name").cast<std::string>();
            auto&&      creg    = clbit.attr("register").attr("name").cast<std::string>();
            auto&&      qidx    = qubit.attr("index").cast<dd::Qubit>();
            auto&&      cidx    = clbit.attr("index").cast<std::size_t>();
            dd::Qubit   control = qc.getIndexFromQubitRegister({qreg, qidx});
            std::size_t target  = qc.getIndexFromClassicalRegister({creg, cidx});
            qc.emplace_back<qc::NonUnitaryOperation>(qc.getNqubits(), control, target);
        } else if (instructionName == "barrier") {
            Targets targets{};
            for (const auto qubit: qargs) {
                auto&&    qreg   = qubit.attr("register").attr("name").cast<std::string>();
                auto&&    qidx   = qubit.attr("index").cast<dd::Qubit>();
                dd::Qubit target = qc.getIndexFromQubitRegister({qreg, qidx});
                targets.emplace_back(target);
            }
            qc.emplace_back<qc::NonUnitaryOperation>(qc.getNqubits(), targets, qc::Barrier);
        } else if (nativelySupportedGates.count(instructionName)) {
            // natively supported operations
            if (instructionName == "i" || instructionName == "id" || instructionName == "iden") {
                addQiskitOperation(qc, qc::I, qargs, params);
            } else if (instructionName == "x" || instructionName == "cx" || instructionName == "ccx" || instructionName == "mcx_gray") {
                addQiskitOperation(qc, qc::X, qargs, params);
            } else if (instructionName == "y" || instructionName == "cy") {
                addQiskitOperation(qc, qc::Y, qargs, params);
            } else if (instructionName == "z" || instructionName == "cz") {
                addQiskitOperation(qc, qc::Z, qargs, params);
            } else if (instructionName == "h" || instructionName == "ch") {
                addQiskitOperation(qc, qc::H, qargs, params);
            } else if (instructionName == "s") {
                addQiskitOperation(qc, qc::S, qargs, params);
            } else if (instructionName == "sdg") {
                addQiskitOperation(qc, qc::Sdag, qargs, params);
            } else if (instructionName == "t") {
                addQiskitOperation(qc, qc::T, qargs, params);
            } else if (instructionName == "tdg") {
                addQiskitOperation(qc, qc::Tdag, qargs, params);
            } else if (instructionName == "rx" || instructionName == "crx" || instructionName == "mcrx") {
                addQiskitOperation(qc, qc::RX, qargs, params);
            } else if (instructionName == "ry" || instructionName == "cry" || instructionName == "mcry") {
                addQiskitOperation(qc, qc::RY, qargs, params);
            } else if (instructionName == "rz" || instructionName == "crz" || instructionName == "mcrz") {
                addQiskitOperation(qc, qc::RZ, qargs, params);
            } else if (instructionName == "p" || instructionName == "u1" || instructionName == "cp" || instructionName == "cu1" || instructionName == "mcphase") {
                addQiskitOperation(qc, qc::Phase, qargs, params);
            } else if (instructionName == "sx" || instructionName == "csx") {
                addQiskitOperation(qc, qc::SX, qargs, params);
            } else if (instructionName == "sxdg") {
                addQiskitOperation(qc, qc::SXdag, qargs, params);
            } else if (instructionName == "u2") {
                addQiskitOperation(qc, qc::U2, qargs, params);
            } else if (instructionName == "u" || instructionName == "u3" || instructionName == "cu3") {
                addQiskitOperation(qc, qc::U3, qargs, params);
            } else if (instructionName == "swap" || instructionName == "cswap") {
                addTwoTargetQiskitOperation(qc, qc::SWAP, qargs, params);
            } else if (instructionName == "iswap") {
                addTwoTargetQiskitOperation(qc, qc::iSWAP, qargs, params);
            } else if (instructionName == "mcx_recursive") {
                if (qargs.size() <= 5) {
                    addQiskitOperation(qc, qc::X, qargs, params);
                } else {
                    auto qargs_copy = qargs.attr("copy")();
                    qargs_copy.attr("pop")(); // discard ancillaries
                    addQiskitOperation(qc, qc::X, qargs_copy, params);
                }
            } else if (instructionName == "mcx_vchain") {
                auto        size       = qargs.size();
                std::size_t ncontrols  = (size + 1) / 2;
                auto        qargs_copy = qargs.attr("copy")();
                // discard ancillaries
                for (std::size_t i = 0; i < ncontrols - 2; ++i) {
                    qargs_copy.attr("pop")();
                }
                addQiskitOperation(qc, qc::X, qargs_copy, params);
            }
        } else {
            try {
                importQiskitDefinition(qc, instruction.attr("definition"), qargs, cargs);
            } catch (py::error_already_set& e) {
                std::cerr << "Failed to import instruction " << instructionName << " from Qiskit" << std::endl;
                std::cerr << e.what() << std::endl;
            }
        }
    }

    void import(QuantumComputation& qc, const py::object& circ) {
        qc.reset();

        py::object QuantumCircuit = py::module::import("qiskit").attr("QuantumCircuit");
        if (!py::isinstance(circ, QuantumCircuit)) {
            throw QFRException("[import] Python object needs to be a Qiskit QuantumCircuit");
        }

        auto&& circQregs = circ.attr("qregs");
        for (const auto qreg: circQregs) {
            qc.addQubitRegister(qreg.attr("size").cast<dd::QubitCount>(), qreg.attr("name").cast<std::string>().c_str());
        }

        auto&& circCregs = circ.attr("cregs");
        for (const auto creg: circCregs) {
            qc.addClassicalRegister(creg.attr("size").cast<std::size_t>(), creg.attr("name").cast<std::string>().c_str());
        }

        // import initial layout in case it is available
        if (!circ.attr("_layout").is_none()) {
            auto&&    virtual_bits        = circ.attr("_layout").attr("get_virtual_bits")().cast<py::dict>();
            dd::Qubit logical_qubit_index = 0;
            for (auto qubit: virtual_bits) {
                auto&& logical_qubit = qubit.first;
                auto&& register_name = logical_qubit.attr("register").attr("name").cast<std::string>();
                //				auto&& register_index = logical_qubit.attr("index").cast<unsigned short>();
                if (register_name != "ancilla") {
                    qc.initialLayout[qubit.second.cast<dd::Qubit>()] = logical_qubit_index;
                    ++logical_qubit_index;
                }
            }
        }

        auto&& data = circ.attr("data");
        for (const auto pyinst: data) {
            auto&& inst        = pyinst.cast<std::tuple<py::object, py::list, py::list>>();
            auto&& instruction = std::get<0>(inst);
            auto&& qargs       = std::get<1>(inst);
            auto&& cargs       = std::get<2>(inst);
            auto&& params      = instruction.attr("params");

            emplaceQiskitOperation(qc, instruction, qargs, cargs, params);
        }

        qc.initializeIOMapping();
    }
} // namespace qc
#endif //QFR_QISKITIMPORT_HPP
