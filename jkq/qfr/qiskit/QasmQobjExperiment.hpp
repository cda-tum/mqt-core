/*
 * This file is part of JKQ QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#ifndef QFR_QASMQOBJEXPERIMENT_HPP
#define QFR_QASMQOBJEXPERIMENT_HPP

namespace qiskit {
    class QasmQobjExperiment {
    public:
        static void import(qc::QuantumComputation& qc, const py::object& circ) {
            qc.reset();

            py::object pyQasmQobjExperiment = py::module::import("qiskit.qobj").attr("QasmQobjExperiment");

            if (!py::isinstance(circ, pyQasmQobjExperiment)) {
                throw qc::QFRException("[import] Python object needs to be a Qiskit QasmQobjExperiment");
            }

            auto&& header = circ.attr("header");
            //            auto&& config       = circ.attr("config");
            auto&& instructions = circ.attr("instructions");

            auto&& circQregs = header.attr("qreg_sizes");
            for (const auto& qreg: circQregs) {
                qc.addQubitRegister(qreg.cast<py::list>()[1].cast<dd::QubitCount>(), qreg.cast<py::list>()[0].cast<std::string>().c_str());
            }

            auto&& circCregs = header.attr("creg_sizes");
            for (const auto& creg: circCregs) {
                qc.addClassicalRegister(creg.cast<py::list>()[1].cast<std::size_t>(), creg.cast<py::list>()[0].cast<std::string>().c_str());
            }

            for (const auto& instruction: instructions) {
                emplaceInstruction(qc, instruction.cast<py::object>());
            }
            qc.initializeIOMapping();
        }

    protected:
        static void emplaceInstruction(qc::QuantumComputation& qc, const py::object& instruction) {
            static const auto nativelySupportedGates = std::set<std::string>{"i", "id", "iden", "x", "y", "z", "h", "s", "sdg", "t", "tdg", "p", "u1", "rx", "ry", "rz", "u2", "u", "u3", "cx", "cy", "cz", "cp", "cu1", "ch", "crx", "cry", "crz", "cu3", "ccx", "swap", "cswap", "iswap", "sx", "sxdg", "csx", "mcx_gray", "mcx_recursive", "mcx_vchain", "mcphase", "mcrx", "mcry", "mcrz"};

            auto instructionName = instruction.attr("name").cast<std::string>();
            if (instructionName == "measure") {
                auto qubit = instruction.attr("qubits").cast<py::list>()[0].cast<dd::Qubit>();
                auto clbit = instruction.attr("memory").cast<py::list>()[0].cast<std::size_t>();
                qc.emplace_back<qc::NonUnitaryOperation>(qc.getNqubits(), qubit, clbit);
            } else if (instructionName == "barrier") {
                qc::Targets targets{};
                for (const auto qubit: instruction.attr("qubits")) {
                    auto target = qubit.cast<dd::Qubit>();
                    targets.emplace_back(target);
                }
                qc.emplace_back<qc::NonUnitaryOperation>(qc.getNqubits(), targets, qc::Barrier);
            } else if (nativelySupportedGates.count(instructionName)) {
                auto&&   qubits = instruction.attr("qubits").cast<py::list>();
                py::list params{};
                // natively supported operations
                if (instructionName == "i" || instructionName == "id" || instructionName == "iden") {
                    addOperation(qc, qc::I, qubits, params);
                } else if (instructionName == "x" || instructionName == "cx" || instructionName == "ccx" || instructionName == "mcx_gray") {
                    addOperation(qc, qc::X, qubits, params);
                } else if (instructionName == "y" || instructionName == "cy") {
                    addOperation(qc, qc::Y, qubits, params);
                } else if (instructionName == "z" || instructionName == "cz") {
                    addOperation(qc, qc::Z, qubits, params);
                } else if (instructionName == "h" || instructionName == "ch") {
                    addOperation(qc, qc::H, qubits, params);
                } else if (instructionName == "s") {
                    addOperation(qc, qc::S, qubits, params);
                } else if (instructionName == "sdg") {
                    addOperation(qc, qc::Sdag, qubits, params);
                } else if (instructionName == "t") {
                    addOperation(qc, qc::T, qubits, params);
                } else if (instructionName == "tdg") {
                    addOperation(qc, qc::Tdag, qubits, params);
                } else if (instructionName == "rx" || instructionName == "crx" || instructionName == "mcrx") {
                    params = instruction.attr("params").cast<py::list>();
                    addOperation(qc, qc::RX, qubits, params);
                } else if (instructionName == "ry" || instructionName == "cry" || instructionName == "mcry") {
                    params = instruction.attr("params").cast<py::list>();
                    addOperation(qc, qc::RY, qubits, params);
                } else if (instructionName == "rz" || instructionName == "crz" || instructionName == "mcrz") {
                    params = instruction.attr("params").cast<py::list>();
                    addOperation(qc, qc::RZ, qubits, params);
                } else if (instructionName == "p" || instructionName == "u1" || instructionName == "cp" || instructionName == "cu1" || instructionName == "mcphase") {
                    params = instruction.attr("params").cast<py::list>();
                    addOperation(qc, qc::Phase, qubits, params);
                } else if (instructionName == "sx" || instructionName == "csx") {
                    addOperation(qc, qc::SX, qubits, params);
                } else if (instructionName == "sxdg") {
                    addOperation(qc, qc::SXdag, qubits, params);
                } else if (instructionName == "u2") {
                    params = instruction.attr("params").cast<py::list>();
                    addOperation(qc, qc::U2, qubits, params);
                } else if (instructionName == "u" || instructionName == "u3" || instructionName == "cu3") {
                    params = instruction.attr("params").cast<py::list>();
                    addOperation(qc, qc::U3, qubits, params);
                } else if (instructionName == "swap" || instructionName == "cswap") {
                    addTwoTargetOperation(qc, qc::SWAP, qubits, params);
                } else if (instructionName == "iswap") {
                    addTwoTargetOperation(qc, qc::iSWAP, qubits, params);
                } else if (instructionName == "mcx_recursive") {
                    if (qubits.size() <= 5) {
                        addOperation(qc, qc::X, qubits, params);
                    } else {
                        auto qubitsCopy = qubits.attr("copy")();
                        qubitsCopy.attr("pop")(); // discard ancillaries
                        addOperation(qc, qc::X, qubitsCopy, params);
                    }
                } else if (instructionName == "mcx_vchain") {
                    auto        size       = qubits.size();
                    std::size_t ncontrols  = (size + 1) / 2;
                    auto        qubitsCopy = qubits.attr("copy")();
                    // discard ancillaries
                    for (std::size_t i = 0; i < ncontrols - 2; ++i) {
                        qubitsCopy.attr("pop")();
                    }
                    addOperation(qc, qc::X, qubitsCopy, params);
                }
            } else {
                std::cerr << "Failed to import instruction " << instructionName << " from Qiskit QasmQobjExperiment" << std::endl;
            }
        }

        static void addOperation(qc::QuantumComputation& qc, qc::OpType type, const py::list& qubits, const py::list& params) {
            std::vector<dd::Control> qargs{};
            for (const auto& qubit: qubits) {
                auto target = qubit.cast<dd::Qubit>();
                qargs.emplace_back(dd::Control{target});
            }
            auto target = qargs.back().qubit;
            qargs.pop_back();

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
            dd::Controls controls(qargs.cbegin(), qargs.cend());
            qc.emplace_back<qc::StandardOperation>(qc.getNqubits(), controls, target, type, lambda, phi, theta);
        }

        static void addTwoTargetOperation(qc::QuantumComputation& qc, qc::OpType type, const py::list& qubits, const py::list& params) {
            std::vector<dd::Control> qargs{};
            for (const auto& qubit: qubits) {
                auto target = qubit.cast<dd::Qubit>();
                qargs.emplace_back(dd::Control{target});
            }
            auto target1 = qargs.back().qubit;
            qargs.pop_back();
            auto target0 = qargs.back().qubit;
            qargs.pop_back();
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
            dd::Controls controls(qargs.cbegin(), qargs.cend());
            qc.emplace_back<qc::StandardOperation>(qc.getNqubits(), controls, target0, target1, type, lambda, phi, theta);
        }
    };
} // namespace qiskit
#endif //QFR_QASMQOBJEXPERIMENT_HPP
