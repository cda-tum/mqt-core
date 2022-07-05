/*
 * This file is part of MQT QFR library which is released under the MIT license.
 * See file README.md or go to https://www.cda.cit.tum.de/research/quantum/ for more information.
 */

#ifndef QFR_QUANTUMCIRCUIT_HPP
#define QFR_QUANTUMCIRCUIT_HPP

#include "pybind11/pybind11.h"
#include "regex"

#include <regex>
#include <type_traits>
#include <variant>

namespace py = pybind11;
using namespace pybind11::literals;

#include "QuantumComputation.hpp"
#include "SymbolicQuantumComputation.hpp"

namespace qc::qiskit {
    class QuantumCircuit {
    public:
        static void import(QuantumComputation& qc, const py::object& circ) {
            qc.reset();

            py::object QuantumCircuit = py::module::import("qiskit").attr("QuantumCircuit");

            if (!py::isinstance(circ, QuantumCircuit)) {
                throw QFRException("[import] Python object needs to be a Qiskit QuantumCircuit");
            }

            if (!circ.attr("name").is_none()) {
                qc.setName(circ.attr("name").cast<std::string>());
            }

            // handle qubit registers
            py::object Qubit           = py::module::import("qiskit.circuit").attr("Qubit");
            py::object AncillaQubit    = py::module::import("qiskit.circuit").attr("AncillaQubit");
            py::object AncillaRegister = py::module::import("qiskit.circuit").attr("AncillaRegister");
            int        qubitIndex      = 0;
            py::dict   qubitMap{};
            auto&&     circQregs = circ.attr("qregs");
            for (const auto qreg: circQregs) {
                // create corresponding register in quantum computation
                auto size = qreg.attr("size").cast<dd::QubitCount>();
                auto name = qreg.attr("name").cast<std::string>();
                if (py::isinstance(qreg, AncillaRegister)) {
                    qc.addAncillaryRegister(size, name.c_str());
                    // add ancillas to qubit map
                    for (int i = 0; i < size; ++i) {
                        qubitMap[AncillaQubit(qreg, i)] = qubitIndex;
                        qubitIndex++;
                    }
                } else {
                    qc.addQubitRegister(size, name.c_str());
                    // add qubits to qubit map
                    for (int i = 0; i < size; ++i) {
                        qubitMap[Qubit(qreg, i)] = qubitIndex;
                        qubitIndex++;
                    }
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

            // import initial layout in case it is available
            if (!circ.attr("_layout").is_none()) {
                importInitialLayout(qc, circ);
            }
            qc.initializeIOMapping();
        }

        static void dumpTensorNetwork(const py::object& circ, const std::string& filename) {
            QuantumComputation qc{};
            import(qc, circ);
            std::ofstream ofs(filename);
            qc.dump(ofs, qc::Tensor);
        }

    protected:
        static void emplaceOperation(QuantumComputation& qc, const py::object& instruction, const py::list& qargs, const py::list& cargs, const py::list& params, const py::dict& qubitMap, const py::dict& clbitMap) {
            static const auto nativelySupportedGates = std::set<std::string>{"i", "id", "iden", "x", "y", "z", "h", "s", "sdg", "t", "tdg", "p", "u1", "rx", "ry", "rz", "u2", "u", "u3", "cx", "cy", "cz", "cp", "cu1", "ch", "crx", "cry", "crz", "cu3", "ccx", "swap", "cswap", "iswap", "sx", "sxdg", "csx", "mcx", "mcx_gray", "mcx_recursive", "mcx_vchain", "mcphase", "mcrx", "mcry", "mcrz"};

            auto instructionName = instruction.attr("name").cast<std::string>();
            if (instructionName == "measure") {
                auto control = qubitMap[qargs[0]].cast<dd::Qubit>();
                auto target  = clbitMap[cargs[0]].cast<std::size_t>();
                qc.emplace_back<NonUnitaryOperation>(qc.getNqubits(), control, target);
            } else if (instructionName == "barrier") {
                Targets targets{};
                for (const auto qubit: qargs) {
                    auto target = qubitMap[qubit].cast<dd::Qubit>();
                    targets.emplace_back(target);
                }
                qc.emplace_back<NonUnitaryOperation>(qc.getNqubits(), targets, Barrier);
            } else if (nativelySupportedGates.count(instructionName)) {
                // natively supported operations
                if (instructionName == "i" || instructionName == "id" || instructionName == "iden") {
                    addOperation(qc, I, qargs, params, qubitMap);
                } else if (instructionName == "x" || instructionName == "cx" || instructionName == "ccx" || instructionName == "mcx_gray" || instructionName == "mcx") {
                    addOperation(qc, X, qargs, params, qubitMap);
                } else if (instructionName == "y" || instructionName == "cy") {
                    addOperation(qc, Y, qargs, params, qubitMap);
                } else if (instructionName == "z" || instructionName == "cz") {
                    addOperation(qc, Z, qargs, params, qubitMap);
                } else if (instructionName == "h" || instructionName == "ch") {
                    addOperation(qc, H, qargs, params, qubitMap);
                } else if (instructionName == "s") {
                    addOperation(qc, S, qargs, params, qubitMap);
                } else if (instructionName == "sdg") {
                    addOperation(qc, Sdag, qargs, params, qubitMap);
                } else if (instructionName == "t") {
                    addOperation(qc, T, qargs, params, qubitMap);
                } else if (instructionName == "tdg") {
                    addOperation(qc, Tdag, qargs, params, qubitMap);
                } else if (instructionName == "rx" || instructionName == "crx" || instructionName == "mcrx") {
                    addOperation(qc, RX, qargs, params, qubitMap);
                } else if (instructionName == "ry" || instructionName == "cry" || instructionName == "mcry") {
                    addOperation(qc, RY, qargs, params, qubitMap);
                } else if (instructionName == "rz" || instructionName == "crz" || instructionName == "mcrz") {
                    addOperation(qc, RZ, qargs, params, qubitMap);
                } else if (instructionName == "p" || instructionName == "u1" || instructionName == "cp" || instructionName == "cu1" || instructionName == "mcphase") {
                    addOperation(qc, Phase, qargs, params, qubitMap);
                } else if (instructionName == "sx" || instructionName == "csx") {
                    addOperation(qc, SX, qargs, params, qubitMap);
                } else if (instructionName == "sxdg") {
                    addOperation(qc, SXdag, qargs, params, qubitMap);
                } else if (instructionName == "u2") {
                    addOperation(qc, U2, qargs, params, qubitMap);
                } else if (instructionName == "u" || instructionName == "u3" || instructionName == "cu3") {
                    addOperation(qc, U3, qargs, params, qubitMap);
                } else if (instructionName == "swap" || instructionName == "cswap") {
                    addTwoTargetOperation(qc, SWAP, qargs, params, qubitMap);
                } else if (instructionName == "iswap") {
                    addTwoTargetOperation(qc, iSWAP, qargs, params, qubitMap);
                } else if (instructionName == "mcx_recursive") {
                    if (qargs.size() <= 5) {
                        addOperation(qc, X, qargs, params, qubitMap);
                    } else {
                        auto qargs_copy = qargs.attr("copy")();
                        qargs_copy.attr("pop")(); // discard ancillaries
                        addOperation(qc, X, qargs_copy, params, qubitMap);
                    }
                } else if (instructionName == "mcx_vchain") {
                    auto        size       = qargs.size();
                    std::size_t ncontrols  = (size + 1) / 2;
                    auto        qargs_copy = qargs.attr("copy")();
                    // discard ancillaries
                    for (std::size_t i = 0; i < ncontrols - 2; ++i) {
                        qargs_copy.attr("pop")();
                    }
                    addOperation(qc, X, qargs_copy, params, qubitMap);
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

        static SymbolOrNumber parseSymbolicExpr(const py::object& pyExpr) {
            static const std::regex summands("[+|-]?[^+-]+");
            static const std::regex products("[\\*/]?[^\\*/]+");
            // static const std::regex products(".*");
            // static const products("")
            auto exprStr = pyExpr.attr("__str__")().cast<std::string>();
            exprStr.erase(std::remove(exprStr.begin(), exprStr.end(), ' '),
                          exprStr.end()); // strip whitespace

            auto sumIt  = std::sregex_iterator(exprStr.begin(), exprStr.end(), summands);
            auto sumEnd = std::sregex_iterator();

            qc::Symbolic sym;
            bool         isConst = true;

            while (sumIt != sumEnd) {
                auto match    = *sumIt;
                auto matchStr = match.str();
                int  sign     = matchStr[0] == '-' ? -1 : 1;
                if (matchStr[0] == '+' || matchStr[0] == '-') {
                    matchStr.erase(0, 1);
                }
                // std::cout << matchStr << std::endl;

                auto prodIt  = std::sregex_iterator(matchStr.begin(), matchStr.end(), products);
                auto prodEnd = std::sregex_iterator();

                dd::fp      coeff = 1.0;
                std::string var   = "";
                while (prodIt != prodEnd) {
                    auto prodMatch = *prodIt;
                    auto prodStr   = prodMatch.str();

                    bool isDiv = prodStr[0] == '/';
                    if (prodStr[0] == '*' || prodStr[0] == '/') {
                        prodStr.erase(0, 1);
                    }

                    std::istringstream iss(prodStr);
                    dd::fp             f;
                    iss >> f;
                    bool isNum = iss.eof() && !iss.fail();

                    if (isNum) {
                        coeff *= isDiv ? 1.0 / f : f;
                    } else {
                        var = prodStr;
                    }

                    prodIt++;
                }
                if (var.empty()) {
                    sym += coeff;
                } else {
                    isConst = false;
                    sym += sym::Term(sign * coeff, sym::Variable{var});
                }
                sumIt++;
            }

            return isConst ? SymbolOrNumber{sym.getConst()} : SymbolOrNumber{sym};
        }

        static SymbolOrNumber parseParam(const py::object& param) {
            try {
                return param.cast<dd::fp>();
            } catch (py::cast_error& e) {
                return parseSymbolicExpr(param);
            }
        }

        static void addOperation(QuantumComputation& qc, OpType type, const py::list& qargs, const py::list& params, const py::dict& qubitMap) {
            std::vector<dd::Control> qubits{};
            for (const auto qubit: qargs) {
                auto target = qubitMap[qubit].cast<dd::Qubit>();
                qubits.emplace_back(dd::Control{target});
            }
            auto target = qubits.back().qubit;
            qubits.pop_back();
            // dd::fp theta = 0., phi = 0., lambda = 0.;
            qc::SymbolOrNumber theta = 0., phi = 0., lambda = 0.;

            if (params.size() == 1) {
                lambda = parseParam(params[0]);
            } else if (params.size() == 2) {
                phi    = parseParam(params[0]);
                lambda = parseParam(params[1]);
            } else if (params.size() == 3) {
                theta  = parseParam(params[0]);
                phi    = parseParam(params[1]);
                lambda = parseParam(params[2]);
            }
            dd::Controls controls(qubits.cbegin(), qubits.cend());
            if (std::holds_alternative<dd::fp>(lambda) && std::holds_alternative<dd::fp>(phi) && std::holds_alternative<dd::fp>(theta)) {
                qc.emplace_back<StandardOperation>(qc.getNqubits(), controls, target, type, std::get<dd::fp>(lambda), std::get<dd::fp>(phi), std::get<dd::fp>(theta));
            } else {
                qc.emplace_back<SymbolicOperation>(qc.getNqubits(), controls, target, type, lambda, phi, theta);
                qc.addVariables(lambda, phi, theta);
            }
        }

        static void addTwoTargetOperation(QuantumComputation& qc, OpType type, const py::list& qargs, const py::list& params, const py::dict& qubitMap) {
            std::vector<dd::Control> qubits{};
            for (const auto qubit: qargs) {
                auto target = qubitMap[qubit].cast<dd::Qubit>();
                qubits.emplace_back(dd::Control{target});
            }
            auto target1 = qubits.back().qubit;
            qubits.pop_back();
            auto target0 = qubits.back().qubit;
            qubits.pop_back();
            // dd::fp             theta = 0., phi = 0., lambda = 0.;
            qc::SymbolOrNumber theta = 0., phi = 0., lambda = 0.;
            if (params.size() == 1) {
                lambda = parseParam(params[0]);
            } else if (params.size() == 2) {
                phi    = parseParam(params[0]);
                lambda = parseParam(params[1]);
            } else if (params.size() == 3) {
                theta  = parseParam(params[0]);
                phi    = parseParam(params[1]);
                lambda = parseParam(params[2]);
            }
            dd::Controls controls(qubits.cbegin(), qubits.cend());
            if (std::holds_alternative<dd::fp>(lambda) && std::holds_alternative<dd::fp>(phi) && std::holds_alternative<dd::fp>(theta)) {
                qc.emplace_back<StandardOperation>(qc.getNqubits(), controls, target0, target1, type, std::get<dd::fp>(lambda), std::get<dd::fp>(phi), std::get<dd::fp>(theta));
            } else {
                qc.emplace_back<SymbolicOperation>(qc.getNqubits(), controls, target0, target1, type, lambda, phi, theta);
                qc.addVariables(lambda, phi, theta);
            }
        }

        static void importDefinition(QuantumComputation& qc, const py::object& circ, const py::list& qargs, const py::list& cargs, const py::dict& qubitMap, const py::dict& clbitMap) {
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

        static void importInitialLayout(QuantumComputation& qc, const py::object& circ) {
            py::object Qubit = py::module::import("qiskit.circuit").attr("Qubit");

            // get layout
            auto&& layout = circ.attr("_layout");

            // create map between registers used in the layout and logical qubit indices
            // NOTE: this only works correctly if the registers were originally declared in alphabetical order!
            auto&&   registers         = layout.attr("get_registers")().cast<py::set>();
            int      logicalQubitIndex = 0;
            py::dict logicalQubitIndices{};

            // the ancilla register
            decltype(registers.get_type()) ancillaRegister = py::none();

            for (const auto qreg: registers) {
                const auto qregName = qreg.attr("name").cast<std::string>();
                // skip ancillary register since it is handled as the very last qubit register
                if (qregName == "ancilla") {
                    ancillaRegister = qreg;
                    continue;
                }

                const auto size = qreg.attr("size").cast<dd::QubitCount>();
                for (int i = 0; i < size; ++i) {
                    logicalQubitIndices[Qubit(qreg, i)] = logicalQubitIndex;
                    logicalQubitIndex++;
                }
            }

            // handle ancillary register, if there is one
            if (!ancillaRegister.is_none()) {
                const auto size = ancillaRegister.attr("size").cast<dd::QubitCount>();
                for (int i = 0; i < size; ++i) {
                    logicalQubitIndices[Qubit(ancillaRegister, i)] = logicalQubitIndex;
                    qc.setLogicalQubitAncillary(static_cast<dd::Qubit>(logicalQubitIndex));
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
} // namespace qc::qiskit
#endif //QFR_QUANTUMCIRCUIT_HPP
