#include "python/qiskit/nanobind/QuantumCircuit.hpp"

#include "Definitions.hpp"

#include <cstddef>
#include <nanobind/stl/bind_map.h>
#include <nanobind/stl/set.h>
#include <set>
#include <string>

void qc::qiskit::QuantumCircuit::import(qc::QuantumComputation& qc,
                                        const nb::object& circ) {
  qc.reset();

  const nb::object quantumCircuit =
      nb::module_::import_("qiskit").attr("QuantumCircuit");
  // if (!nb::isinstance(circ, quantumCircuit))
  if (circ.type().is(quantumCircuit)) {
    throw QFRException(
        "[import] Python object needs to be a Qiskit QuantumCircuit");
  }

  if (!circ.attr("name").is_none()) {
    qc.setName(nb::cast<std::string>(circ.attr("name")));
  }

  // handle qubit registers
  const nb::object qubit = nb::module_::import_("qiskit.circuit").attr("Qubit");
  const nb::object ancillaQubit =
      nb::module_::import_("qiskit.circuit").attr("AncillaQubit");
  const nb::object ancillaRegister =
      nb::module_::import_("qiskit.circuit").attr("AncillaRegister");
  int qubitIndex = 0;
  const nb::dict qubitMap{};
  auto&& circQregs = circ.attr("qregs");
  for (const auto qreg : circQregs) {
    // create corresponding register in quantum computation
    auto size = nb::cast<std::size_t>(qreg.attr("size"));
    auto name = nb::cast<std::string>(qreg.attr("name"));
    // if (nb::isinstance(qreg, ancillaRegister))
    if (qreg.type().is(ancillaRegister)) {
      qc.addAncillaryRegister(size, name);
      // add ancillas to qubit map
      for (std::size_t i = 0; i < size; ++i) {
        qubitMap[ancillaQubit(qreg, i)] = qubitIndex;
        qubitIndex++;
      }
    } else {
      qc.addQubitRegister(size, name);
      // add qubits to qubit map
      for (std::size_t i = 0; i < size; ++i) {
        qubitMap[qubit(qreg, i)] = qubitIndex;
        qubitIndex++;
      }
    }
  }

  // handle classical registers
  const nb::object clbit = nb::module_::import_("qiskit.circuit").attr("Clbit");
  int clbitIndex = 0;
  const nb::dict clbitMap{};
  auto&& circCregs = circ.attr("cregs");
  for (const auto creg : circCregs) {
    // create corresponding register in quantum computation
    auto size = nb::cast<std::size_t>(creg.attr("size"));
    auto name = nb::cast<std::string>(creg.attr("name"));
    qc.addClassicalRegister(size, name);

    // add clbits to clbit map
    for (std::size_t i = 0; i < size; ++i) {
      clbitMap[clbit(creg, i)] = clbitIndex;
      clbitIndex++;
    }
  }

  try {
    qc.gphase(nb::cast<fp>(circ.attr("global_phase")));
  } catch (const nb::cast_error& e) {
    std::clog << e.what() << "\n";
    std::clog << "[import_] Warning: Symbolic global phase values are not "
                 "supported yet. Ignoring global phase.\n";
  }

  // iterate over instructions
  auto&& data = circ.attr("data");
  for (const auto pyinst : data) {
    auto&& inst = nb::cast<std::tuple<nb::object, nb::list, nb::list>>(pyinst);
    auto&& instruction = std::get<0>(inst);
    auto&& qargs = std::get<1>(inst);
    auto&& cargs = std::get<2>(inst);
    auto&& params = instruction.attr("params");

    emplaceOperation(qc, instruction, qargs, cargs, params, qubitMap, clbitMap);
  }

  // import_ initial layout in case it is available
  if (!circ.attr("_layout").is_none()) {
    importInitialLayout(qc, circ);
  }
  qc.initializeIOMapping();
}

void qc::qiskit::QuantumCircuit::emplaceOperation(
    qc::QuantumComputation& qc, const nb::object& instruction,
    const nb::list& qargs, const nb::list& cargs, const nb::list& params,
    const nb::dict& qubitMap, const nb::dict& clbitMap) {
  static const auto NATIVELY_SUPPORTED_GATES =
      std::set<std::string>{"i",          "id",       "iden",
                            "x",          "y",        "z",
                            "h",          "s",        "sdg",
                            "t",          "tdg",      "p",
                            "u1",         "rx",       "ry",
                            "rz",         "u2",       "u",
                            "u3",         "cx",       "cy",
                            "cz",         "cp",       "cu1",
                            "ch",         "crx",      "cry",
                            "crz",        "cu3",      "ccx",
                            "swap",       "cswap",    "iswap",
                            "sx",         "sxdg",     "csx",
                            "mcx",        "mcx_gray", "mcx_recursive",
                            "mcx_vchain", "mcphase",  "mcrx",
                            "mcry",       "mcrz",     "dcx",
                            "ecr",        "rxx",      "ryy",
                            "rzx",        "rzz",      "xx_minus_yy",
                            "xx_plus_yy"};

  auto instructionName = nb::cast<std::string>(instruction.attr("name"));
  if (instructionName == "measure") {
    auto control = nb::cast<Qubit>(qubitMap[qargs[0]]);
    auto target = nb::cast<std::size_t>(clbitMap[cargs[0]]);
    qc.emplace_back<NonUnitaryOperation>(qc.getNqubits(), control, target);
  } else if (instructionName == "barrier") {
    Targets targets{};
    for (const auto qubit : qargs) {
      auto target = nb::cast<Qubit>(qubitMap[qubit]);
      targets.emplace_back(target);
    }
    qc.emplace_back<NonUnitaryOperation>(qc.getNqubits(), targets, Barrier);
  } else if (instructionName == "reset") {
    Targets targets{};
    for (const auto qubit : qargs) {
      auto target = nb::cast<Qubit>(qubitMap[qubit]);
      targets.emplace_back(target);
    }
    qc.reset(targets);
  } else if (NATIVELY_SUPPORTED_GATES.count(instructionName) != 0) {
    // natively supported operations
    if (instructionName == "i" || instructionName == "id" ||
        instructionName == "iden") {
      addOperation(qc, I, qargs, params, qubitMap);
    } else if (instructionName == "x" || instructionName == "cx" ||
               instructionName == "ccx" || instructionName == "mcx_gray" ||
               instructionName == "mcx") {
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
    } else if (instructionName == "rx" || instructionName == "crx" ||
               instructionName == "mcrx") {
      addOperation(qc, RX, qargs, params, qubitMap);
    } else if (instructionName == "ry" || instructionName == "cry" ||
               instructionName == "mcry") {
      addOperation(qc, RY, qargs, params, qubitMap);
    } else if (instructionName == "rz" || instructionName == "crz" ||
               instructionName == "mcrz") {
      addOperation(qc, RZ, qargs, params, qubitMap);
    } else if (instructionName == "p" || instructionName == "u1" ||
               instructionName == "cp" || instructionName == "cu1" ||
               instructionName == "mcphase") {
      addOperation(qc, Phase, qargs, params, qubitMap);
    } else if (instructionName == "sx" || instructionName == "csx") {
      addOperation(qc, SX, qargs, params, qubitMap);
    } else if (instructionName == "sxdg") {
      addOperation(qc, SXdag, qargs, params, qubitMap);
    } else if (instructionName == "u2") {
      addOperation(qc, U2, qargs, params, qubitMap);
    } else if (instructionName == "u" || instructionName == "u3" ||
               instructionName == "cu3") {
      addOperation(qc, U3, qargs, params, qubitMap);
    } else if (instructionName == "swap" || instructionName == "cswap") {
      addTwoTargetOperation(qc, SWAP, qargs, params, qubitMap);
    } else if (instructionName == "iswap") {
      addTwoTargetOperation(qc, iSWAP, qargs, params, qubitMap);
    } else if (instructionName == "dcx") {
      addTwoTargetOperation(qc, DCX, qargs, params, qubitMap);
    } else if (instructionName == "ecr") {
      addTwoTargetOperation(qc, ECR, qargs, params, qubitMap);
    } else if (instructionName == "rxx") {
      addTwoTargetOperation(qc, RXX, qargs, params, qubitMap);
    } else if (instructionName == "ryy") {
      addTwoTargetOperation(qc, RYY, qargs, params, qubitMap);
    } else if (instructionName == "rzx") {
      addTwoTargetOperation(qc, RZX, qargs, params, qubitMap);
    } else if (instructionName == "rzz") {
      addTwoTargetOperation(qc, RZZ, qargs, params, qubitMap);
    } else if (instructionName == "xx_minus_yy") {
      addTwoTargetOperation(qc, XXminusYY, qargs, params, qubitMap);
    } else if (instructionName == "xx_plus_yy") {
      addTwoTargetOperation(qc, XXplusYY, qargs, params, qubitMap);
    } else if (instructionName == "mcx_recursive") {
      if (qargs.size() <= 5) {
        addOperation(qc, X, qargs, params, qubitMap);
      } else {
        auto qargsCopy = nb::cast<nb::list>(qargs.attr("copy")());
        qargsCopy.attr("pop")(); // discard ancillaries
        addOperation(qc, X, qargsCopy, params, qubitMap);
      }
    } else if (instructionName == "mcx_vchain") {
      auto size = qargs.size();
      const std::size_t ncontrols = (size + 1) / 2;
      auto qargsCopy = nb::cast<nb::list>(qargs.attr("copy")());
      // discard ancillaries
      for (std::size_t i = 0; i < ncontrols - 2; ++i) {
        qargsCopy.attr("pop")();
      }
      addOperation(qc, X, qargsCopy, params, qubitMap);
    }
  } else {
    try {
      importDefinition(qc, instruction.attr("definition"), qargs, cargs,
                       qubitMap, clbitMap);
    } catch (nb::python_error& e) {
      std::cerr << "Failed to import_ instruction " << instructionName
                << " from Qiskit QuantumCircuit" << std::endl;
      std::cerr << e.what() << std::endl;
    }
  }
}

qc::SymbolOrNumber
qc::qiskit::QuantumCircuit::parseSymbolicExpr(const nb::object& pyExpr) {
  static const std::regex SUMMANDS("[+|-]?[^+-]+");
  static const std::regex PRODUCTS("[\\*/]?[^\\*/]+");

  auto exprStr = nb::cast<std::string>(pyExpr.attr("__str__")());
  exprStr.erase(std::remove(exprStr.begin(), exprStr.end(), ' '),
                exprStr.end()); // strip whitespace

  auto sumIt = std::sregex_iterator(exprStr.begin(), exprStr.end(), SUMMANDS);
  const auto sumEnd = std::sregex_iterator();

  qc::Symbolic sym;
  bool isConst = true;

  while (sumIt != sumEnd) {
    auto match = *sumIt;
    auto matchStr = match.str();
    const int sign = matchStr[0] == '-' ? -1 : 1;
    if (matchStr[0] == '+' || matchStr[0] == '-') {
      matchStr.erase(0, 1);
    }

    auto prodIt =
        std::sregex_iterator(matchStr.begin(), matchStr.end(), PRODUCTS);
    auto prodEnd = std::sregex_iterator();

    fp coeff = 1.0;
    std::string var;
    while (prodIt != prodEnd) {
      auto prodMatch = *prodIt;
      auto prodStr = prodMatch.str();

      const bool isDiv = prodStr[0] == '/';
      if (prodStr[0] == '*' || prodStr[0] == '/') {
        prodStr.erase(0, 1);
      }

      std::istringstream iss(prodStr);
      fp f{};
      iss >> f;

      if (iss.eof() && !iss.fail()) {
        coeff *= isDiv ? 1.0 / f : f;
      } else {
        var = prodStr;
      }

      ++prodIt;
    }
    if (var.empty()) {
      sym += coeff;
    } else {
      isConst = false;
      sym += sym::Term(sign * coeff, sym::Variable{var});
    }
    ++sumIt;
  }

  if (isConst) {
    return {sym.getConst()};
  }
  return {sym};
}

qc::SymbolOrNumber
qc::qiskit::QuantumCircuit::parseParam(const nb::object& param) {
  try {
    return nb::cast<fp>(param);
  } catch ([[maybe_unused]] nb::cast_error& e) {
    return parseSymbolicExpr(param);
  }
}

void qc::qiskit::QuantumCircuit::addOperation(qc::QuantumComputation& qc,
                                              qc::OpType type,
                                              const nb::list& qargs,
                                              const nb::list& params,
                                              const nb::dict& qubitMap) {
  std::vector<Control> qubits{};
  for (const auto qubit : qargs) {
    auto target = nb::cast<Qubit>(qubitMap[qubit]);
    qubits.emplace_back(Control{target});
  }
  auto target = qubits.back().qubit;
  qubits.pop_back();
  std::vector<qc::SymbolOrNumber> parameters{};
  for (const auto& param : params) {
    parameters.emplace_back(parseParam(nb::borrow<nb::object>(param)));
  }
  const Controls controls(qubits.cbegin(), qubits.cend());
  if (std::all_of(parameters.cbegin(), parameters.cend(), [](const auto& p) {
        return std::holds_alternative<fp>(p);
      })) {
    std::vector<fp> fpParams{};
    std::transform(parameters.cbegin(), parameters.cend(),
                   std::back_inserter(fpParams),
                   [](const auto& p) { return std::get<fp>(p); });
    qc.emplace_back<StandardOperation>(qc.getNqubits(), controls, target, type,
                                       fpParams);
  } else {
    qc.emplace_back<SymbolicOperation>(qc.getNqubits(), controls, target, type,
                                       parameters);
    for (const auto& p : parameters) {
      qc.addVariables(p);
    }
  }
}

void qc::qiskit::QuantumCircuit::addTwoTargetOperation(
    qc::QuantumComputation& qc, qc::OpType type, const nb::list& qargs,
    const nb::list& params, const nb::dict& qubitMap) {
  std::vector<Control> qubits{};
  for (const auto qubit : qargs) {
    auto target = nb::cast<Qubit>(qubitMap[qubit]);
    qubits.emplace_back(Control{target});
  }
  auto target1 = qubits.back().qubit;
  qubits.pop_back();
  auto target0 = qubits.back().qubit;
  qubits.pop_back();
  std::vector<qc::SymbolOrNumber> parameters{};
  for (const auto& param : params) {
    parameters.emplace_back(parseParam(nb::borrow<nb::object>(param)));
  }
  const Controls controls(qubits.cbegin(), qubits.cend());
  if (std::all_of(parameters.cbegin(), parameters.cend(), [](const auto& p) {
        return std::holds_alternative<fp>(p);
      })) {
    std::vector<fp> fpParams{};
    std::transform(parameters.cbegin(), parameters.cend(),
                   std::back_inserter(fpParams),
                   [](const auto& p) { return std::get<fp>(p); });
    qc.emplace_back<StandardOperation>(qc.getNqubits(), controls, target0,
                                       target1, type, fpParams);
  } else {
    qc.emplace_back<SymbolicOperation>(qc.getNqubits(), controls, target0,
                                       target1, type, parameters);
    for (const auto& p : parameters) {
      qc.addVariables(p);
    }
  }
}

void qc::qiskit::QuantumCircuit::importDefinition(
    qc::QuantumComputation& qc, const nb::object& circ, const nb::list& qargs,
    const nb::list& cargs, const nb::dict& qubitMap, const nb::dict& clbitMap) {
  const nb::dict qargMap{};
  nb::list&& defQubits = circ.attr("qubits");
  for (size_t i = 0; i < qargs.size(); ++i) {
    qargMap[defQubits[i]] = qargs[i];
  }

  const nb::dict cargMap{};
  nb::list&& defClbits = circ.attr("clbits");
  for (size_t i = 0; i < cargs.size(); ++i) {
    cargMap[defClbits[i]] = cargs[i];
  }

  auto&& data = circ.attr("data");
  for (const auto pyinst : data) {
    auto&& inst = nb::cast<std::tuple<nb::object, nb::list, nb::list>>(pyinst);
    auto&& instruction = std::get<0>(inst);

    const nb::list& instQargs = std::get<1>(inst);
    nb::list mappedQargs{};
    for (auto&& instQarg : instQargs) {
      mappedQargs.append(qargMap[instQarg]);
    }

    const nb::list& instCargs = std::get<2>(inst);
    nb::list mappedCargs{};
    for (auto&& instCarg : instCargs) {
      mappedCargs.append(cargMap[instCarg]);
    }

    auto&& instParams = instruction.attr("params");

    emplaceOperation(qc, instruction, mappedQargs, mappedCargs, instParams,
                     qubitMap, clbitMap);
  }
}

void qc::qiskit::QuantumCircuit::importInitialLayout(qc::QuantumComputation& qc,
                                                     const nb::object& circ) {
  const nb::object qubit = nb::module_::import_("qiskit.circuit").attr("Qubit");

  // get layout
  auto layout = circ.attr("_layout");

  // qiskit-terra 0.22.0 changed the `_layout` attribute to a
  // `TranspileLayout` dataclass object that contains the initial layout as a
  // `Layout` object in the `initial_layout` attribute.
  if (nb::hasattr(layout, "initial_layout")) {
    layout = layout.attr("initial_layout");
  }

  // create map between registers used in the layout and logical qubit indices
  // NOTE: this only works correctly if the registers were originally declared
  // in alphabetical order!
  const auto registers =
      layout.attr("get_registers")(); // potential cast to set necessary?
  std::size_t logicalQubitIndex = 0U;
  std::map<nb::object, std::size_t> logicalQubitIndices{};

  // the ancilla register
  decltype(registers.type()) ancillaRegister = nb::none();

  for (const auto qreg : registers) {
    // skip ancillary register since it is handled as the very last qubit
    // register
    if (const auto qregName = nb::cast<std::string>(qreg.attr("name"));
        qregName == "ancilla") {
      ancillaRegister = qreg;
      continue;
    }

    const auto size = nb::cast<std::size_t>(qreg.attr("size"));
    for (std::size_t i = 0U; i < size; ++i) {
      logicalQubitIndices[qubit(qreg, i)] = logicalQubitIndex;
      ++logicalQubitIndex;
    }
  }

  // handle ancillary register, if there is one
  if (!ancillaRegister.is_none()) {
    const auto size = nb::cast<std::size_t>(ancillaRegister.attr("size"));
    for (std::size_t i = 0U; i < size; ++i) {
      logicalQubitIndices[qubit(ancillaRegister, i)] = logicalQubitIndex;
      qc.setLogicalQubitAncillary(static_cast<Qubit>(logicalQubitIndex));
      ++logicalQubitIndex;
    }
  }

  // get a map of physical to logical qubits
  const auto physicalQubits =
      nb::cast<nb::dict>(layout.attr("get_physical_bits")());

  // create initial layout
  // for (const auto& [physicalQubit, logicalQubit] : physicalQubits) {
  //   if (logicalQubitIndices.find(logicalQubit) != logicalQubitIndices.end())
  //   {
  //     qc.initialLayout[nb::cast<Qubit>(physicalQubit)] =
  //       nb::cast<Qubit>(logicalQubitIndices[logicalQubit]);
  //   }
  // }
}
