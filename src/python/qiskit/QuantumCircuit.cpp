#include "python/qiskit/QuantumCircuit.hpp"

void qc::qiskit::QuantumCircuit::import(qc::QuantumComputation& qc,
                                        const py::object& circ) {
  qc.reset();

  const py::object quantumCircuit =
      py::module::import("qiskit").attr("QuantumCircuit");

  if (!py::isinstance(circ, quantumCircuit)) {
    throw QFRException(
        "[import] Python object needs to be a Qiskit QuantumCircuit");
  }

  if (!circ.attr("name").is_none()) {
    qc.setName(circ.attr("name").cast<std::string>());
  }

  // handle qubit registers
  const py::object qubit = py::module::import("qiskit.circuit").attr("Qubit");
  const py::object ancillaQubit =
      py::module::import("qiskit.circuit").attr("AncillaQubit");
  const py::object ancillaRegister =
      py::module::import("qiskit.circuit").attr("AncillaRegister");
  int qubitIndex = 0;
  const py::dict qubitMap{};
  auto&& circQregs = circ.attr("qregs");
  for (const auto qreg : circQregs) {
    // create corresponding register in quantum computation
    auto size = qreg.attr("size").cast<std::size_t>();
    auto name = qreg.attr("name").cast<std::string>();
    if (py::isinstance(qreg, ancillaRegister)) {
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
  const py::object clbit = py::module::import("qiskit.circuit").attr("Clbit");
  int clbitIndex = 0;
  const py::dict clbitMap{};
  auto&& circCregs = circ.attr("cregs");
  for (const auto creg : circCregs) {
    // create corresponding register in quantum computation
    auto size = creg.attr("size").cast<std::size_t>();
    auto name = creg.attr("name").cast<std::string>();
    qc.addClassicalRegister(size, name);

    // add clbits to clbit map
    for (std::size_t i = 0; i < size; ++i) {
      clbitMap[clbit(creg, i)] = clbitIndex;
      clbitIndex++;
    }
  }

  try {
    qc.gphase(circ.attr("global_phase").cast<fp>());
  } catch (const py::cast_error& e) {
    std::clog << e.what() << "\n";
    std::clog << "[import] Warning: Symbolic global phase values are not "
                 "supported yet. Ignoring global phase.\n";
  }

  // iterate over instructions
  auto&& data = circ.attr("data");
  for (const auto pyinst : data) {
    auto&& inst = pyinst.cast<std::tuple<py::object, py::list, py::list>>();
    auto&& instruction = std::get<0>(inst);
    auto&& qargs = std::get<1>(inst);
    auto&& cargs = std::get<2>(inst);
    auto&& params = instruction.attr("params");

    emplaceOperation(qc, instruction, qargs, cargs, params, qubitMap, clbitMap);
  }

  // import initial layout in case it is available
  if (!circ.attr("_layout").is_none()) {
    importInitialLayout(qc, circ);
  }
  qc.initializeIOMapping();
}

void qc::qiskit::QuantumCircuit::emplaceOperation(
    qc::QuantumComputation& qc, const py::object& instruction,
    const py::list& qargs, const py::list& cargs, const py::list& params,
    const py::dict& qubitMap, const py::dict& clbitMap) {
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

  auto instructionName = instruction.attr("name").cast<std::string>();
  if (instructionName == "measure") {
    auto control = qubitMap[qargs[0]].cast<Qubit>();
    auto target = clbitMap[cargs[0]].cast<std::size_t>();
    qc.emplace_back<NonUnitaryOperation>(qc.getNqubits(), control, target);
  } else if (instructionName == "barrier") {
    Targets targets{};
    for (const auto qubit : qargs) {
      auto target = qubitMap[qubit].cast<Qubit>();
      targets.emplace_back(target);
    }
    qc.emplace_back<StandardOperation>(qc.getNqubits(), targets, Barrier);
  } else if (instructionName == "reset") {
    Targets targets{};
    for (const auto qubit : qargs) {
      auto target = qubitMap[qubit].cast<Qubit>();
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
      addOperation(qc, Sdg, qargs, params, qubitMap);
    } else if (instructionName == "t") {
      addOperation(qc, T, qargs, params, qubitMap);
    } else if (instructionName == "tdg") {
      addOperation(qc, Tdg, qargs, params, qubitMap);
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
      addOperation(qc, P, qargs, params, qubitMap);
    } else if (instructionName == "sx" || instructionName == "csx") {
      addOperation(qc, SX, qargs, params, qubitMap);
    } else if (instructionName == "sxdg") {
      addOperation(qc, SXdg, qargs, params, qubitMap);
    } else if (instructionName == "u2") {
      addOperation(qc, U2, qargs, params, qubitMap);
    } else if (instructionName == "u" || instructionName == "u3" ||
               instructionName == "cu3") {
      addOperation(qc, U, qargs, params, qubitMap);
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
        auto qargsCopy = qargs.attr("copy")();
        qargsCopy.attr("pop")(); // discard ancillaries
        addOperation(qc, X, qargsCopy, params, qubitMap);
      }
    } else if (instructionName == "mcx_vchain") {
      auto size = qargs.size();
      const std::size_t ncontrols = (size + 1) / 2;
      auto qargsCopy = qargs.attr("copy")();
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
    } catch (py::error_already_set& e) {
      std::cerr << "Failed to import instruction " << instructionName
                << " from Qiskit QuantumCircuit" << std::endl;
      std::cerr << e.what() << std::endl;
    }
  }
}

qc::SymbolOrNumber
qc::qiskit::QuantumCircuit::parseSymbolicExpr(const py::object& pyExpr) {
  static const std::regex SUMMANDS("[+|-]?[^+-]+");
  static const std::regex PRODUCTS("[\\*/]?[^\\*/]+");

  auto exprStr = pyExpr.attr("__str__")().cast<std::string>();
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
      sym += sym::Term(sym::Variable{var}, sign * coeff);
    }
    ++sumIt;
  }

  if (isConst) {
    return {sym.getConst()};
  }
  return {sym};
}

qc::SymbolOrNumber
qc::qiskit::QuantumCircuit::parseParam(const py::object& param) {
  try {
    return param.cast<fp>();
  } catch ([[maybe_unused]] py::cast_error& e) {
    return parseSymbolicExpr(param);
  }
}

void qc::qiskit::QuantumCircuit::addOperation(qc::QuantumComputation& qc,
                                              qc::OpType type,
                                              const py::list& qargs,
                                              const py::list& params,
                                              const py::dict& qubitMap) {
  std::vector<Control> qubits{};
  for (const auto qubit : qargs) {
    auto target = qubitMap[qubit].cast<Qubit>();
    qubits.emplace_back(Control{target});
  }
  auto target = qubits.back().qubit;
  qubits.pop_back();
  std::vector<qc::SymbolOrNumber> parameters{};
  for (const auto& param : params) {
    parameters.emplace_back(
        parseParam(py::reinterpret_borrow<py::object>(param)));
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
    qc::QuantumComputation& qc, qc::OpType type, const py::list& qargs,
    const py::list& params, const py::dict& qubitMap) {
  std::vector<Control> qubits{};
  for (const auto qubit : qargs) {
    auto target = qubitMap[qubit].cast<Qubit>();
    qubits.emplace_back(Control{target});
  }
  auto target1 = qubits.back().qubit;
  qubits.pop_back();
  auto target0 = qubits.back().qubit;
  qubits.pop_back();
  std::vector<qc::SymbolOrNumber> parameters{};
  for (const auto& param : params) {
    parameters.emplace_back(
        parseParam(py::reinterpret_borrow<py::object>(param)));
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
    qc::QuantumComputation& qc, const py::object& circ, const py::list& qargs,
    const py::list& cargs, const py::dict& qubitMap, const py::dict& clbitMap) {
  const py::dict qargMap{};
  py::list&& defQubits = circ.attr("qubits");
  for (size_t i = 0; i < qargs.size(); ++i) {
    qargMap[defQubits[i]] = qargs[i];
  }

  const py::dict cargMap{};
  py::list&& defClbits = circ.attr("clbits");
  for (size_t i = 0; i < cargs.size(); ++i) {
    cargMap[defClbits[i]] = cargs[i];
  }

  auto&& data = circ.attr("data");
  for (const auto pyinst : data) {
    auto&& inst = pyinst.cast<std::tuple<py::object, py::list, py::list>>();
    auto&& instruction = std::get<0>(inst);

    const py::list& instQargs = std::get<1>(inst);
    py::list mappedQargs{};
    for (auto&& instQarg : instQargs) {
      mappedQargs.append(qargMap[instQarg]);
    }

    const py::list& instCargs = std::get<2>(inst);
    py::list mappedCargs{};
    for (auto&& instCarg : instCargs) {
      mappedCargs.append(cargMap[instCarg]);
    }

    auto&& instParams = instruction.attr("params");

    emplaceOperation(qc, instruction, mappedQargs, mappedCargs, instParams,
                     qubitMap, clbitMap);
  }
}

void qc::qiskit::QuantumCircuit::importInitialLayout(qc::QuantumComputation& qc,
                                                     const py::object& circ) {
  const py::object qubit = py::module::import("qiskit.circuit").attr("Qubit");

  // get layout
  auto layout = circ.attr("_layout");

  // qiskit-terra 0.22.0 changed the `_layout` attribute to a
  // `TranspileLayout` dataclass object that contains the initial layout as a
  // `Layout` object in the `initial_layout` attribute.
  if (py::hasattr(layout, "initial_layout")) {
    layout = layout.attr("initial_layout");
  }

  // create map between registers used in the layout and logical qubit indices
  // NOTE: this only works correctly if the registers were originally declared
  // in alphabetical order!
  const auto registers = layout.attr("get_registers")().cast<py::set>();
  std::size_t logicalQubitIndex = 0U;
  const py::dict logicalQubitIndices{};

  // the ancilla register
  decltype(registers.get_type()) ancillaRegister = py::none();

  for (const auto qreg : registers) {
    // skip ancillary register since it is handled as the very last qubit
    // register
    if (const auto qregName = qreg.attr("name").cast<std::string>();
        qregName == "ancilla") {
      ancillaRegister = qreg;
      continue;
    }

    const auto size = qreg.attr("size").cast<std::size_t>();
    for (std::size_t i = 0U; i < size; ++i) {
      logicalQubitIndices[qubit(qreg, i)] = logicalQubitIndex;
      ++logicalQubitIndex;
    }
  }

  // handle ancillary register, if there is one
  if (!ancillaRegister.is_none()) {
    const auto size = ancillaRegister.attr("size").cast<std::size_t>();
    for (std::size_t i = 0U; i < size; ++i) {
      logicalQubitIndices[qubit(ancillaRegister, i)] = logicalQubitIndex;
      qc.setLogicalQubitAncillary(static_cast<Qubit>(logicalQubitIndex));
      ++logicalQubitIndex;
    }
  }

  // get a map of physical to logical qubits
  const auto physicalQubits =
      layout.attr("get_physical_bits")().cast<py::dict>();

  // create initial layout (and assume identical output permutation)
  for (const auto& [physicalQubit, logicalQubit] : physicalQubits) {
    if (logicalQubitIndices.contains(logicalQubit)) {
      qc.initialLayout[physicalQubit.cast<Qubit>()] =
          logicalQubitIndices[logicalQubit].cast<Qubit>();
      qc.outputPermutation[physicalQubit.cast<Qubit>()] =
          logicalQubitIndices[logicalQubit].cast<Qubit>();
    }
  }
}
