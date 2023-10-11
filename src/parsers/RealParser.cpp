#include "QuantumComputation.hpp"

void qc::QuantumComputation::importReal(std::istream& is) {
  auto line = readRealHeader(is);
  readRealGateDescriptions(is, line);
}

int qc::QuantumComputation::readRealHeader(std::istream& is) {
  std::string cmd;
  std::string variable;
  int line = 0;

  while (true) {
    if (!static_cast<bool>(is >> cmd)) {
      throw QFRException("[real parser] l:" + std::to_string(line) +
                         " msg: Invalid file header");
    }
    std::transform(cmd.begin(), cmd.end(), cmd.begin(), [](unsigned char ch) {
      return static_cast<char>(toupper(ch));
    });
    ++line;

    // skip comments
    if (cmd.front() == '#') {
      is.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
      continue;
    }

    // valid header commands start with '.'
    if (cmd.front() != '.') {
      throw QFRException("[real parser] l:" + std::to_string(line) +
                         " msg: Invalid file header");
    }

    if (cmd == ".BEGIN") {
      // header read complete
      return line;
    }

    if (cmd == ".NUMVARS") {
      std::size_t nq{};
      if (!static_cast<bool>(is >> nq)) {
        nqubits = 0;
      } else {
        nqubits = nq;
      }
      nclassics = nqubits;
    } else if (cmd == ".VARIABLES") {
      for (std::size_t i = 0; i < nqubits; ++i) {
        if (!static_cast<bool>(is >> variable) || variable.at(0) == '.') {
          throw QFRException(
              "[real parser] l:" + std::to_string(line) +
              " msg: Invalid or insufficient variables declared");
        }
        const auto qubit = static_cast<Qubit>(i);
        qregs.insert({variable, {qubit, 1U}});
        cregs.insert({"c_" + variable, {qubit, 1U}});
        initialLayout.insert({qubit, qubit});
        outputPermutation.insert({qubit, qubit});
        ancillary.resize(nqubits);
        garbage.resize(nqubits);
      }
    } else if (cmd == ".CONSTANTS") {
      is >> std::ws;
      for (std::size_t i = 0; i < nqubits; ++i) {
        const auto value = is.get();
        if (!is.good()) {
          throw QFRException("[real parser] l:" + std::to_string(line) +
                             " msg: Failed read in '.constants' line");
        }
        if (value == '1') {
          x(static_cast<Qubit>(i));
        } else if (value != '-' && value != '0') {
          throw QFRException("[real parser] l:" + std::to_string(line) +
                             " msg: Invalid value in '.constants' header: '" +
                             std::to_string(value) + "'");
        }
      }
      is.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    } else if (cmd == ".INPUTS" || cmd == ".OUTPUTS" || cmd == ".GARBAGE" ||
               cmd == ".VERSION" || cmd == ".INPUTBUS" || cmd == ".OUTPUTBUS") {
      // TODO .inputs: specifies initial layout (and ancillaries)
      // TODO .outputs: specifies output permutation
      // TODO .garbage: specifies garbage outputs
      is.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
      continue;
    } else if (cmd == ".DEFINE") {
      // TODO: Defines currently not supported
      std::cerr << "[WARN] File contains 'define' statement, which is "
                   "currently not supported and thus simply skipped.\n";
      while (cmd != ".ENDDEFINE") {
        is.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        is >> cmd;
        std::transform(cmd.begin(), cmd.end(), cmd.begin(),
                       [](const unsigned char c) {
                         return static_cast<char>(toupper(c));
                       });
      }
    } else {
      throw QFRException("[real parser] l:" + std::to_string(line) +
                         " msg: Unknown command: " + cmd);
    }
  }
}

void qc::QuantumComputation::readRealGateDescriptions(std::istream& is,
                                                      int line) {
  const std::regex gateRegex =
      std::regex("(r[xyz]|i[df]|q|[0a-z](?:[+ip])?)(\\d+)?(?::([-+]?[0-9]+[.]?["
                 "0-9]*(?:[eE][-+]?[0-9]+)?))?");
  std::smatch m;
  std::string cmd;

  static const std::map<std::string, OpType> IDENTIFIER_MAP{
      {"0", I},     {"id", I},    {"h", H},        {"n", X},        {"c", X},
      {"x", X},     {"y", Y},     {"z", Z},        {"s", S},        {"si", Sdg},
      {"sp", Sdg},  {"s+", Sdg},  {"v", V},        {"vi", Vdg},     {"vp", Vdg},
      {"v+", Vdg},  {"rx", RX},   {"ry", RY},      {"rz", RZ},      {"f", SWAP},
      {"if", SWAP}, {"p", Peres}, {"pi", Peresdg}, {"p+", Peresdg}, {"q", P}};

  while (!is.eof()) {
    if (!static_cast<bool>(is >> cmd)) {
      throw QFRException("[real parser] l:" + std::to_string(line) +
                         " msg: Failed to read command");
    }
    std::transform(
        cmd.begin(), cmd.end(), cmd.begin(),
        [](const unsigned char c) { return static_cast<char>(tolower(c)); });
    ++line;

    if (cmd.front() == '#') {
      is.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
      continue;
    }

    if (cmd == ".end") {
      break;
    }

    // match gate declaration
    if (!std::regex_match(cmd, m, gateRegex)) {
      throw QFRException("[real parser] l:" + std::to_string(line) +
                         " msg: Unsupported gate detected: " + cmd);
    }

    // extract gate information (identifier, #controls, divisor)
    OpType gate{};
    if (m.str(1) == "t") { // special treatment of t(offoli) for real format
      gate = X;
    } else {
      auto it = IDENTIFIER_MAP.find(m.str(1));
      if (it == IDENTIFIER_MAP.end()) {
        throw QFRException("[real parser] l:" + std::to_string(line) +
                           " msg: Unknown gate identifier: " + m.str(1));
      }
      gate = (*it).second;
    }
    auto ncontrols =
        m.str(2).empty() ? 0 : std::stoul(m.str(2), nullptr, 0) - 1;
    const fp lambda = m.str(3).empty() ? static_cast<fp>(0L)
                                       : static_cast<fp>(std::stold(m.str(3)));

    if (gate == V || gate == Vdg || m.str(1) == "c" || gate == SWAP) {
      ncontrols = 1;
    } else if (gate == Peres || gate == Peresdg) {
      ncontrols = 2;
    }

    if (ncontrols >= nqubits) {
      throw QFRException("[real parser] l:" + std::to_string(line) +
                         " msg: Gate acts on " + std::to_string(ncontrols + 1) +
                         " qubits, but only " + std::to_string(nqubits) +
                         " qubits are available.");
    }

    std::string qubits;
    std::string label;
    getline(is, qubits);

    std::vector<Control> controls{};
    std::istringstream iss(qubits);

    // get controls and target
    for (std::size_t i = 0; i < ncontrols; ++i) {
      if (!(iss >> label)) {
        throw QFRException("[real parser] l:" + std::to_string(line) +
                           " msg: Too few variables for gate " + m.str(1));
      }

      const bool negativeControl = (label.at(0) == '-');
      if (negativeControl) {
        label.erase(label.begin());
      }

      auto iter = qregs.find(label);
      if (iter == qregs.end()) {
        throw QFRException("[real parser] l:" + std::to_string(line) +
                           " msg: Label " + label + " not found!");
      }
      controls.emplace_back(iter->second.first, negativeControl
                                                    ? Control::Type::Neg
                                                    : Control::Type::Pos);
    }

    if (!(iss >> label)) {
      throw QFRException("[real parser] l:" + std::to_string(line) +
                         " msg: Too few variables (no target) for gate " +
                         m.str(1));
    }
    auto iter = qregs.find(label);
    if (iter == qregs.end()) {
      throw QFRException("[real parser] l:" + std::to_string(line) +
                         " msg: Label " + label + " not found!");
    }

    updateMaxControls(ncontrols);
    const Qubit target = iter->second.first;
    switch (gate) {
    case I:
    case H:
    case Y:
    case Z:
    case S:
    case Sdg:
    case T:
    case Tdg:
    case V:
    case Vdg:
      emplace_back<StandardOperation>(
          nqubits, Controls{controls.cbegin(), controls.cend()}, target, gate);
      break;
    case X:
      mcx(Controls{controls.cbegin(), controls.cend()}, target);
      break;
    case RX:
    case RY:
    case RZ:
    case P:
      emplace_back<StandardOperation>(
          nqubits, Controls{controls.cbegin(), controls.cend()}, target, gate,
          std::vector{PI / (lambda)});
      break;
    case SWAP:
    case Peres:
    case Peresdg:
    case iSWAP: {
      const auto target1 = controls.back().qubit;
      controls.pop_back();
      emplace_back<StandardOperation>(
          nqubits, Controls{controls.cbegin(), controls.cend()}, target1,
          target, gate);
      break;
    }
    default:
      std::cerr << "Unsupported operation encountered:  " << gate << "!\n";
      break;
    }
  }
}
