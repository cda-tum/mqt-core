#include "QuantumComputation.hpp"

void qc::QuantumComputation::importQC(std::istream& is) {
  std::map<std::string, Qubit> varMap{};
  auto line = readQCHeader(is, varMap);
  readQCGateDescriptions(is, line, varMap);
}

int qc::QuantumComputation::readQCHeader(std::istream& is,
                                         std::map<std::string, Qubit>& varMap) {
  std::string cmd;
  std::string variable;
  std::string identifier;
  int line = 0;

  const std::string delimiter = " ";
  std::size_t pos{};

  std::vector<std::string> variables{};
  std::vector<std::string> inputs{};
  std::vector<std::string> outputs{};
  std::vector<std::string> constants{};

  while (true) {
    if (!static_cast<bool>(is >> cmd)) {
      throw QFRException("[qc parser] l:" + std::to_string(line) +
                         " msg: Invalid file header");
    }
    ++line;

    // skip comments
    if (cmd.front() == '#') {
      is.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
      continue;
    }

    // valid header commands start with '.' or end the header with BEGIN
    if (cmd.front() != '.' && cmd != "BEGIN" && cmd != "begin") {
      throw QFRException("[qc parser] l:" + std::to_string(line) +
                         " msg: Invalid file header");
    }

    // header read complete
    if (cmd == "BEGIN" || cmd == "begin") {
      break;
    }

    if (cmd == ".v") {
      is >> std::ws;
      std::getline(is, identifier);
      while ((pos = identifier.find(delimiter)) != std::string::npos) {
        variable = identifier.substr(0, pos);
        variables.emplace_back(variable);
        identifier.erase(0, pos + 1);
      }
      variables.emplace_back(identifier);
    } else if (cmd == ".i") {
      is >> std::ws;
      std::getline(is, identifier);
      while ((pos = identifier.find(delimiter)) != std::string::npos) {
        variable = identifier.substr(0, pos);
        if (std::find(variables.begin(), variables.end(), variable) !=
            variables.end()) {
          inputs.emplace_back(variable);
        } else {
          throw QFRException(
              "[qc parser] l:" + std::to_string(line) +
              " msg: Unknown variable in input statement: " + cmd);
        }
        identifier.erase(0, pos + 1);
      }
      if (std::find(variables.begin(), variables.end(), identifier) !=
          variables.end()) {
        inputs.emplace_back(identifier);
      } else {
        throw QFRException("[qc parser] l:" + std::to_string(line) +
                           " msg: Unknown variable in input statement: " + cmd);
      }
    } else if (cmd == ".o") {
      is >> std::ws;
      std::getline(is, identifier);
      while ((pos = identifier.find(delimiter)) != std::string::npos) {
        variable = identifier.substr(0, pos);
        if (std::find(variables.begin(), variables.end(), variable) !=
            variables.end()) {
          outputs.emplace_back(variable);
        } else {
          throw QFRException(
              "[qc parser] l:" + std::to_string(line) +
              " msg: Unknown variable in output statement: " + cmd);
        }
        identifier.erase(0, pos + 1);
      }
      if (std::find(variables.begin(), variables.end(), identifier) !=
          variables.end()) {
        outputs.emplace_back(identifier);
      } else {
        throw QFRException(
            "[qc parser] l:" + std::to_string(line) +
            " msg: Unknown variable in output statement: " + cmd);
      }
    } else if (cmd == ".c") {
      is >> std::ws;
      std::getline(is, identifier);
      while ((pos = identifier.find(delimiter)) != std::string::npos) {
        variable = identifier.substr(0, pos);
        constants.emplace_back(variable);
        identifier.erase(0, pos + 1);
      }
      constants.emplace_back(identifier);
    } else if (cmd == ".ol") { // ignore output labels
      is.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
      continue;
    } else {
      throw QFRException("[qc parser] l:" + std::to_string(line) +
                         " msg: Unknown command: " + cmd);
    }
  }
  addQubitRegister(inputs.size());
  auto nconstants = variables.size() - inputs.size();
  if (nconstants > 0) {
    addAncillaryRegister(nconstants);
  }

  auto qidx = 0;
  auto constidx = inputs.size();
  for (auto& var : variables) {
    // check if variable is input
    if (std::count(inputs.begin(), inputs.end(), var) != 0) {
      varMap.insert({var, qidx++});
    } else {
      if (!constants.empty()) {
        if (constants.at(constidx - inputs.size()) == "0" ||
            constants.at(constidx - inputs.size()) == "1") {
          // add X operation in case of initial value 1
          if (constants.at(constidx - inputs.size()) == "1") {
            emplace_back<StandardOperation>(nqubits + nancillae,
                                            static_cast<Qubit>(constidx), X);
          }
          varMap.insert({var, static_cast<Qubit>(constidx++)});
        } else {
          throw QFRException("[qc parser] l:" + std::to_string(line) +
                             " msg: Non-binary constant specified: " + cmd);
        }
      } else {
        // variable does not occur in input statement --> assumed to be |0>
        // ancillary
        varMap.insert({var, static_cast<Qubit>(constidx++)});
      }
    }
  }

  for (std::size_t q = 0; q < variables.size(); ++q) {
    variable = variables.at(q);
    auto p = varMap.at(variable);
    initialLayout[static_cast<Qubit>(q)] = p;
    if (!outputs.empty()) {
      if (std::count(outputs.begin(), outputs.end(), variable) != 0) {
        outputPermutation[static_cast<Qubit>(q)] = p;
      } else {
        outputPermutation.erase(static_cast<Qubit>(q));
        garbage.at(p) = true;
      }
    } else {
      // no output statement given --> assume all outputs are relevant
      outputPermutation[static_cast<Qubit>(q)] = p;
    }
  }

  return line;
}

void qc::QuantumComputation::readQCGateDescriptions(
    std::istream& is, int line, std::map<std::string, Qubit>& varMap) {
  const std::regex gateRegex = std::regex(
      R"((H|X|Y|Zd?|[SPT]\*?|tof|cnot|swap|R[xyz])(?:\((pi\/2\^(\d+)|(?:[-+]?[0-9]+[.]?[0-9]*(?:[eE][-+]?[0-9]+)?))\))?)");
  std::smatch m;
  std::string cmd;

  while (!is.eof()) {
    if (!static_cast<bool>(is >> cmd)) {
      throw QFRException("[qc parser] l:" + std::to_string(line) +
                         " msg: Failed to read command");
    }
    ++line;

    if (cmd.front() == '#') {
      is.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
      continue;
    }

    if (cmd == "END" || cmd == "end") {
      break;
    }

    // match gate declaration
    if (!std::regex_match(cmd, m, gateRegex)) {
      throw QFRException("[qc parser] l:" + std::to_string(line) +
                         " msg: Unsupported gate detected: " + cmd);
    }

    // extract gate information (identifier, #controls, divisor)
    auto lambda = static_cast<fp>(0L);
    OpType gate = None;
    const std::string gateType = m.str(1);
    if (gateType == "H") {
      gate = H;
    } else if (gateType == "X" || gateType == "cnot" || gateType == "tof") {
      gate = X;
    } else if (gateType == "Y") {
      gate = Y;
    } else if (gateType == "Z" || gateType == "Zd") {
      gate = Z;
    } else if (gateType == "P" || gateType == "S") {
      gate = S;
    } else if (gateType == "P*" || gateType == "S*") {
      gate = Sdg;
    } else if (gateType == "T") {
      gate = T;
    } else if (gateType == "T*") {
      gate = Tdg;
    } else if (gateType == "swap") {
      gate = SWAP;
    } else if (gateType == "Rx") {
      gate = RX;
    } else if (gateType == "Ry") {
      gate = RY;
    } else if (gateType == "Rz") {
      gate = RZ;
    }

    if (gate == RX || gate == RY || gate == RZ) {
      if (m.str(3).empty()) {
        // float definition
        lambda = static_cast<fp>(std::stold(m.str(2)));
      } else if (!m.str(2).empty()) {
        // pi/2^x definition
        auto power = std::stoul(m.str(3));
        if (power == 0UL) {
          lambda = PI;
        } else if (power == 1UL) {
          lambda = PI_2;
        } else if (power == 2UL) {
          lambda = PI_4;
        } else {
          lambda = PI_4 / (std::pow(static_cast<fp>(2), power - 2UL));
        }
      } else {
        throw QFRException("Rotation gate without angle detected");
      }
    }

    std::string qubits;
    std::string label;
    is >> std::ws;
    getline(is, qubits);

    std::vector<Control> controls{};

    auto delimiter = ' ';
    std::size_t pos{};

    while ((pos = qubits.find(delimiter)) != std::string::npos) {
      label = qubits.substr(0, pos);
      if (label.back() == '\'') {
        label.erase(label.size() - 1);
        controls.emplace_back(varMap.at(label), Control::Type::Neg);
      } else {
        controls.emplace_back(varMap.at(label));
      }
      qubits.erase(0, pos + 1);
    }
    // delete whitespace at the end
    qubits.erase(std::remove(qubits.begin(), qubits.end(), delimiter),
                 qubits.end());
    controls.emplace_back(varMap.at(qubits));

    if (controls.size() > nqubits + nancillae) {
      throw QFRException(
          "[qc parser] l:" + std::to_string(line) + " msg: Gate acts on " +
          std::to_string(controls.size()) + " qubits, but only " +
          std::to_string(nqubits + nancillae) + " qubits are available.");
    }

    if (gate == X) {
      const Qubit target = controls.back().qubit;
      controls.pop_back();
      mcx(Controls{controls.cbegin(), controls.cend()}, target);
    } else if (gate == H || gate == Y || gate == Z || gate == S ||
               gate == Sdg || gate == T || gate == Tdg) {
      const Qubit target = controls.back().qubit;
      controls.pop_back();
      emplace_back<StandardOperation>(
          nqubits, Controls{controls.cbegin(), controls.cend()}, target, gate);
    } else if (gate == SWAP) {
      const Qubit target0 = controls.back().qubit;
      controls.pop_back();
      const Qubit target1 = controls.back().qubit;
      controls.pop_back();
      mcswap(Controls{controls.cbegin(), controls.cend()}, target0, target1);
    } else if (gate == RX || gate == RY || gate == RZ) {
      const Qubit target = controls.back().qubit;
      controls.pop_back();
      emplace_back<StandardOperation>(
          nqubits, Controls{controls.cbegin(), controls.cend()}, target, gate,
          std::vector{lambda});
    }
  }
}
