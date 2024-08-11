#include "Definitions.hpp"
#include "QuantumComputation.hpp"
#include "operations/Control.hpp"
#include "operations/OpType.hpp"
#include "operations/StandardOperation.hpp"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <ios>
#include <iostream>
#include <istream>
#include <limits>
#include <map>
#include <optional>
#include <regex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

std::optional<qc::Qubit> getQubitForVariableIdentFromAnyLookup(
    const std::string& variableIdent, const qc::QuantumRegisterMap& dataQubits,
    const qc::QuantumRegisterMap& ancillaryQubits) {
  if (const auto& matchingEntryInDataQubits = dataQubits.find(variableIdent);
      matchingEntryInDataQubits != dataQubits.end())
    return matchingEntryInDataQubits->second.first;

  if (const auto& matchingEntryInAncillaryQubits =
          ancillaryQubits.find(variableIdent);
      matchingEntryInAncillaryQubits != ancillaryQubits.end())
    return matchingEntryInAncillaryQubits->second.first;

  return std::nullopt;
}

/// Determine whether the given io name value that is not enclosed in quotes
/// consists of only letters, digits and underscore characters.
/// @param ioName The name to valid
/// @return Whether the given io name is valid
bool isValidIoName(const std::string_view& ioName) {
  return !ioName.empty() &&
         std::all_of(
             ioName.cbegin(), ioName.cend(), [](const char ioNameCharacter) {
               return static_cast<bool>(std::isalnum(
                          static_cast<unsigned char>(ioNameCharacter))) ||
                      ioNameCharacter == '_';
             });
}

std::vector<std::string>
parseVariableNames(const std::size_t processedLineNumberInRealFile,
                   const std::size_t expectedNumberOfVariables,
                   const std::string& readInRawVariableIdentValues,
                   const std::unordered_set<std::string>& variableIdentsLookup,
                   bool allowedDuplicateVariableIdentDeclarations,
                   const std::string_view& trimableVariableIdentPrefix) {
  std::vector<std::string> variableNames;
  variableNames.reserve(expectedNumberOfVariables);

  std::unordered_set<std::string> processedVariableIdents;
  std::size_t variableIdentStartIdx = 0;
  std::size_t variableIdentEndIdx = 0;

  while (variableIdentStartIdx < readInRawVariableIdentValues.size() &&
         variableNames.size() <= expectedNumberOfVariables &&
         variableNames.size() < expectedNumberOfVariables) {
    variableIdentEndIdx =
        readInRawVariableIdentValues.find_first_of(' ', variableIdentStartIdx);

    if (variableIdentEndIdx == std::string::npos)
      variableIdentEndIdx = readInRawVariableIdentValues.size();

    std::size_t variableIdentLength =
        variableIdentEndIdx - variableIdentStartIdx;
    // On windows the line ending could be the character sequence \r\n while on
    // linux system it would only be \n
    if (variableIdentLength > 0 &&
        readInRawVariableIdentValues.at(std::min(
            variableIdentEndIdx, readInRawVariableIdentValues.size() - 1)) ==
            '\r')
      --variableIdentLength;

    auto variableIdent = readInRawVariableIdentValues.substr(
        variableIdentStartIdx, variableIdentLength);
    const bool trimVariableIdent =
        variableIdent.find_first_of(trimableVariableIdentPrefix) == 0;
    if (trimVariableIdent)
      variableIdent =
          variableIdent.replace(0, trimableVariableIdentPrefix.size(), "");

    if (!isValidIoName(variableIdent)) {
      throw qc::QFRException(
          "[real parser] l: " + std::to_string(processedLineNumberInRealFile) +
          " msg: invalid variable name: " + variableIdent);
    }

    if (!allowedDuplicateVariableIdentDeclarations &&
        processedVariableIdents.count(variableIdent) > 0) {
      throw qc::QFRException(
          "[real parser] l: " + std::to_string(processedLineNumberInRealFile) +
          " msg: duplicate variable name: " + variableIdent);
    }

    if (!variableIdentsLookup.empty() &&
        variableIdentsLookup.count(variableIdent) == 0) {
      throw qc::QFRException(
          "[real parser] l: " + std::to_string(processedLineNumberInRealFile) +
          " msg: given variable name " + variableIdent +
          " was not declared in .variables entry");
    }
    processedVariableIdents.emplace(variableIdent);
    variableNames.emplace_back(trimVariableIdent
                                   ? std::string(trimableVariableIdentPrefix) +
                                         variableIdent
                                   : variableIdent);
    variableIdentStartIdx = variableIdentEndIdx + 1;
  }

  if (variableIdentEndIdx < readInRawVariableIdentValues.size() &&
      readInRawVariableIdentValues.at(variableIdentEndIdx) == ' ') {
    throw qc::QFRException(
        "[real parser] l: " + std::to_string(processedLineNumberInRealFile) +
        " msg: expected only " + std::to_string(expectedNumberOfVariables) +
        " variable identifiers to be declared but variable identifier "
        "delimiter was found"
        " after " +
        std::to_string(expectedNumberOfVariables) +
        " identifiers were detected (which we assume will be followed by "
        "another io identifier)!");
  }

  if (variableNames.size() < expectedNumberOfVariables) {
    throw qc::QFRException(
        "[real parser] l:" + std::to_string(processedLineNumberInRealFile) +
        " msg: Expected " + std::to_string(expectedNumberOfVariables) +
        " variable idents but only " + std::to_string(variableNames.size()) +
        " were declared!");
  }
  return variableNames;
}

std::unordered_map<std::string, qc::Qubit>
parseIoNames(const std::size_t lineInRealFileDefiningIoNames,
             const std::size_t expectedNumberOfIos,
             const std::string& ioNameIdentsRawValues,
             const std::unordered_set<std::string>& variableIdentLookup) {
  std::unordered_map<std::string, qc::Qubit> foundIoNames;
  std::size_t ioNameStartIdx = 0;
  std::size_t ioNameEndIdx = 0;
  std::size_t ioIdx = 0;

  bool searchingForWhitespaceCharacter = false;
  while (ioNameStartIdx < ioNameIdentsRawValues.size() &&
         foundIoNames.size() <= expectedNumberOfIos) {
    searchingForWhitespaceCharacter =
        ioNameIdentsRawValues.at(ioNameStartIdx) != '"';
    if (searchingForWhitespaceCharacter)
      ioNameEndIdx = ioNameIdentsRawValues.find_first_of(' ', ioNameStartIdx);
    else
      ioNameEndIdx =
          ioNameIdentsRawValues.find_first_of('"', ioNameStartIdx + 1);

    if (ioNameEndIdx == std::string::npos) {
      ioNameEndIdx = ioNameIdentsRawValues.size();
      if (!searchingForWhitespaceCharacter) {
        throw qc::QFRException(
            "[real parser] l: " +
            std::to_string(lineInRealFileDefiningIoNames) +
            " no matching closing quote found for name of io: " +
            std::to_string(ioIdx));
      }
    } else {
      ioNameEndIdx +=
          static_cast<std::size_t>(!searchingForWhitespaceCharacter);
    }

    std::size_t ioNameLength = ioNameEndIdx - ioNameStartIdx;
    // On windows the line ending could be the character sequence \r\n while on
    // linux system it would only be \n
    if (ioNameLength > 0 &&
        ioNameIdentsRawValues.at(
            std::min(ioNameEndIdx, ioNameIdentsRawValues.size() - 1)) == '\r')
      --ioNameLength;

    const auto& ioName =
        ioNameIdentsRawValues.substr(ioNameStartIdx, ioNameLength);

    std::string_view ioNameToValidate = ioName;
    if (!searchingForWhitespaceCharacter) {
      ioNameToValidate =
          ioNameToValidate.substr(1, ioNameToValidate.size() - 2);
    }

    if (!isValidIoName(ioNameToValidate)) {
      throw qc::QFRException(
          "[real parser] l: " + std::to_string(lineInRealFileDefiningIoNames) +
          " msg: invalid io name: " + ioName);
    }

    if (variableIdentLookup.count(ioName) > 0) {
      throw qc::QFRException(
          "[real parser] l: " + std::to_string(lineInRealFileDefiningIoNames) +
          " msg: IO ident matched already declared variable with name " +
          ioName);
    }

    ioNameStartIdx = ioNameEndIdx + 1;
    if (const auto& ioNameInsertionIntoLookupResult =
            foundIoNames.emplace(ioName, static_cast<qc::Qubit>(ioIdx++));
        !ioNameInsertionIntoLookupResult.second) {
      throw qc::QFRException(
          "[real parser] l: " + std::to_string(lineInRealFileDefiningIoNames) +
          " msg: duplicate io name: " + ioName);
    }
  }

  if (searchingForWhitespaceCharacter &&
      ioNameEndIdx + 1 < ioNameIdentsRawValues.size() &&
      ioNameIdentsRawValues.at(ioNameEndIdx + 1) == ' ') {
    throw qc::QFRException(
        "[real parser] l: " + std::to_string(lineInRealFileDefiningIoNames) +
        " msg: expected only " + std::to_string(expectedNumberOfIos) +
        " io identifiers to be declared but io identifier delimiter was found"
        " after " +
        std::to_string(expectedNumberOfIos) +
        " identifiers were detected (which we assume will be followed by "
        "another io identifier)!");
  }
  return foundIoNames;
}

void qc::QuantumComputation::importReal(std::istream& is) {
  auto line = readRealHeader(is);
  readRealGateDescriptions(is, line);
}

int qc::QuantumComputation::readRealHeader(std::istream& is) {
  std::string cmd;
  int line = 0;

  /*
   * We could reuse the QuantumRegisterMap type defined in the qc namespace but
   * to avoid potential errors due to any future refactoring of said type, we
   * use an std::unordered_map instead
   */
  std::unordered_map<std::string, Qubit> userDefinedInputIdents;
  std::unordered_map<std::string, Qubit> userDefinedOutputIdents;
  std::unordered_set<std::string> userDeclaredVariableIdents;
  std::unordered_set<Qubit> outputQubitsMarkedAsGarbage;

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
      /*
       * The garbage declarations in the .real file are defined on the outputs
       * while the garbage state of the quantum computation operates on the
       * defined inputs, thus we perform a mapping from the output marked as
       * garbage back to the input using the output permutation.
       */
      for (const auto& outputQubitMarkedAsGarbage :
           outputQubitsMarkedAsGarbage) {
        /*
         * Since the call setLogicalQubitAsGarbage(...) assumes that the qubit
         * parameter is an input qubit, we need to manually mark the output
         * qubit as garbage by using the output qubit instead.
         */
        garbage[outputQubitMarkedAsGarbage] = true;
        outputPermutation.erase(outputQubitMarkedAsGarbage);
      }

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
      is >> std::ws;
      userDeclaredVariableIdents.reserve(nclassics);

      std::string variableDefinitionEntry;
      if (!std::getline(is, variableDefinitionEntry)) {
        throw QFRException("[real parser] l:" + std::to_string(line) +
                           " msg: Failed read in '.variables' line");
      }

      const auto& processedVariableIdents =
          parseVariableNames(static_cast<std::size_t>(line), nclassics,
                             variableDefinitionEntry, {}, false, "");
      userDeclaredVariableIdents.insert(processedVariableIdents.cbegin(),
                                        processedVariableIdents.cend());

      ancillary.resize(nqubits);
      garbage.resize(nqubits);
      for (std::size_t i = 0; i < nclassics; ++i) {
        const auto qubit = static_cast<Qubit>(i);
        qregs.insert({processedVariableIdents.at(i), {qubit, 1U}});
        cregs.insert({"c_" + processedVariableIdents.at(i), {qubit, 1U}});
        initialLayout.insert({qubit, qubit});
        outputPermutation.insert({qubit, qubit});
      }
    } else if (cmd == ".INITIAL_LAYOUT") {
      is >> std::ws;
      std::string initialLayoutDefinitionEntry;
      if (!std::getline(is, initialLayoutDefinitionEntry)) {
        throw QFRException("[real parser] l:" + std::to_string(line) +
                           " msg: Failed read in '.initial_layout' line");
      }

      const auto& processedVariableIdents = parseVariableNames(
          static_cast<std::size_t>(line), nclassics,
          initialLayoutDefinitionEntry, userDeclaredVariableIdents, false, "");

      /* Map the user declared variable idents in the .variable entry to the
       * ones declared in the .initial_layout as explained in
       * https://mqt.readthedocs.io/projects/core/en/latest/quickstart.html#layout-information
       */
      for (std::size_t i = 0; i < nclassics; ++i) {
        const auto algorithmicQubit = static_cast<Qubit>(i);
        const auto deviceQubitForVariableIdentInInitialLayout =
            qregs[processedVariableIdents.at(i)].first;
        initialLayout[deviceQubitForVariableIdentInInitialLayout] =
            algorithmicQubit;
      }
    } else if (cmd == ".CONSTANTS") {
      is >> std::ws;
      std::string constantsValuePerIoDefinition;
      if (!std::getline(is, constantsValuePerIoDefinition)) {
        throw QFRException("[real parser] l:" + std::to_string(line) +
                           " msg: Failed read in '.constants' line");
      }

      if (constantsValuePerIoDefinition.size() != nclassics) {
        throw QFRException(
            "[real parser] l: " + std::to_string(line) + " msg: Expected " +
            std::to_string(nclassics) + " constant values but " +
            std::to_string(constantsValuePerIoDefinition.size()) +
            " were declared!");
      }

      std::size_t constantValueIdx = 0;
      for (const auto constantValuePerIo : constantsValuePerIoDefinition) {
        if (const bool isCurrentQubitMarkedAsAncillary =
                constantValuePerIo == '0' || constantValuePerIo == '1';
            isCurrentQubitMarkedAsAncillary) {
          const auto& ancillaryQubit = static_cast<Qubit>(constantValueIdx);
          // Since ancillary qubits are assumed to have an initial value of
          // zero, we need to add an inversion gate to derive the correct
          // initial value of 1.
          if (constantValuePerIo == '1')
            x(ancillaryQubit);

          setLogicalQubitAncillary(ancillaryQubit);

          /*
           * Since the call to setLogicalQubitAncillary does not actually
           * transfer the qubit from the data qubit lookup into the ancillary
           * lookup we will 'manually' perform this transfer.
           */
          const std::string& associatedVariableNameForQubitRegister =
              getQubitRegister(ancillaryQubit);
          qregs.erase(associatedVariableNameForQubitRegister);
          ancregs.insert_or_assign(
              associatedVariableNameForQubitRegister,
              qc::QuantumRegister(std::make_pair(ancillaryQubit, 1U)));
        } else if (constantValuePerIo != '-') {
          throw QFRException("[real parser] l:" + std::to_string(line) +
                             " msg: Invalid value in '.constants' header: '" +
                             std::to_string(constantValuePerIo) + "'");
        }
        ++constantValueIdx;
      }
    } else if (cmd == ".GARBAGE") {
      is >> std::ws;
      std::string garbageStatePerIoDefinition;
      if (!std::getline(is, garbageStatePerIoDefinition)) {
        throw QFRException("[real parser] l:" + std::to_string(line) +
                           " msg: Failed read in '.garbage' line");
      }

      if (garbageStatePerIoDefinition.size() != nclassics) {
        throw QFRException("[real parser] l: " + std::to_string(line) +
                           " msg: Expected " + std::to_string(nclassics) +
                           " garbage state values but " +
                           std::to_string(garbageStatePerIoDefinition.size()) +
                           " were declared!");
      }

      std::size_t garbageStateIdx = 0;
      for (const auto garbageStateValue : garbageStatePerIoDefinition) {
        if (const bool isCurrentQubitMarkedAsGarbage = garbageStateValue == '1';
            isCurrentQubitMarkedAsGarbage) {
          outputQubitsMarkedAsGarbage.emplace(
              static_cast<Qubit>(garbageStateIdx));
        } else if (garbageStateValue != '-') {
          throw QFRException("[real parser] l:" + std::to_string(line) +
                             " msg: Invalid value in '.garbage' header: '" +
                             std::to_string(garbageStateValue) + "'");
        }
        garbageStateIdx++;
      }
    } else if (cmd == ".INPUTS") {
      // .INPUT: specifies initial layout
      is >> std::ws;
      const std::size_t expectedNumInputIos = nclassics;
      std::string ioNameIdentsLine;
      if (!std::getline(is, ioNameIdentsLine)) {
        throw QFRException("[real parser] l:" + std::to_string(line) +
                           " msg: Failed read in '.inputs' line");
      }

      userDefinedInputIdents =
          parseIoNames(static_cast<std::size_t>(line), expectedNumInputIos,
                       ioNameIdentsLine, userDeclaredVariableIdents);

      if (userDefinedInputIdents.size() != expectedNumInputIos) {
        throw QFRException(
            "[real parser] l: " + std::to_string(line) + "msg: Expected " +
            std::to_string(expectedNumInputIos) + " inputs to be declared!");
      }
    } else if (cmd == ".OUTPUTS") {
      // .OUTPUTS: specifies output permutation
      is >> std::ws;
      const std::size_t expectedNumOutputIos = nclassics;
      std::string ioNameIdentsLine;
      if (!std::getline(is, ioNameIdentsLine)) {
        throw QFRException("[real parser] l:" + std::to_string(line) +
                           " msg: Failed read in '.outputs' line");
      }

      userDefinedOutputIdents =
          parseIoNames(static_cast<std::size_t>(line), expectedNumOutputIos,
                       ioNameIdentsLine, userDeclaredVariableIdents);

      if (userDefinedOutputIdents.size() != expectedNumOutputIos) {
        throw QFRException(
            "[real parser] l: " + std::to_string(line) + "msg: Expected " +
            std::to_string(expectedNumOutputIos) + " outputs to be declared!");
      }

      if (userDefinedInputIdents.empty())
        continue;

      for (const auto& [outputIoIdent, outputIoQubit] :
           userDefinedOutputIdents) {
        /*
         * We assume that a permutation of a given input qubit Q at index i
         * is performed in the circuit if an entry in both in the .output
         * as well as the .input definition using the same literal is found,
         * with the input literal being defined at position i in the .input
         * definition. If no such matching is found, we require that the output
         * is marked as garbage.
         *
         * The outputPermutation map will use be structured as shown in the
         * documentation
         * (https://mqt.readthedocs.io/projects/core/en/latest/quickstart.html#layout-information)
         * with the output qubit being used as the key while the input qubit
         * serves as the map entries value.
         */
        if (userDefinedInputIdents.count(outputIoIdent) == 0) {
          /*
           * The current implementation requires that the .garbage definition is
           * define prior to the .output one.
           */
          if (outputQubitsMarkedAsGarbage.count(outputIoQubit) == 0) {
            throw QFRException("[real parser] l: " + std::to_string(line) +
                               " msg: outputs without matching inputs are "
                               "expected to be marked as garbage");
          }
        } else if (const Qubit matchingInputQubitForOutputLiteral =
                       userDefinedInputIdents.at(outputIoIdent);
                   matchingInputQubitForOutputLiteral != outputIoQubit &&
                   !logicalQubitIsGarbage(outputIoQubit)) {
          /*
           * We do not need to check whether a mapping from one input to any
           * output exists, since we require that the idents defined in either
           * of the .input as well as the .output definition are unique in their
           * definition.
           *
           * Only if the matching entries where defined at different indices
           * in their respective IO declaration do we update the existing 1-1
           * mapping for the given output qubit
           */
          outputPermutation.insert_or_assign(
              outputIoQubit, matchingInputQubitForOutputLiteral);

          /*
           * If we have determined a non-identity permutation of an input qubit,
           * (i.e. output 2 <- input 1) delete any existing identify permutation
           * of the input qubit since for the output 1 of the previous identity
           * mapping either another non-identity permutation must exist or the
           * output 1 must be declared as garbage.
           */
          if (outputPermutation.count(matchingInputQubitForOutputLiteral) > 0 &&
              outputPermutation[matchingInputQubitForOutputLiteral] ==
                  matchingInputQubitForOutputLiteral)
            outputPermutation.erase(matchingInputQubitForOutputLiteral);
        }
      }
    } else if (cmd == ".VERSION" || cmd == ".INPUTBUS" || cmd == ".OUTPUTBUS") {
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

    if (gate == V || gate == Vdg || m.str(1) == "c") {
      ncontrols = 1;
    } else if (gate == Peres || gate == Peresdg) {
      ncontrols = 2;
    }

    if (ncontrols >= getNqubits()) {
      throw QFRException("[real parser] l:" + std::to_string(line) +
                         " msg: Gate acts on " + std::to_string(ncontrols + 1) +
                         " qubits, but only " + std::to_string(getNqubits()) +
                         " qubits are available.");
    }

    std::string qubits;
    if (!getline(is, qubits)) {
      throw QFRException("[real parser] l:" + std::to_string(line) +
                         " msg: Failed read in gate definition");
    }

    const std::string& stringifiedNumberOfGateLines = m.str(2);
    const auto numberOfGateLines =
        stringifiedNumberOfGateLines.empty()
            ? 0
            : std::stoul(stringifiedNumberOfGateLines, nullptr, 0);
    // Current parser implementation defines number of expected control lines
    // (nControl) as nLines (of gate definition) - 1. Controlled swap gate has
    // at most two target lines so we define the number of control lines as
    // nLines - 2.
    if (gate == SWAP) {
      if (numberOfGateLines < 2) {
        throw QFRException("[real parser] l: " + std::to_string(line) +
                           "msg: SWAP gate is expected to operate on at least "
                           "two qubits but only " +
                           std::to_string(ncontrols) + " were defined");
      }
      ncontrols = numberOfGateLines - 2;
    }

    std::vector<Control> controls(ncontrols, Qubit());
    const auto& gateLines = qubits.empty() ? "" : qubits.substr(1);
    std::unordered_set<std::string> validVariableIdentLookup;

    /* Use the entries of the creg register map prefixed with 'c_' to determine
     * the declared variable idents in the .variable entry
     */
    for (const auto& qregNameAndQubitIndexPair : cregs)
      validVariableIdentLookup.emplace(
          qregNameAndQubitIndexPair.first.substr(2));

    // TODO: Check that no control line is used as a target line
    // We will ignore the prefix '-' when validating a given gate line ident
    auto processedGateLines =
        parseVariableNames(static_cast<std::size_t>(line), numberOfGateLines,
                           gateLines, validVariableIdentLookup, true, "-");

    std::size_t lineIdx = 0;
    // get controls and target
    for (std::size_t i = 0; i < ncontrols; ++i) {
      std::string_view gateIdent = processedGateLines.at(lineIdx++);
      const bool negativeControl = gateIdent.front() == '-';
      if (negativeControl)
        gateIdent = gateIdent.substr(1);

      // Since variable qubits can either be data or ancillary qubits our search
      // will have to be conducted in both lookups
      if (const std::optional<Qubit> controlLineQubit =
              getQubitForVariableIdentFromAnyLookup(std::string(gateIdent),
                                                    qregs, ancregs);
          controlLineQubit.has_value()) {
        controls.emplace_back(*controlLineQubit, negativeControl
                                                     ? Control::Type::Neg
                                                     : Control::Type::Pos);
      } else {
        throw QFRException("[real parser] l:" + std::to_string(line) +
                           " msg: Matching qubit for control line " +
                           std::string(gateIdent) + " not found!");
      }
    }

    const auto numberOfTargetLines = numberOfGateLines - ncontrols;
    std::vector targetLineQubits(numberOfTargetLines, Qubit());
    for (std::size_t i = 0; i < numberOfTargetLines; ++i) {
      const auto& targetLineIdent = processedGateLines.at(lineIdx++);
      // Since variable qubits can either be data or ancillary qubits our search
      // will have to be conducted in both lookups
      if (const std::optional<Qubit> targetLineQubit =
              getQubitForVariableIdentFromAnyLookup(targetLineIdent, qregs,
                                                    ancregs);
          targetLineQubit.has_value()) {
        targetLineQubits[i] = *targetLineQubit;
      } else {
        throw QFRException("[real parser] l:" + std::to_string(line) +
                           " msg: Matching qubit for target line " +
                           targetLineIdent + " not found!");
      }
    }

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
          Controls{controls.cbegin(), controls.cend()},
          targetLineQubits.front(), gate);
      break;
    case X:
      mcx(Controls{controls.cbegin(), controls.cend()},
          targetLineQubits.front());
      break;
    case RX:
    case RY:
    case RZ:
    case P:
      emplace_back<StandardOperation>(
          Controls{controls.cbegin(), controls.cend()},
          targetLineQubits.front(), gate, std::vector{PI / (lambda)});
      break;
    case SWAP:
    case iSWAP:
      emplace_back<StandardOperation>(
          Controls{controls.cbegin(), controls.cend()},
          Targets{targetLineQubits.cbegin(), targetLineQubits.cend()}, gate);
      break;
    case Peres:
    case Peresdg: {
      const auto target1 = controls.back().qubit;
      controls.pop_back();
      emplace_back<StandardOperation>(
          Controls{controls.cbegin(), controls.cend()}, target1,
          targetLineQubits.front(), gate);
      break;
    }
    default:
      std::cerr << "Unsupported operation encountered:  " << gate << "!\n";
      break;
    }
  }
}
