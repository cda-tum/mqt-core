#include "Definitions.hpp"
#include "ir/Permutation.hpp"
#include "ir/QuantumComputation.hpp"

#include "gmock/gmock-matchers.h"
#include <cstddef>
#include <cstdint>
#include <functional>
#include <gtest/gtest.h>
#include <initializer_list>
#include <iomanip>
#include <ios>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

using namespace qc;
using ::testing::NotNull;

class RealParserTest : public testing::Test {
public:
  RealParserTest& usingVersion(double versionNumber) {
    realFileContent << realHeaderVersionCommandPrefix << " " << std::fixed
                    << std::setprecision(1) << versionNumber << "\n";
    return *this;
  }

  RealParserTest& usingNVariables(std::size_t numVariables) {
    realFileContent << realHeaderNumVarsCommandPrefix << " "
                    << std::to_string(numVariables) << "\n";
    return *this;
  }

  RealParserTest& usingVariables(
      const std::initializer_list<std::string_view>& variableIdents) {
    realFileContent << realHeaderVariablesCommandPrefix;
    for (const auto& variableIdent : variableIdents)
      realFileContent << " " << variableIdent;

    realFileContent << "\n";
    return *this;
  }

  RealParserTest& usingInitialLayout(
      const std::initializer_list<std::string_view>& variableIdents) {
    realFileContent << realHeaderInitialLayoutCommandPrefix;
    for (const auto& variableIdent : variableIdents)
      realFileContent << " " << variableIdent;

    realFileContent << "\n";
    return *this;
  }

  RealParserTest&
  usingInputs(const std::initializer_list<std::string_view>& inputIdents) {
    realFileContent << realHeaderInputCommandPrefix;
    for (const auto& inputIdent : inputIdents)
      realFileContent << " " << inputIdent;

    realFileContent << "\n";
    return *this;
  }

  RealParserTest&
  usingOutputs(const std::initializer_list<std::string_view>& outputIdents) {
    realFileContent << realHeaderOutputCommandPrefix;
    for (const auto& outputIdent : outputIdents)
      realFileContent << " " << outputIdent;

    realFileContent << "\n";
    return *this;
  }

  RealParserTest&
  withConstants(const std::initializer_list<char>& constantValuePerVariable) {
    realFileContent << realHeaderConstantsCommandPrefix << " ";
    for (const auto& constantValue : constantValuePerVariable)
      realFileContent << constantValue;

    realFileContent << "\n";
    return *this;
  }

  RealParserTest& withGarbageValues(
      const std::initializer_list<char>& isGarbageValuePerVariable) {
    realFileContent << realHeaderGarbageCommandPrefix << " ";
    for (const auto& garbageValue : isGarbageValuePerVariable)
      realFileContent << garbageValue;

    realFileContent << "\n";
    return *this;
  }

  RealParserTest& withEmptyGateList() {
    realFileContent << realHeaderGateListPrefix << "\n"
                    << reakHeaderGateListPostfix;
    return *this;
  }

  RealParserTest& withGates(
      const std::initializer_list<std::string_view>& stringifiedGateList) {
    if (stringifiedGateList.size() == 0)
      return withEmptyGateList();

    realFileContent << realHeaderGateListPrefix << "\n";
    for (const auto& stringifiedGate : stringifiedGateList)
      realFileContent << stringifiedGate << "\n";

    realFileContent << reakHeaderGateListPostfix;
    return *this;
  }

protected:
  const std::string realHeaderVersionCommandPrefix = ".version";
  const std::string realHeaderNumVarsCommandPrefix = ".numvars";
  const std::string realHeaderVariablesCommandPrefix = ".variables";
  const std::string realHeaderInitialLayoutCommandPrefix = ".initial_layout";
  const std::string realHeaderInputCommandPrefix = ".inputs";
  const std::string realHeaderOutputCommandPrefix = ".outputs";
  const std::string realHeaderConstantsCommandPrefix = ".constants";
  const std::string realHeaderGarbageCommandPrefix = ".garbage";
  const std::string realHeaderGateListPrefix = ".begin";
  const std::string reakHeaderGateListPostfix = ".end";

  static constexpr double DEFAULT_REAL_VERSION = 2.0;

  const char constantValueZero = '0';
  const char constantValueOne = '1';
  const char constantValueNone = '-';

  const char isGarbageState = '1';
  const char isNotGarbageState = '-';

  enum class GateType : std::uint8_t { Toffoli, V };

  std::unique_ptr<QuantumComputation> quantumComputationInstance;
  std::stringstream realFileContent;

  void SetUp() override {
    quantumComputationInstance = std::make_unique<QuantumComputation>();
    ASSERT_THAT(quantumComputationInstance, NotNull());
  }

  static Permutation getIdentityPermutation(std::size_t nQubits) {
    auto identityPermutation = Permutation();
    for (std::size_t i = 0; i < nQubits; ++i) {
      const auto qubit = static_cast<Qubit>(i);
      identityPermutation.insert({qubit, qubit});
    }
    return identityPermutation;
  }

  static std::string stringifyGateType(const GateType gateType) {
    if (gateType == GateType::Toffoli)
      return "t";
    if (gateType == GateType::V)
      return "v";

    throw std::invalid_argument("Failed to stringify gate type");
  }

  static std::string
  stringifyGate(const GateType gateType,
                const std::initializer_list<std::string_view>& controlLines,
                const std::initializer_list<std::string_view>& targetLines) {
    return stringifyGate(gateType, std::nullopt, controlLines, targetLines);
  }

  static std::string
  stringifyGate(const GateType gateType,
                const std::optional<std::size_t>& optionalNumberOfGateLines,
                const std::initializer_list<std::string_view>& controlLines,
                const std::initializer_list<std::string_view>& targetLines) {
    EXPECT_TRUE(targetLines.size() > static_cast<std::size_t>(0))
        << "Gate must have at least one line defined";

    std::stringstream stringifiedGateBuffer;
    if (controlLines.size() == 0 && !optionalNumberOfGateLines.has_value())
      stringifiedGateBuffer << stringifyGateType(gateType);
    else
      stringifiedGateBuffer
          << stringifyGateType(gateType)
          << std::to_string(optionalNumberOfGateLines.value_or(
                 controlLines.size() + targetLines.size()));

    for (const auto& controlLine : controlLines)
      stringifiedGateBuffer << " " << controlLine;

    for (const auto& targetLine : targetLines)
      stringifiedGateBuffer << " " << targetLine;

    return stringifiedGateBuffer.str();
  }
};

// ERROR TESTS
TEST_F(RealParserTest, MoreVariablesThanNumVariablesDeclared) {
  usingVersion(DEFAULT_REAL_VERSION)
      .usingNVariables(2)
      .usingVariables({"v1", "v2", "v3"})
      .withEmptyGateList();

  EXPECT_THROW(
      quantumComputationInstance->import(realFileContent, Format::Real),
      QFRException);
}

TEST_F(RealParserTest, MoreInputsThanVariablesDeclared) {
  usingVersion(DEFAULT_REAL_VERSION)
      .usingNVariables(2)
      .usingVariables({"v1", "v2"})
      .usingInputs({"i1", "i2", "i3"})
      .withEmptyGateList();

  EXPECT_THROW(
      quantumComputationInstance->import(realFileContent, Format::Real),
      QFRException);
}

TEST_F(RealParserTest, MoreOutputsThanVariablesDeclared) {
  usingVersion(DEFAULT_REAL_VERSION)
      .usingNVariables(2)
      .usingVariables({"v1", "v2"})
      .usingOutputs({"o1", "o2", "o3"})
      .withEmptyGateList();

  EXPECT_THROW(
      quantumComputationInstance->import(realFileContent, Format::Real),
      QFRException);
}

TEST_F(RealParserTest, MoreConstantsThanVariablesDeclared) {
  usingVersion(DEFAULT_REAL_VERSION)
      .usingNVariables(2)
      .usingVariables({"v1", "v2"})
      .withConstants({constantValueZero, constantValueZero, constantValueZero})
      .withEmptyGateList();

  EXPECT_THROW(
      quantumComputationInstance->import(realFileContent, Format::Real),
      QFRException);
}

TEST_F(RealParserTest, MoreGarbageEntriesThanVariablesDeclared) {
  usingVersion(DEFAULT_REAL_VERSION)
      .usingNVariables(2)
      .usingVariables({"v1", "v2"})
      .withGarbageValues({isGarbageState, isNotGarbageState, isGarbageState})
      .withEmptyGateList();

  EXPECT_THROW(
      quantumComputationInstance->import(realFileContent, Format::Real),
      QFRException);
}

TEST_F(RealParserTest, MoreIdentsInInitialLayoutThanVariablesDeclared) {
  usingVersion(DEFAULT_REAL_VERSION)
      .usingNVariables(2)
      .usingVariables({"v1", "v2"})
      .usingInitialLayout({"v1", "v2", "v3"})
      .withEmptyGateList();

  EXPECT_THROW(
      quantumComputationInstance->import(realFileContent, Format::Real),
      QFRException);
}

TEST_F(RealParserTest, LessVariablesThanNumVariablesDeclared) {
  usingVersion(DEFAULT_REAL_VERSION)
      .usingNVariables(2)
      .usingVariables({"v1"})
      .withEmptyGateList();

  EXPECT_THROW(
      quantumComputationInstance->import(realFileContent, Format::Real),
      QFRException);
}

TEST_F(RealParserTest, LessInputsThanNumVariablesDeclared) {
  usingVersion(DEFAULT_REAL_VERSION)
      .usingNVariables(2)
      .usingVariables({"v1", "v2"})
      .usingInputs({"i1"})
      .withEmptyGateList();

  EXPECT_THROW(
      quantumComputationInstance->import(realFileContent, Format::Real),
      QFRException);
}

TEST_F(RealParserTest, LessOutputsThanNumVariablesDeclared) {
  usingVersion(DEFAULT_REAL_VERSION)
      .usingNVariables(2)
      .usingVariables({"v1", "v2"})
      .usingOutputs({"o1"})
      .withEmptyGateList();

  EXPECT_THROW(
      quantumComputationInstance->import(realFileContent, Format::Real),
      QFRException);
}

TEST_F(RealParserTest, LessConstantsThanNumVariablesDeclared) {
  usingVersion(DEFAULT_REAL_VERSION)
      .usingNVariables(2)
      .usingVariables({"v1", "v2"})
      .withConstants({constantValueNone})
      .withEmptyGateList();

  EXPECT_THROW(
      quantumComputationInstance->import(realFileContent, Format::Real),
      QFRException);
}

TEST_F(RealParserTest, LessGarbageEntriesThanNumVariablesDeclared) {
  usingVersion(DEFAULT_REAL_VERSION)
      .usingNVariables(2)
      .usingVariables({"v1", "v2"})
      .withGarbageValues({isNotGarbageState})
      .withEmptyGateList();

  EXPECT_THROW(
      quantumComputationInstance->import(realFileContent, Format::Real),
      QFRException);
}

TEST_F(RealParserTest, LessIdentsInInitialLayoutThanVariablesDeclared) {
  usingVersion(DEFAULT_REAL_VERSION)
      .usingNVariables(2)
      .usingVariables({"v1", "v2"})
      .usingInitialLayout({"v2"})
      .withEmptyGateList();

  EXPECT_THROW(
      quantumComputationInstance->import(realFileContent, Format::Real),
      QFRException);
}

TEST_F(RealParserTest, InvalidVariableIdentDeclaration) {
  usingVersion(DEFAULT_REAL_VERSION)
      .usingNVariables(2)
      .usingVariables({"variable-1", "v2"})
      .withEmptyGateList();

  EXPECT_THROW(
      quantumComputationInstance->import(realFileContent, Format::Real),
      QFRException);
}

TEST_F(RealParserTest, InvalidInputIdentDeclaration) {
  usingVersion(DEFAULT_REAL_VERSION)
      .usingNVariables(2)
      .usingVariables({"v1", "v2"})
      .usingInputs({"test-input1", "i2"})
      .withEmptyGateList();

  EXPECT_THROW(
      quantumComputationInstance->import(realFileContent, Format::Real),
      QFRException);
}

TEST_F(RealParserTest, InvalidInputIdentDeclarationInQuote) {
  usingVersion(DEFAULT_REAL_VERSION)
      .usingNVariables(2)
      .usingVariables({"v1", "v2"})
      .usingInputs({"\"test-input1\"", "i2"})
      .withEmptyGateList();

  EXPECT_THROW(
      quantumComputationInstance->import(realFileContent, Format::Real),
      QFRException);
}

TEST_F(RealParserTest, InvalidVariableIdentDeclarationInInitialLayout) {
  usingVersion(DEFAULT_REAL_VERSION)
      .usingNVariables(2)
      .usingVariables({"v1", "v2"})
      .usingInitialLayout({"v-1", "v2"})
      .withEmptyGateList();

  EXPECT_THROW(
      quantumComputationInstance->import(realFileContent, Format::Real),
      QFRException);
}

TEST_F(RealParserTest, EmptyInputIdentInQuotesNotAllowed) {
  usingVersion(DEFAULT_REAL_VERSION)
      .usingNVariables(2)
      .usingVariables({"v1", "v2"})
      .usingInputs({"i1", "\"\""})
      .withEmptyGateList();

  EXPECT_THROW(
      quantumComputationInstance->import(realFileContent, Format::Real),
      QFRException);
}

TEST_F(RealParserTest, InvalidOutputIdentDeclaration) {
  usingVersion(DEFAULT_REAL_VERSION)
      .usingNVariables(2)
      .usingVariables({"v1", "v2"})
      .usingOutputs({"i1", "test-output1"})
      .withEmptyGateList();

  EXPECT_THROW(
      quantumComputationInstance->import(realFileContent, Format::Real),
      QFRException);
}

TEST_F(RealParserTest, InvalidOutputIdentDeclarationInQuote) {
  usingVersion(DEFAULT_REAL_VERSION)
      .usingNVariables(2)
      .usingVariables({"v1", "v2"})
      .usingInputs({"\"test-output1\"", "o2"})
      .withEmptyGateList();

  EXPECT_THROW(
      quantumComputationInstance->import(realFileContent, Format::Real),
      QFRException);
}

TEST_F(RealParserTest, EmptyOutputIdentInQuotesNotAllowed) {
  usingVersion(DEFAULT_REAL_VERSION)
      .usingNVariables(2)
      .usingVariables({"v1", "v2"})
      .usingOutputs({"\"\"", "o2"})
      .withEmptyGateList();

  EXPECT_THROW(
      quantumComputationInstance->import(realFileContent, Format::Real),
      QFRException);
}

TEST_F(RealParserTest, InputIdentMatchingVariableIdentIsNotAllowed) {
  usingVersion(DEFAULT_REAL_VERSION)
      .usingNVariables(2)
      .usingVariables({"v1", "v2"})
      .usingInputs({"i1", "v2"})
      .withEmptyGateList();

  EXPECT_THROW(
      quantumComputationInstance->import(realFileContent, Format::Real),
      QFRException);
}

TEST_F(RealParserTest, OutputIdentMatchingVariableIdentIsNotAllowed) {
  usingVersion(DEFAULT_REAL_VERSION)
      .usingNVariables(2)
      .usingVariables({"v1", "v2"})
      .usingOutputs({"v1", "o2"})
      .withEmptyGateList();

  EXPECT_THROW(
      quantumComputationInstance->import(realFileContent, Format::Real),
      QFRException);
}

TEST_F(RealParserTest, DuplicateVariableIdentDeclaration) {
  usingVersion(DEFAULT_REAL_VERSION)
      .usingNVariables(2)
      .usingVariables({"v1", "v1"})
      .withEmptyGateList();

  EXPECT_THROW(
      quantumComputationInstance->import(realFileContent, Format::Real),
      QFRException);
}

TEST_F(RealParserTest, DuplicateInputIdentDeclaration) {
  usingVersion(DEFAULT_REAL_VERSION)
      .usingNVariables(2)
      .usingVariables({"v1", "v2"})
      .usingInputs({"i1", "i1"})
      .withEmptyGateList();

  EXPECT_THROW(
      quantumComputationInstance->import(realFileContent, Format::Real),
      QFRException);
}

TEST_F(RealParserTest, DuplicateOutputIdentDeclaration) {
  usingVersion(DEFAULT_REAL_VERSION)
      .usingNVariables(2)
      .usingVariables({"v1", "v2"})
      .usingOutputs({"o1", "o1"})
      .withEmptyGateList();

  EXPECT_THROW(
      quantumComputationInstance->import(realFileContent, Format::Real),
      QFRException);
}

TEST_F(RealParserTest, DuplicateVariableIdentDeclarationInInitialLayout) {
  usingVersion(DEFAULT_REAL_VERSION)
      .usingNVariables(2)
      .usingVariables({"v1", "v2"})
      .usingInitialLayout({"v1", "v1"})
      .withEmptyGateList();

  EXPECT_THROW(
      quantumComputationInstance->import(realFileContent, Format::Real),
      QFRException);
}

TEST_F(RealParserTest,
       MissingClosingQuoteInIoIdentifierDoesNotLeadToInfinityLoop) {
  usingVersion(DEFAULT_REAL_VERSION)
      .usingNVariables(2)
      .usingVariables({"v1", "v2"})
      .usingOutputs({"\"o1", "o1"})
      .withEmptyGateList();

  EXPECT_THROW(
      quantumComputationInstance->import(realFileContent, Format::Real),
      QFRException);
}

TEST_F(RealParserTest, MissingOpeningQuoteInIoIdentifierIsDetectedAsFaulty) {
  usingVersion(DEFAULT_REAL_VERSION)
      .usingNVariables(2)
      .usingVariables({"v1", "v2"})
      .usingOutputs({"o1\"", "o1"})
      .withEmptyGateList();

  EXPECT_THROW(
      quantumComputationInstance->import(realFileContent, Format::Real),
      QFRException);
}

TEST_F(RealParserTest, InvalidConstantStateValue) {
  usingVersion(DEFAULT_REAL_VERSION)
      .usingNVariables(2)
      .usingVariables({"v1", "v2"})
      .withConstants({constantValueOne, 't'})
      .withEmptyGateList();

  EXPECT_THROW(
      quantumComputationInstance->import(realFileContent, Format::Real),
      QFRException);
}

TEST_F(RealParserTest, InvalidGarbageStateValue) {
  usingVersion(DEFAULT_REAL_VERSION)
      .usingNVariables(2)
      .usingVariables({"v1", "v2"})
      .withGarbageValues({'t', isNotGarbageState});

  EXPECT_THROW(
      quantumComputationInstance->import(realFileContent, Format::Real),
      QFRException);
}

TEST_F(RealParserTest, GateWithMoreLinesThanDeclared) {
  usingVersion(DEFAULT_REAL_VERSION)
      .usingNVariables(3)
      .usingVariables({"v1", "v2", "v3"})
      .withGates({stringifyGate(GateType::Toffoli, std::optional(2),
                                {"v1", "v2"}, {"v3"})});

  EXPECT_THROW(
      quantumComputationInstance->import(realFileContent, Format::Real),
      QFRException);
}

TEST_F(RealParserTest, GateWithLessLinesThanDeclared) {
  usingVersion(DEFAULT_REAL_VERSION)
      .usingNVariables(3)
      .usingVariables({"v1", "v2", "v3"})
      .withGates(
          {stringifyGate(GateType::Toffoli, std::optional(3), {"v1"}, {"v3"})});

  EXPECT_THROW(
      quantumComputationInstance->import(realFileContent, Format::Real),
      QFRException);
}

TEST_F(RealParserTest, GateWithControlLineTargetingUnknownVariable) {
  usingVersion(DEFAULT_REAL_VERSION)
      .usingNVariables(2)
      .usingVariables({"v1", "v2"})
      .withGates({stringifyGate(GateType::Toffoli, {"v3"}, {"v2"})});

  EXPECT_THROW(
      quantumComputationInstance->import(realFileContent, Format::Real),
      QFRException);
}

TEST_F(RealParserTest, GateWithTargetLineTargetingUnknownVariable) {
  usingVersion(DEFAULT_REAL_VERSION)
      .usingNVariables(2)
      .usingVariables({"v1", "v2"})
      .withGates({stringifyGate(GateType::Toffoli, {"v1"}, {"v3"})});

  EXPECT_THROW(
      quantumComputationInstance->import(realFileContent, Format::Real),
      QFRException);
}

TEST_F(RealParserTest, UnknownVariableIdentDeclarationInInitialLayout) {
  usingVersion(DEFAULT_REAL_VERSION)
      .usingNVariables(2)
      .usingVariables({"v1", "v2"})
      .usingInitialLayout({"v4", "v1"})
      .withEmptyGateList();

  EXPECT_THROW(
      quantumComputationInstance->import(realFileContent, Format::Real),
      QFRException);
}

TEST_F(RealParserTest, DuplicateNumVarsDefinitionNotPossible) {
  usingVersion(DEFAULT_REAL_VERSION)
      .usingNVariables(2)
      .usingNVariables(3)
      .usingVariables({"v1", "v2"})
      .withEmptyGateList();

  EXPECT_THROW(
      quantumComputationInstance->import(realFileContent, Format::Real),
      QFRException);
}

TEST_F(RealParserTest, DuplicateVariablesDefinitionNotPossible) {
  usingVersion(DEFAULT_REAL_VERSION)
      .usingNVariables(2)
      .usingVariables({"v1", "v2"})
      .usingInputs({"i1", "i2"})
      .usingVariables({"v1", "v2"})
      .withEmptyGateList();

  EXPECT_THROW(
      quantumComputationInstance->import(realFileContent, Format::Real),
      QFRException);
}

TEST_F(RealParserTest, DuplicateInputsDefinitionNotPossible) {
  usingVersion(DEFAULT_REAL_VERSION)
      .usingNVariables(2)
      .usingVariables({"v1", "v2"})
      .usingInputs({"i1", "i2"})
      .withConstants({constantValueOne, constantValueNone})
      .usingOutputs({"o1", "o2"})
      .usingInputs({"i1", "i2"})
      .withEmptyGateList();

  EXPECT_THROW(
      quantumComputationInstance->import(realFileContent, Format::Real),
      QFRException);
}

TEST_F(RealParserTest, DuplicateConstantsDefinitionNotPossible) {
  usingVersion(DEFAULT_REAL_VERSION)
      .usingNVariables(2)
      .usingVariables({"v1", "v2"})
      .usingInputs({"i1", "i2"})
      .withConstants({constantValueOne, constantValueNone})
      .usingOutputs({"o1", "o2"})
      .withConstants({constantValueOne, constantValueNone})
      .withEmptyGateList();

  EXPECT_THROW(
      quantumComputationInstance->import(realFileContent, Format::Real),
      QFRException);
}

TEST_F(RealParserTest, DuplicateOutputsDefinitionNotPossible) {
  usingVersion(DEFAULT_REAL_VERSION)
      .usingNVariables(2)
      .usingVariables({"v1", "v2"})
      .usingOutputs({"o1", "o2"})
      .withGarbageValues({isGarbageState, isNotGarbageState})
      .usingOutputs({"o1", "o2"})
      .withEmptyGateList();

  EXPECT_THROW(
      quantumComputationInstance->import(realFileContent, Format::Real),
      QFRException);
}

TEST_F(RealParserTest, DuplicateGarbageDefinitionNotPossible) {
  usingVersion(DEFAULT_REAL_VERSION)
      .usingNVariables(2)
      .usingVariables({"v1", "v2"})
      .usingOutputs({"o1", "o2"})
      .withGarbageValues({isGarbageState, isNotGarbageState})
      .withGarbageValues({isGarbageState, isNotGarbageState})
      .withEmptyGateList();

  EXPECT_THROW(
      quantumComputationInstance->import(realFileContent, Format::Real),
      QFRException);
}

TEST_F(RealParserTest, DuplicateInitialLayoutDefinitionNotPossible) {
  usingVersion(DEFAULT_REAL_VERSION)
      .usingNVariables(2)
      .usingVariables({"v1", "v2"})
      .usingInitialLayout({"v2", "v1"})
      .withGarbageValues({isGarbageState, isNotGarbageState})
      .usingInitialLayout({"v2", "v1"})
      .withEmptyGateList();

  EXPECT_THROW(
      quantumComputationInstance->import(realFileContent, Format::Real),
      QFRException);
}

TEST_F(RealParserTest, HeaderWithoutNumVarsDefinitionNotPossible) {
  usingVersion(DEFAULT_REAL_VERSION)
      .usingVariables({"v1", "v2"})
      .withEmptyGateList();

  EXPECT_THROW(
      quantumComputationInstance->import(realFileContent, Format::Real),
      QFRException);
}

TEST_F(RealParserTest, HeaderWithoutVariablesDefinitionNotPossible) {
  usingVersion(DEFAULT_REAL_VERSION).usingNVariables(2).withEmptyGateList();

  EXPECT_THROW(
      quantumComputationInstance->import(realFileContent, Format::Real),
      QFRException);
}

TEST_F(RealParserTest, ContentWithoutGateListNotPossible) {
  usingVersion(DEFAULT_REAL_VERSION)
      .usingNVariables(2)
      .usingVariables({"v1", "v2"});

  EXPECT_THROW(
      quantumComputationInstance->import(realFileContent, Format::Real),
      QFRException);
}

TEST_F(RealParserTest, VariableDefinitionPriorToNumVarsDefinitionNotPossible) {
  usingVersion(DEFAULT_REAL_VERSION)
      .usingVariables({"v1", "v2"})
      .usingNVariables(2);

  EXPECT_THROW(
      quantumComputationInstance->import(realFileContent, Format::Real),
      QFRException);
}

TEST_F(RealParserTest, InputsDefinitionPrioToVariableDefinitionNotPossible) {
  usingVersion(DEFAULT_REAL_VERSION)
      .usingNVariables(2)
      .usingInputs({"i1", "i2"})
      .usingVariables({"v1", "v2"});

  EXPECT_THROW(
      quantumComputationInstance->import(realFileContent, Format::Real),
      QFRException);
}

TEST_F(RealParserTest, OutputDefinitionPriorToVariableDefinitionNotPossible) {
  usingVersion(DEFAULT_REAL_VERSION)
      .usingNVariables(2)
      .usingOutputs({"o1", "o2"})
      .usingVariables({"v1", "v2"});

  EXPECT_THROW(
      quantumComputationInstance->import(realFileContent, Format::Real),
      QFRException);
}

TEST_F(RealParserTest, ConstantsDefinitionPriorToNumVarsDefinitionNotPossible) {
  usingVersion(DEFAULT_REAL_VERSION)
      .withConstants({constantValueOne, constantValueZero})
      .usingNVariables(2)
      .usingVariables({"v1", "v2"});

  EXPECT_THROW(
      quantumComputationInstance->import(realFileContent, Format::Real),
      QFRException);
}

TEST_F(RealParserTest, GarbageDefinitionPriorToNumVarsDefinitionNotPossible) {
  usingVersion(DEFAULT_REAL_VERSION)
      .withGarbageValues({isGarbageState, isGarbageState})
      .usingNVariables(2)
      .usingVariables({"v1", "v2"});

  EXPECT_THROW(
      quantumComputationInstance->import(realFileContent, Format::Real),
      QFRException);
}

TEST_F(RealParserTest, InitialLayoutPriorToVariableDefinitionNotPossible) {
  usingVersion(DEFAULT_REAL_VERSION)
      .usingNVariables(2)
      .usingInitialLayout({"v1", "v2"})
      .usingVariables({"v1", "v2"});

  EXPECT_THROW(
      quantumComputationInstance->import(realFileContent, Format::Real),
      QFRException);
}

TEST_F(RealParserTest, OutputsDefinitionPriorToInputDefinitionNotPossible) {
  usingVersion(DEFAULT_REAL_VERSION)
      .usingNVariables(2)
      .usingVariables({"v1", "v2"})
      .usingOutputs({"i2", "i1"})
      .usingInputs({"i1", "i2"});

  EXPECT_THROW(
      quantumComputationInstance->import(realFileContent, Format::Real),
      QFRException);
}

TEST_F(RealParserTest, DuplicateControlLineInGateDefinitionNotPossible) {
  usingVersion(DEFAULT_REAL_VERSION)
      .usingNVariables(3)
      .usingVariables({"v1", "v2", "v3"})
      .withGates(
          {stringifyGate(GateType::Toffoli, {"v1", "v2", "v1"}, {"v3"})});

  EXPECT_THROW(
      quantumComputationInstance->import(realFileContent, Format::Real),
      QFRException);
}

TEST_F(RealParserTest, DuplicateTargetLineInGateDefinitionNotPossible) {
  usingVersion(DEFAULT_REAL_VERSION)
      .usingNVariables(3)
      .usingVariables({"v1", "v2", "v3"})
      .withGates({stringifyGate(GateType::V, {"v1"}, {"v2", "v3", "v2"})});

  EXPECT_THROW(
      quantumComputationInstance->import(realFileContent, Format::Real),
      QFRException);
}

TEST_F(RealParserTest, NotDefinedVariableNotUsableAsControlLine) {
  usingVersion(DEFAULT_REAL_VERSION)
      .usingNVariables(2)
      .usingVariables({"v1", "v2"})
      .withGates({stringifyGate(GateType::Toffoli, {"v1", "v3"}, {"v2"})});

  EXPECT_THROW(
      quantumComputationInstance->import(realFileContent, Format::Real),
      QFRException);
}

TEST_F(RealParserTest, NotDefinedVariablNotUsableAsTargetLine) {
  usingVersion(DEFAULT_REAL_VERSION)
      .usingNVariables(2)
      .usingVariables({"v1", "v2"})
      .withGates({stringifyGate(GateType::Toffoli, {"v1"}, {"v3"})});

  EXPECT_THROW(
      quantumComputationInstance->import(realFileContent, Format::Real),
      QFRException);
}

TEST_F(RealParserTest, GateLineNotUsableAsControlAndTargetLine) {
  usingVersion(DEFAULT_REAL_VERSION)
      .usingNVariables(2)
      .usingVariables({"v1", "v2"})
      .withGates({stringifyGate(GateType::Toffoli, {"v1"}, {"v1"})});

  EXPECT_THROW(
      quantumComputationInstance->import(realFileContent, Format::Real),
      QFRException);
}

// OK TESTS
TEST_F(RealParserTest, ConstantValueZero) {
  usingVersion(DEFAULT_REAL_VERSION)
      .usingNVariables(2)
      .usingVariables({"v1", "v2"})
      .withConstants({constantValueZero, constantValueNone})
      .withGates({stringifyGate(GateType::Toffoli, {"v1"}, {"v2"}),
                  stringifyGate(GateType::Toffoli, {"v2"}, {"v1"})});

  EXPECT_NO_THROW(
      quantumComputationInstance->import(realFileContent, Format::Real));

  ASSERT_EQ(2, quantumComputationInstance->getNqubits());
  ASSERT_EQ(1, quantumComputationInstance->getNancillae());
  ASSERT_EQ(0, quantumComputationInstance->getNgarbageQubits());
  ASSERT_THAT(quantumComputationInstance->garbage,
              testing::ElementsAre(false, false));
  ASSERT_THAT(quantumComputationInstance->ancillary,
              testing::ElementsAre(true, false));

  ASSERT_EQ(
      std::hash<Permutation>{}(getIdentityPermutation(2)),
      std::hash<Permutation>{}(quantumComputationInstance->outputPermutation));
}

TEST_F(RealParserTest, ConstantValueOne) {
  usingVersion(DEFAULT_REAL_VERSION)
      .usingNVariables(2)
      .usingVariables({"v1", "v2"})
      .withConstants({constantValueNone, constantValueOne})
      .withGates({stringifyGate(GateType::Toffoli, {"v1"}, {"v2"}),
                  stringifyGate(GateType::Toffoli, {"v2"}, {"v1"})});

  EXPECT_NO_THROW(
      quantumComputationInstance->import(realFileContent, Format::Real));

  ASSERT_EQ(2, quantumComputationInstance->getNqubits());
  ASSERT_EQ(1, quantumComputationInstance->getNancillae());
  ASSERT_EQ(0, quantumComputationInstance->getNgarbageQubits());
  ASSERT_THAT(quantumComputationInstance->garbage,
              testing::ElementsAre(false, false));
  ASSERT_THAT(quantumComputationInstance->ancillary,
              testing::ElementsAre(false, true));

  ASSERT_EQ(
      std::hash<Permutation>{}(getIdentityPermutation(2)),
      std::hash<Permutation>{}(quantumComputationInstance->outputPermutation));
}

TEST_F(RealParserTest, GarbageValues) {
  usingVersion(DEFAULT_REAL_VERSION)
      .usingNVariables(2)
      .usingVariables({"v1", "v2"})
      .withGarbageValues({isNotGarbageState, isGarbageState})
      .withGates({stringifyGate(GateType::Toffoli, {"v1"}, {"v2"}),
                  stringifyGate(GateType::Toffoli, {"v2"}, {"v1"})});

  EXPECT_NO_THROW(
      quantumComputationInstance->import(realFileContent, Format::Real));

  ASSERT_EQ(2, quantumComputationInstance->getNqubits());
  ASSERT_EQ(0, quantumComputationInstance->getNancillae());
  ASSERT_EQ(1, quantumComputationInstance->getNgarbageQubits());
  ASSERT_THAT(quantumComputationInstance->garbage,
              testing::ElementsAre(false, true));
  ASSERT_THAT(quantumComputationInstance->ancillary,
              testing::ElementsAre(false, false));

  Permutation expectedOutputPermutation;
  expectedOutputPermutation.emplace(static_cast<Qubit>(0),
                                    static_cast<Qubit>(0));

  ASSERT_EQ(
      std::hash<Permutation>{}(expectedOutputPermutation),
      std::hash<Permutation>{}(quantumComputationInstance->outputPermutation));
}

TEST_F(RealParserTest, InputIdentDeclarationInQuotes) {
  usingVersion(DEFAULT_REAL_VERSION)
      .usingNVariables(2)
      .usingVariables({"v1", "v2"})
      .usingInputs({"i1", "\"test_input_1\""})
      .withGates({stringifyGate(GateType::Toffoli, {"v1"}, {"v2"}),
                  stringifyGate(GateType::Toffoli, {"v2"}, {"v1"})});

  EXPECT_NO_THROW(
      quantumComputationInstance->import(realFileContent, Format::Real));

  ASSERT_EQ(2, quantumComputationInstance->getNqubits());
  ASSERT_EQ(0, quantumComputationInstance->getNancillae());
  ASSERT_EQ(0, quantumComputationInstance->getNgarbageQubits());
  ASSERT_THAT(quantumComputationInstance->garbage,
              testing::ElementsAre(false, false));
  ASSERT_THAT(quantumComputationInstance->ancillary,
              testing::ElementsAre(false, false));

  ASSERT_EQ(
      std::hash<Permutation>{}(getIdentityPermutation(2)),
      std::hash<Permutation>{}(quantumComputationInstance->outputPermutation));
}

TEST_F(RealParserTest, OutputIdentDeclarationInQuotes) {
  usingVersion(DEFAULT_REAL_VERSION)
      .usingNVariables(2)
      .usingVariables({"v1", "v2"})
      .usingOutputs({"\"other_output_2\"", "\"o2\""})
      .withGates({stringifyGate(GateType::Toffoli, {"v1"}, {"v2"}),
                  stringifyGate(GateType::Toffoli, {"v2"}, {"v1"})});

  EXPECT_NO_THROW(
      quantumComputationInstance->import(realFileContent, Format::Real));

  ASSERT_EQ(2, quantumComputationInstance->getNqubits());
  ASSERT_EQ(0, quantumComputationInstance->getNancillae());
  ASSERT_EQ(0, quantumComputationInstance->getNgarbageQubits());
  ASSERT_THAT(quantumComputationInstance->garbage,
              testing::ElementsAre(false, false));
  ASSERT_THAT(quantumComputationInstance->ancillary,
              testing::ElementsAre(false, false));

  ASSERT_EQ(
      std::hash<Permutation>{}(getIdentityPermutation(2)),
      std::hash<Permutation>{}(quantumComputationInstance->outputPermutation));
}

TEST_F(RealParserTest,
       InputIdentInQuotesAndMatchingOutputNotInQuotesNotConsideredEqual) {
  usingVersion(DEFAULT_REAL_VERSION)
      .usingNVariables(4)
      .usingVariables({"v1", "v2", "v3", "v4"})
      .usingInputs({"i1", "\"o2\"", "i3", "\"o4\""})
      .withGarbageValues({isNotGarbageState, isGarbageState, isNotGarbageState,
                          isGarbageState})
      .usingOutputs({"i1", "o2", "i3", "o4"})
      .withGates({
          stringifyGate(GateType::Toffoli, {"v1"}, {"v2"}),
          stringifyGate(GateType::Toffoli, {"v2"}, {"v1"}),
          stringifyGate(GateType::Toffoli, {"v3"}, {"v4"}),
          stringifyGate(GateType::Toffoli, {"v4"}, {"v3"}),
      });

  EXPECT_NO_THROW(
      quantumComputationInstance->import(realFileContent, Format::Real));

  ASSERT_EQ(4, quantumComputationInstance->getNqubits());
  ASSERT_EQ(0, quantumComputationInstance->getNancillae());
  ASSERT_EQ(2, quantumComputationInstance->getNgarbageQubits());
  ASSERT_THAT(quantumComputationInstance->garbage,
              testing::ElementsAre(false, true, false, true));
  ASSERT_THAT(quantumComputationInstance->ancillary,
              testing::ElementsAre(false, false, false, false));

  auto expectedOutputPermutation = getIdentityPermutation(4);
  expectedOutputPermutation.erase(1);
  expectedOutputPermutation.erase(3);

  ASSERT_EQ(
      std::hash<Permutation>{}(expectedOutputPermutation),
      std::hash<Permutation>{}(quantumComputationInstance->outputPermutation));
}

TEST_F(RealParserTest,
       InputIdentNotInQuotesAndMatchingOutputInQuotesNotConsideredEqual) {
  usingVersion(DEFAULT_REAL_VERSION)
      .usingNVariables(4)
      .usingVariables({"v1", "v2", "v3", "v4"})
      .usingInputs({"i1", "i2", "i3", "i4"})
      .withGarbageValues({isNotGarbageState, isGarbageState, isNotGarbageState,
                          isGarbageState})
      .usingOutputs({"i1", "\"i1\"", "i2", "\"i4\""})
      .withGates({
          stringifyGate(GateType::Toffoli, {"v1"}, {"v2"}),
          stringifyGate(GateType::Toffoli, {"v2"}, {"v1"}),
          stringifyGate(GateType::Toffoli, {"v3"}, {"v4"}),
          stringifyGate(GateType::Toffoli, {"v4"}, {"v3"}),
      });

  EXPECT_NO_THROW(
      quantumComputationInstance->import(realFileContent, Format::Real));

  ASSERT_EQ(4, quantumComputationInstance->getNqubits());
  ASSERT_EQ(0, quantumComputationInstance->getNancillae());
  ASSERT_EQ(2, quantumComputationInstance->getNgarbageQubits());
  ASSERT_THAT(quantumComputationInstance->garbage,
              testing::ElementsAre(false, false, true, true));
  ASSERT_THAT(quantumComputationInstance->ancillary,
              testing::ElementsAre(false, false, false, false));

  auto expectedOutputPermutation = getIdentityPermutation(4);
  expectedOutputPermutation.erase(1);
  expectedOutputPermutation.erase(3);
  expectedOutputPermutation[2] = static_cast<Qubit>(1);

  ASSERT_EQ(
      std::hash<Permutation>{}(expectedOutputPermutation),
      std::hash<Permutation>{}(quantumComputationInstance->outputPermutation));
}

TEST_F(RealParserTest, MatchingInputAndOutputNotInQuotes) {
  usingVersion(DEFAULT_REAL_VERSION)
      .usingNVariables(4)
      .usingVariables({"v1", "v2", "v3", "v4"})
      .usingInputs({"i1", "i2", "i3", "i4"})
      .withConstants({constantValueOne, constantValueNone, constantValueNone,
                      constantValueZero})
      .withGarbageValues({isGarbageState, isNotGarbageState, isNotGarbageState,
                          isGarbageState})
      .usingOutputs({"o1", "i1", "i4", "o2"})
      .withGates({
          stringifyGate(GateType::Toffoli, {"v1"}, {"v2"}),
          stringifyGate(GateType::Toffoli, {"v2"}, {"v1"}),
          stringifyGate(GateType::Toffoli, {"v3"}, {"v4"}),
          stringifyGate(GateType::Toffoli, {"v4"}, {"v3"}),
      });

  EXPECT_NO_THROW(
      quantumComputationInstance->import(realFileContent, Format::Real));

  ASSERT_EQ(4, quantumComputationInstance->getNqubits());
  ASSERT_EQ(2, quantumComputationInstance->getNancillae());
  ASSERT_EQ(2, quantumComputationInstance->getNgarbageQubits());
  ASSERT_THAT(quantumComputationInstance->garbage,
              testing::ElementsAre(false, true, true, false));
  ASSERT_THAT(quantumComputationInstance->ancillary,
              testing::ElementsAre(true, false, false, true));

  Permutation expectedOutputPermutation;
  expectedOutputPermutation.emplace(static_cast<Qubit>(1),
                                    static_cast<Qubit>(0));

  expectedOutputPermutation.emplace(static_cast<Qubit>(2),
                                    static_cast<Qubit>(3));

  ASSERT_EQ(
      std::hash<Permutation>{}(expectedOutputPermutation),
      std::hash<Permutation>{}(quantumComputationInstance->outputPermutation));
}

TEST_F(RealParserTest, MatchingInputAndOutputInQuotes) {
  usingVersion(DEFAULT_REAL_VERSION)
      .usingNVariables(4)
      .usingVariables({"v1", "v2", "v3", "v4"})
      .usingInputs({"i1", "\"i2\"", "\"i3\"", "i4"})
      .withConstants({constantValueNone, constantValueOne, constantValueZero,
                      constantValueNone})
      .withGarbageValues({isNotGarbageState, isNotGarbageState,
                          isNotGarbageState, isGarbageState})
      .usingOutputs({"i4", "\"i3\"", "\"i2\"", "o1"})
      .withGates({
          stringifyGate(GateType::Toffoli, {"v1"}, {"v2"}),
          stringifyGate(GateType::Toffoli, {"v2"}, {"v1"}),
          stringifyGate(GateType::Toffoli, {"v3"}, {"v4"}),
          stringifyGate(GateType::Toffoli, {"v4"}, {"v3"}),
      });

  EXPECT_NO_THROW(
      quantumComputationInstance->import(realFileContent, Format::Real));

  ASSERT_EQ(4, quantumComputationInstance->getNqubits());
  ASSERT_EQ(2, quantumComputationInstance->getNancillae());
  ASSERT_EQ(1, quantumComputationInstance->getNgarbageQubits());
  ASSERT_THAT(quantumComputationInstance->garbage,
              testing::ElementsAre(true, false, false, false));
  ASSERT_THAT(quantumComputationInstance->ancillary,
              testing::ElementsAre(false, true, true, false));

  Permutation expectedOutputPermutation;
  expectedOutputPermutation.emplace(static_cast<Qubit>(0),
                                    static_cast<Qubit>(3));

  expectedOutputPermutation.emplace(static_cast<Qubit>(1),
                                    static_cast<Qubit>(2));

  expectedOutputPermutation.emplace(static_cast<Qubit>(2),
                                    static_cast<Qubit>(1));

  ASSERT_EQ(
      std::hash<Permutation>{}(expectedOutputPermutation),
      std::hash<Permutation>{}(quantumComputationInstance->outputPermutation));
}

TEST_F(RealParserTest,
       OutputPermutationCorrectlySetBetweenMatchingInputAndOutputEntries) {
  usingVersion(DEFAULT_REAL_VERSION)
      .usingNVariables(4)
      .usingVariables({"v1", "v2", "v3", "v4"})
      .usingInputs({"i1", "i2", "i3", "i4"})
      .usingOutputs({"i4", "i3", "i2", "i1"})
      .withGates({
          stringifyGate(GateType::Toffoli, {"v1"}, {"v2"}),
          stringifyGate(GateType::Toffoli, {"v2"}, {"v1"}),
          stringifyGate(GateType::Toffoli, {"v3"}, {"v4"}),
          stringifyGate(GateType::Toffoli, {"v4"}, {"v3"}),
      });

  EXPECT_NO_THROW(
      quantumComputationInstance->import(realFileContent, Format::Real));

  ASSERT_EQ(4, quantumComputationInstance->getNqubits());
  ASSERT_EQ(0, quantumComputationInstance->getNancillae());
  ASSERT_EQ(0, quantumComputationInstance->getNgarbageQubits());
  ASSERT_THAT(quantumComputationInstance->garbage,
              testing::ElementsAre(false, false, false, false));
  ASSERT_THAT(quantumComputationInstance->ancillary,
              testing::ElementsAre(false, false, false, false));

  Permutation expectedOutputPermutation;
  expectedOutputPermutation.emplace(static_cast<Qubit>(0),
                                    static_cast<Qubit>(3));

  expectedOutputPermutation.emplace(static_cast<Qubit>(1),
                                    static_cast<Qubit>(2));

  expectedOutputPermutation.emplace(static_cast<Qubit>(2),
                                    static_cast<Qubit>(1));

  expectedOutputPermutation.emplace(static_cast<Qubit>(3),
                                    static_cast<Qubit>(0));

  ASSERT_EQ(
      std::hash<Permutation>{}(expectedOutputPermutation),
      std::hash<Permutation>{}(quantumComputationInstance->outputPermutation));
}

TEST_F(RealParserTest, OutputPermutationForGarbageQubitsNotCreated) {
  usingVersion(DEFAULT_REAL_VERSION)
      .usingNVariables(4)
      .usingVariables({"v1", "v2", "v3", "v4"})
      .usingInputs({"i1", "i2", "i3", "i4"})
      .withGarbageValues({isNotGarbageState, isGarbageState, isGarbageState,
                          isNotGarbageState})
      .usingOutputs({"i4", "o1", "o2", "i1"})
      .withGates({
          stringifyGate(GateType::Toffoli, {"v1"}, {"v2"}),
          stringifyGate(GateType::Toffoli, {"v2"}, {"v1"}),
          stringifyGate(GateType::Toffoli, {"v3"}, {"v4"}),
          stringifyGate(GateType::Toffoli, {"v4"}, {"v3"}),
      });

  EXPECT_NO_THROW(
      quantumComputationInstance->import(realFileContent, Format::Real));

  ASSERT_EQ(4, quantumComputationInstance->getNqubits());
  ASSERT_EQ(0, quantumComputationInstance->getNancillae());
  ASSERT_EQ(2, quantumComputationInstance->getNgarbageQubits());
  ASSERT_THAT(quantumComputationInstance->garbage,
              testing::ElementsAre(false, true, true, false));
  ASSERT_THAT(quantumComputationInstance->ancillary,
              testing::ElementsAre(false, false, false, false));

  Permutation expectedOutputPermutation;
  expectedOutputPermutation.emplace(static_cast<Qubit>(0),
                                    static_cast<Qubit>(3));

  expectedOutputPermutation.emplace(static_cast<Qubit>(3),
                                    static_cast<Qubit>(0));

  ASSERT_EQ(
      std::hash<Permutation>{}(expectedOutputPermutation),
      std::hash<Permutation>{}(quantumComputationInstance->outputPermutation));
}

TEST_F(RealParserTest, CheckIdentityInitialLayout) {
  usingVersion(DEFAULT_REAL_VERSION)
      .usingNVariables(2)
      .usingVariables({"v1", "v2"})
      .usingInitialLayout({"v1", "v2"})
      .withEmptyGateList();

  EXPECT_NO_THROW(
      quantumComputationInstance->import(realFileContent, Format::Real));

  ASSERT_EQ(2, quantumComputationInstance->getNqubits());

  const Permutation expectedInitialLayout = getIdentityPermutation(2);
  ASSERT_EQ(
      std::hash<Permutation>{}(expectedInitialLayout),
      std::hash<Permutation>{}(quantumComputationInstance->initialLayout));
}

TEST_F(RealParserTest, CheckNoneIdentityInitialLayout) {
  usingVersion(DEFAULT_REAL_VERSION)
      .usingNVariables(4)
      .usingVariables({"v1", "v2", "v3", "v4"})
      .usingInitialLayout({"v4", "v2", "v1", "v3"})
      .withEmptyGateList();

  EXPECT_NO_THROW(
      quantumComputationInstance->import(realFileContent, Format::Real));

  ASSERT_EQ(4, quantumComputationInstance->getNqubits());
  Permutation expectedInitialLayout;

  expectedInitialLayout.emplace(static_cast<Qubit>(0), static_cast<Qubit>(2));

  expectedInitialLayout.emplace(static_cast<Qubit>(1), static_cast<Qubit>(1));

  expectedInitialLayout.emplace(static_cast<Qubit>(2), static_cast<Qubit>(3));

  expectedInitialLayout.emplace(static_cast<Qubit>(3), static_cast<Qubit>(0));
  ASSERT_EQ(
      std::hash<Permutation>{}(expectedInitialLayout),
      std::hash<Permutation>{}(quantumComputationInstance->initialLayout));
}

TEST_F(RealParserTest, GateWithoutExplicitNumGateLinesDefinitionOk) {
  usingVersion(DEFAULT_REAL_VERSION)
      .usingNVariables(2)
      .usingVariables({"v1", "v2"})
      .withGates({stringifyGate(GateType::V, {}, {"v1", "v2"})});

  EXPECT_NO_THROW(
      quantumComputationInstance->import(realFileContent, Format::Real));

  ASSERT_EQ(2, quantumComputationInstance->getNqubits());
  ASSERT_EQ(1, quantumComputationInstance->getNops());
}
