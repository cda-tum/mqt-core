#include "Definitions.hpp"
#include "Permutation.hpp"
#include "QuantumComputation.hpp"

#include "gmock/gmock-matchers.h"
#include <functional>
#include <gtest/gtest.h>

using namespace qc;
using ::testing::NotNull;

class RealParserTest : public testing::Test {
public:
  RealParserTest& UsingVersion(double versionNumber) {
    realFileContent << REAL_HEADER_VERSION << " " << std::fixed
                    << std::setprecision(1) << versionNumber << "\n";
    return *this;
  }

  RealParserTest& UsingNVariables(std::size_t numVariables) {
    realFileContent << REAL_HEADER_NUMVARS << " "
                    << std::to_string(numVariables) << "\n";
    return *this;
  }

  RealParserTest& UsingVariables(
      const std::initializer_list<std::string_view>& variableIdents) {
    realFileContent << REAL_HEADER_VARIABLES;
    for (const auto& variableIdent : variableIdents)
      realFileContent << " " << variableIdent;

    realFileContent << "\n";
    return *this;
  }

  RealParserTest&
  UsingInputs(const std::initializer_list<std::string_view>& inputIdents) {
    realFileContent << REAL_HEADER_INPUTS;
    for (const auto& inputIdent : inputIdents)
      realFileContent << " " << inputIdent;

    realFileContent << "\n";
    return *this;
  }

  RealParserTest&
  UsingOutputs(const std::initializer_list<std::string_view>& outputIdents) {
    realFileContent << REAL_HEADER_OUTPUTS;
    for (const auto& outputIdent : outputIdents)
      realFileContent << " " << outputIdent;

    realFileContent << "\n";
    return *this;
  }

  RealParserTest&
  WithConstants(const std::initializer_list<char>& constantValuePerVariable) {
    realFileContent << REAL_HEADER_CONSTANTS << " ";
    for (const auto& constantValue : constantValuePerVariable)
      realFileContent << constantValue;

    realFileContent << "\n";
    return *this;
  }

  RealParserTest& WithGarbageValues(
      const std::initializer_list<char>& isGarbageValuePerVariable) {
    realFileContent << REAL_HEADER_GARBAGE << " ";
    for (const auto& garbageValue : isGarbageValuePerVariable)
      realFileContent << garbageValue;

    realFileContent << "\n";
    return *this;
  }

  RealParserTest& WithEmptyGateList() {
    realFileContent << REAL_HEADER_GATE_LIST_PREFIX << "\n"
                    << REAL_HEADER_GATE_LIST_POSTFIX;
    return *this;
  }

  RealParserTest& WithGates(
      const std::initializer_list<std::string_view>& stringifiedGateList) {
    if (stringifiedGateList.size() == 0)
      return WithEmptyGateList();

    realFileContent << REAL_HEADER_GATE_LIST_PREFIX << "\n";
    for (const auto& stringifiedGate : stringifiedGateList)
      realFileContent << stringifiedGate << "\n";

    realFileContent << REAL_HEADER_GATE_LIST_POSTFIX;
    return *this;
  }

protected:
  const std::string REAL_HEADER_VERSION = ".version";
  const std::string REAL_HEADER_NUMVARS = ".numvars";
  const std::string REAL_HEADER_VARIABLES = ".variables";
  const std::string REAL_HEADER_INPUTS = ".inputs";
  const std::string REAL_HEADER_OUTPUTS = ".outputs";
  const std::string REAL_HEADER_CONSTANTS = ".constants";
  const std::string REAL_HEADER_GARBAGE = ".garbage";
  const std::string REAL_HEADER_GATE_LIST_PREFIX = ".begin";
  const std::string REAL_HEADER_GATE_LIST_POSTFIX = ".end";

  static constexpr double DEFAULT_REAL_VERSION = 2.0;

  const char CONSTANT_VALUE_ZERO = '0';
  const char CONSTANT_VALUE_ONE = '1';
  const char CONSTANT_VALUE_NONE = '-';

  const char IS_GARBAGE_STATE = '1';
  const char IS_NOT_GARBAGE_STATE = '-';

  enum class GateType { Toffoli };

  std::unique_ptr<QuantumComputation> quantumComputationInstance;
  std::stringstream realFileContent;

  void SetUp() override {
    quantumComputationInstance = std::make_unique<QuantumComputation>();
    ASSERT_THAT(quantumComputationInstance, NotNull());
  }

  static Permutation GetIdentityPermutation(std::size_t nQubits) {
    auto identityPermutation = Permutation();
    for (std::size_t i = 0; i < nQubits; ++i) {
      const auto qubit = static_cast<Qubit>(i);
      identityPermutation.insert({qubit, qubit});
    }
    return identityPermutation;
  }

  static std::string stringifyGateType(const GateType gateType) {
    switch (gateType) {
    case GateType::Toffoli:
      return "t";

    default:
      throw new std::invalid_argument("Failed to stringify gate type");
    }
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
    EXPECT_TRUE(controlLines.size() >= static_cast<std::size_t>(0) &&
                targetLines.size() > static_cast<std::size_t>(0))
        << "Gate must have at least one line defined";

    std::stringstream stringifiedGateBuffer;
    if (!controlLines.size() && !optionalNumberOfGateLines.has_value())
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
  UsingVersion(DEFAULT_REAL_VERSION)
      .UsingNVariables(2)
      .UsingVariables({"v1", "v2", "v3"})
      .WithEmptyGateList();

  EXPECT_THROW(
      quantumComputationInstance->import(realFileContent, Format::Real),
      QFRException);
}

TEST_F(RealParserTest, MoreInputsThanVariablesDeclared) {
  UsingVersion(DEFAULT_REAL_VERSION)
      .UsingNVariables(2)
      .UsingVariables({"v1", "v2"})
      .UsingInputs({"i1", "i2", "i3"})
      .WithEmptyGateList();

  EXPECT_THROW(
      quantumComputationInstance->import(realFileContent, Format::Real),
      QFRException);
}

TEST_F(RealParserTest, MoreOutputsThanVariablesDeclared) {
  UsingVersion(DEFAULT_REAL_VERSION)
      .UsingNVariables(2)
      .UsingVariables({"v1", "v2"})
      .UsingOutputs({"o1", "o2", "o3"})
      .WithEmptyGateList();

  EXPECT_THROW(
      quantumComputationInstance->import(realFileContent, Format::Real),
      QFRException);
}

TEST_F(RealParserTest, MoreConstantsThanVariablesDeclared) {
  UsingVersion(DEFAULT_REAL_VERSION)
      .UsingNVariables(2)
      .UsingVariables({"v1", "v2"})
      .WithConstants(
          {CONSTANT_VALUE_ZERO, CONSTANT_VALUE_ZERO, CONSTANT_VALUE_ZERO})
      .WithEmptyGateList();

  EXPECT_THROW(
      quantumComputationInstance->import(realFileContent, Format::Real),
      QFRException);
}

TEST_F(RealParserTest, MoreGarbageEntriesThanVariablesDeclared) {
  UsingVersion(DEFAULT_REAL_VERSION)
      .UsingNVariables(2)
      .UsingVariables({"v1", "v2"})
      .WithGarbageValues(
          {IS_GARBAGE_STATE, IS_NOT_GARBAGE_STATE, IS_GARBAGE_STATE})
      .WithEmptyGateList();

  EXPECT_THROW(
      quantumComputationInstance->import(realFileContent, Format::Real),
      QFRException);
}

TEST_F(RealParserTest, LessVariablesThanNumVariablesDeclared) {
  UsingVersion(DEFAULT_REAL_VERSION)
      .UsingNVariables(2)
      .UsingVariables({"v1"})
      .WithEmptyGateList();

  EXPECT_THROW(
      quantumComputationInstance->import(realFileContent, Format::Real),
      QFRException);
}

TEST_F(RealParserTest, LessInputsThanNumVariablesDeclared) {
  UsingVersion(DEFAULT_REAL_VERSION)
      .UsingNVariables(2)
      .UsingVariables({"v1", "v2"})
      .UsingInputs({"i1"})
      .WithEmptyGateList();

  EXPECT_THROW(
      quantumComputationInstance->import(realFileContent, Format::Real),
      QFRException);
}

TEST_F(RealParserTest, LessOutputsThanNumVariablesDeclared) {
  UsingVersion(DEFAULT_REAL_VERSION)
      .UsingNVariables(2)
      .UsingVariables({"v1", "v2"})
      .UsingOutputs({"o1"})
      .WithEmptyGateList();

  EXPECT_THROW(
      quantumComputationInstance->import(realFileContent, Format::Real),
      QFRException);
}

TEST_F(RealParserTest, LessConstantsThanNumVariablesDeclared) {
  UsingVersion(DEFAULT_REAL_VERSION)
      .UsingNVariables(2)
      .UsingVariables({"v1", "v2"})
      .WithConstants({CONSTANT_VALUE_NONE})
      .WithEmptyGateList();

  EXPECT_THROW(
      quantumComputationInstance->import(realFileContent, Format::Real),
      QFRException);
}

TEST_F(RealParserTest, LessGarbageEntriesThanNumVariablesDeclared) {
  UsingVersion(DEFAULT_REAL_VERSION)
      .UsingNVariables(2)
      .UsingVariables({"v1", "v2"})
      .WithGarbageValues({IS_NOT_GARBAGE_STATE})
      .WithEmptyGateList();

  EXPECT_THROW(
      quantumComputationInstance->import(realFileContent, Format::Real),
      QFRException);
}

TEST_F(RealParserTest, InvalidVariableIdentDeclaration) {
  UsingVersion(DEFAULT_REAL_VERSION)
      .UsingNVariables(2)
      .UsingVariables({"variable-1", "v2"})
      .WithEmptyGateList();

  EXPECT_THROW(
      quantumComputationInstance->import(realFileContent, Format::Real),
      QFRException);
}

TEST_F(RealParserTest, InvalidInputIdentDeclaration) {
  UsingVersion(DEFAULT_REAL_VERSION)
      .UsingNVariables(2)
      .UsingVariables({"v1", "v2"})
      .UsingInputs({"test-input1", "i2"})
      .WithEmptyGateList();

  EXPECT_THROW(
      quantumComputationInstance->import(realFileContent, Format::Real),
      QFRException);
}

TEST_F(RealParserTest, InvalidInputIdentDeclarationInQuote) {
  UsingVersion(DEFAULT_REAL_VERSION)
      .UsingNVariables(2)
      .UsingVariables({"v1", "v2"})
      .UsingInputs({"\"test-input1\"", "i2"})
      .WithEmptyGateList();

  EXPECT_THROW(
      quantumComputationInstance->import(realFileContent, Format::Real),
      QFRException);
}

TEST_F(RealParserTest, InvalidOutputIdentDeclaration) {
  UsingVersion(DEFAULT_REAL_VERSION)
      .UsingNVariables(2)
      .UsingVariables({"v1", "v2"})
      .UsingOutputs({"i1", "test-output1"})
      .WithEmptyGateList();

  EXPECT_THROW(
      quantumComputationInstance->import(realFileContent, Format::Real),
      QFRException);
}

TEST_F(RealParserTest, InvalidOutputIdentDeclarationInQuote) {
  UsingVersion(DEFAULT_REAL_VERSION)
      .UsingNVariables(2)
      .UsingVariables({"v1", "v2"})
      .UsingInputs({"\"test-output1\"", "o2"})
      .WithEmptyGateList();

  EXPECT_THROW(
      quantumComputationInstance->import(realFileContent, Format::Real),
      QFRException);
}

TEST_F(RealParserTest, InputIdentMatchingVariableIdentIsNotAllowed) {
  UsingVersion(DEFAULT_REAL_VERSION)
      .UsingNVariables(2)
      .UsingVariables({"v1", "v2"})
      .UsingInputs({"i1", "v2"})
      .WithEmptyGateList();

  EXPECT_THROW(
      quantumComputationInstance->import(realFileContent, Format::Real),
      QFRException);
}

TEST_F(RealParserTest, OutputIdentMatchingVariableIdentIsNotAllowed) {
  UsingVersion(DEFAULT_REAL_VERSION)
      .UsingNVariables(2)
      .UsingVariables({"v1", "v2"})
      .UsingOutputs({"v1", "o2"})
      .WithEmptyGateList();

  EXPECT_THROW(
      quantumComputationInstance->import(realFileContent, Format::Real),
      QFRException);
}

TEST_F(RealParserTest, DuplicateVariableIdentDeclaration) {
  UsingVersion(DEFAULT_REAL_VERSION)
      .UsingNVariables(2)
      .UsingVariables({"v1", "v1"})
      .WithEmptyGateList();

  EXPECT_THROW(
      quantumComputationInstance->import(realFileContent, Format::Real),
      QFRException);
}

TEST_F(RealParserTest, DuplicateInputIdentDeclaration) {
  UsingVersion(DEFAULT_REAL_VERSION)
      .UsingNVariables(2)
      .UsingVariables({"v1", "v2"})
      .UsingInputs({"i1", "i1"})
      .WithEmptyGateList();

  EXPECT_THROW(
      quantumComputationInstance->import(realFileContent, Format::Real),
      QFRException);
}

TEST_F(RealParserTest, DuplicateOutputIdentDeclaration) {
  UsingVersion(DEFAULT_REAL_VERSION)
      .UsingNVariables(2)
      .UsingVariables({"v1", "v2"})
      .UsingOutputs({"o1", "o1"})
      .WithEmptyGateList();

  EXPECT_THROW(
      quantumComputationInstance->import(realFileContent, Format::Real),
      QFRException);
}

TEST_F(RealParserTest,
       MissingClosingQuoteInIoIdentifierDoesNotLeadToInfinityLoop) {
  UsingVersion(DEFAULT_REAL_VERSION)
      .UsingNVariables(2)
      .UsingVariables({"v1", "v2"})
      .UsingOutputs({"\"o1", "o1"})
      .WithEmptyGateList();

  EXPECT_THROW(
      quantumComputationInstance->import(realFileContent, Format::Real),
      QFRException);
}

TEST_F(RealParserTest, MissingOpeningQuoteInIoIdentifierIsDetectedAsFaulty) {
  UsingVersion(DEFAULT_REAL_VERSION)
      .UsingNVariables(2)
      .UsingVariables({"v1", "v2"})
      .UsingOutputs({"o1\"", "o1"})
      .WithEmptyGateList();

  EXPECT_THROW(
      quantumComputationInstance->import(realFileContent, Format::Real),
      QFRException);
}

TEST_F(RealParserTest, InvalidConstantStateValue) {
  UsingVersion(DEFAULT_REAL_VERSION)
      .UsingNVariables(2)
      .UsingVariables({"v1", "v2"})
      .WithConstants({CONSTANT_VALUE_ONE, 't'})
      .WithEmptyGateList();

  EXPECT_THROW(
      quantumComputationInstance->import(realFileContent, Format::Real),
      QFRException);
}

TEST_F(RealParserTest, InvalidGarbageStateValue) {
  UsingVersion(DEFAULT_REAL_VERSION)
      .UsingNVariables(2)
      .UsingVariables({"v1", "v2"})
      .WithGarbageValues({'t', IS_NOT_GARBAGE_STATE});

  EXPECT_THROW(
      quantumComputationInstance->import(realFileContent, Format::Real),
      QFRException);
}

TEST_F(RealParserTest, GateWithMoreLinesThanDeclared) {
  UsingVersion(DEFAULT_REAL_VERSION)
      .UsingNVariables(3)
      .UsingVariables({"v1", "v2", "v3"})
      .WithGates({stringifyGate(GateType::Toffoli, std::optional(2),
                                {"v1", "v2"}, {"v3"})});

  EXPECT_THROW(
      quantumComputationInstance->import(realFileContent, Format::Real),
      QFRException);
}

TEST_F(RealParserTest, GateWithLessLinesThanDeclared) {
  UsingVersion(DEFAULT_REAL_VERSION)
      .UsingNVariables(3)
      .UsingVariables({"v1", "v2", "v3"})
      .WithGates(
          {stringifyGate(GateType::Toffoli, std::optional(3), {"v1"}, {"v3"})});

  EXPECT_THROW(
      quantumComputationInstance->import(realFileContent, Format::Real),
      QFRException);
}

TEST_F(RealParserTest, GateWithControlLineTargetingUnknownVariable) {
  UsingVersion(DEFAULT_REAL_VERSION)
      .UsingNVariables(2)
      .UsingVariables({"v1", "v2"})
      .WithGates({stringifyGate(GateType::Toffoli, {"v3"}, {"v2"})});

  EXPECT_THROW(
      quantumComputationInstance->import(realFileContent, Format::Real),
      QFRException);
}

TEST_F(RealParserTest, GateWithTargetLineTargetingUnknownVariable) {
  UsingVersion(DEFAULT_REAL_VERSION)
      .UsingNVariables(2)
      .UsingVariables({"v1", "v2"})
      .WithGates({stringifyGate(GateType::Toffoli, {"v1"}, {"v3"})});

  EXPECT_THROW(
      quantumComputationInstance->import(realFileContent, Format::Real),
      QFRException);
}

// OK TESTS
TEST_F(RealParserTest, ConstantValueZero) {
  UsingVersion(DEFAULT_REAL_VERSION)
      .UsingNVariables(2)
      .UsingVariables({"v1", "v2"})
      .WithConstants({CONSTANT_VALUE_ZERO, CONSTANT_VALUE_NONE})
      .WithGates({stringifyGate(GateType::Toffoli, {"v1"}, {"v2"}),
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
      std::hash<Permutation>{}(GetIdentityPermutation(2)),
      std::hash<Permutation>{}(quantumComputationInstance->outputPermutation));
}

TEST_F(RealParserTest, ConstantValueOne) {
  UsingVersion(DEFAULT_REAL_VERSION)
      .UsingNVariables(2)
      .UsingVariables({"v1", "v2"})
      .WithConstants({CONSTANT_VALUE_NONE, CONSTANT_VALUE_ONE})
      .WithGates({stringifyGate(GateType::Toffoli, {"v1"}, {"v2"}),
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
      std::hash<Permutation>{}(GetIdentityPermutation(2)),
      std::hash<Permutation>{}(quantumComputationInstance->outputPermutation));
}

TEST_F(RealParserTest, GarbageValues) {
  UsingVersion(DEFAULT_REAL_VERSION)
      .UsingNVariables(2)
      .UsingVariables({"v1", "v2"})
      .WithGarbageValues({IS_NOT_GARBAGE_STATE, IS_GARBAGE_STATE})
      .WithGates({stringifyGate(GateType::Toffoli, {"v1"}, {"v2"}),
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
  UsingVersion(DEFAULT_REAL_VERSION)
      .UsingNVariables(2)
      .UsingVariables({"v1", "v2"})
      .UsingInputs({"i1", "\"test_input_1\""})
      .WithGates({stringifyGate(GateType::Toffoli, {"v1"}, {"v2"}),
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
      std::hash<Permutation>{}(GetIdentityPermutation(2)),
      std::hash<Permutation>{}(quantumComputationInstance->outputPermutation));
}

TEST_F(RealParserTest, OutputIdentDeclarationInQuotes) {
  UsingVersion(DEFAULT_REAL_VERSION)
      .UsingNVariables(2)
      .UsingVariables({"v1", "v2"})
      .UsingOutputs({"\"other_output_2\"", "\"o2\""})
      .WithGates({stringifyGate(GateType::Toffoli, {"v1"}, {"v2"}),
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
      std::hash<Permutation>{}(GetIdentityPermutation(2)),
      std::hash<Permutation>{}(quantumComputationInstance->outputPermutation));
}

TEST_F(RealParserTest,
       InputIdentInQuotesAndMatchingOutputNotInQuotesNotConsideredEqual) {
  UsingVersion(DEFAULT_REAL_VERSION)
      .UsingNVariables(4)
      .UsingVariables({"v1", "v2", "v3", "v4"})
      .UsingInputs({"i1", "\"o2\"", "i3", "\"o4\""})
      .UsingOutputs({"o1", "o2", "o3", "o4"})
      .WithGates({
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

  ASSERT_EQ(
      std::hash<Permutation>{}(GetIdentityPermutation(4)),
      std::hash<Permutation>{}(quantumComputationInstance->outputPermutation));
}

TEST_F(RealParserTest,
       InputIdentNotInQuotesAndMatchingOutputInQuotesNotConsideredEqual) {
  UsingVersion(DEFAULT_REAL_VERSION)
      .UsingNVariables(4)
      .UsingVariables({"v1", "v2", "v3", "v4"})
      .UsingInputs({"i1", "i2", "i3", "i4"})
      .UsingOutputs({"o1", "\"i1\"", "o2", "\"i4\""})
      .WithGates({
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

  ASSERT_EQ(
      std::hash<Permutation>{}(GetIdentityPermutation(4)),
      std::hash<Permutation>{}(quantumComputationInstance->outputPermutation));
}

TEST_F(RealParserTest, MatchingInputAndOutputNotInQuotes) {
  UsingVersion(DEFAULT_REAL_VERSION)
      .UsingNVariables(4)
      .UsingVariables({"v1", "v2", "v3", "v4"})
      .UsingInputs({"i1", "i2", "i3", "i4"})
      .WithConstants({CONSTANT_VALUE_ONE, CONSTANT_VALUE_NONE,
                      CONSTANT_VALUE_NONE, CONSTANT_VALUE_ZERO})
      .UsingOutputs({"o1", "i1", "i4", "o2"})
      .WithGarbageValues({IS_GARBAGE_STATE, IS_NOT_GARBAGE_STATE,
                          IS_NOT_GARBAGE_STATE, IS_GARBAGE_STATE})
      .WithGates({
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
              testing::ElementsAre(true, false, false, true));
  ASSERT_THAT(quantumComputationInstance->ancillary,
              testing::ElementsAre(true, false, false, true));

  Permutation expectedOutputPermutation;
  expectedOutputPermutation.emplace(static_cast<Qubit>(2),
                                    static_cast<Qubit>(1));

  expectedOutputPermutation.emplace(static_cast<Qubit>(3),
                                    static_cast<Qubit>(4));

  ASSERT_EQ(
      std::hash<Permutation>{}(expectedOutputPermutation),
      std::hash<Permutation>{}(quantumComputationInstance->outputPermutation));
}

TEST_F(RealParserTest, MatchingInputAndOutputInQuotes) {
  UsingVersion(DEFAULT_REAL_VERSION)
      .UsingNVariables(4)
      .UsingVariables({"v1", "v2", "v3", "v4"})
      .UsingInputs({"i1", "\"i2\"", "\"i3\"", "i4"})
      .WithConstants({CONSTANT_VALUE_NONE, CONSTANT_VALUE_ONE,
                      CONSTANT_VALUE_ZERO, CONSTANT_VALUE_NONE})
      .UsingOutputs({"i4", "\"i3\"", "\"i2\"", "o1"})
      .WithGarbageValues({IS_GARBAGE_STATE, IS_NOT_GARBAGE_STATE,
                          IS_NOT_GARBAGE_STATE, IS_GARBAGE_STATE})
      .WithGates({
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
              testing::ElementsAre(true, false, false, true));
  ASSERT_THAT(quantumComputationInstance->ancillary,
              testing::ElementsAre(false, true, true, false));

  Permutation expectedOutputPermutation;
  expectedOutputPermutation.emplace(static_cast<Qubit>(1),
                                    static_cast<Qubit>(4));

  expectedOutputPermutation.emplace(static_cast<Qubit>(2),
                                    static_cast<Qubit>(3));

  expectedOutputPermutation.emplace(static_cast<Qubit>(3),
                                    static_cast<Qubit>(2));

  ASSERT_EQ(
      std::hash<Permutation>{}(expectedOutputPermutation),
      std::hash<Permutation>{}(quantumComputationInstance->outputPermutation));
}

TEST_F(RealParserTest,
       OutputPermutationCorrectlySetBetweenMatchingInputAndOutputEntries) {
  UsingVersion(DEFAULT_REAL_VERSION)
      .UsingNVariables(4)
      .UsingVariables({"v1", "v2", "v3", "v4"})
      .UsingInputs({"i1", "i2", "i3", "i4"})
      .UsingOutputs({"\"i4\"", "i3", "\"i2\"", "i1"})
      .WithGates({
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
  expectedOutputPermutation.emplace(static_cast<Qubit>(3),
                                    static_cast<Qubit>(0));

  expectedOutputPermutation.emplace(static_cast<Qubit>(1),
                                    static_cast<Qubit>(2));

  ASSERT_EQ(
      std::hash<Permutation>{}(expectedOutputPermutation),
      std::hash<Permutation>{}(quantumComputationInstance->outputPermutation));
}

TEST_F(RealParserTest, OutputPermutationForGarbageQubitsNotCreated) {
  UsingVersion(DEFAULT_REAL_VERSION)
      .UsingNVariables(4)
      .UsingVariables({"v1", "v2", "v3", "v4"})
      .UsingInputs({"i1", "i2", "i3", "i4"})
      .UsingOutputs({"i4", "o1", "o2", "i1"})
      .WithGarbageValues({IS_GARBAGE_STATE, IS_NOT_GARBAGE_STATE,
                          IS_NOT_GARBAGE_STATE, IS_GARBAGE_STATE})
      .WithGates({
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
              testing::ElementsAre(true, false, false, true));
  ASSERT_THAT(quantumComputationInstance->ancillary,
              testing::ElementsAre(false, false, false, false));

  Permutation expectedOutputPermutation;
  expectedOutputPermutation.emplace(static_cast<Qubit>(1),
                                    static_cast<Qubit>(1));

  expectedOutputPermutation.emplace(static_cast<Qubit>(2),
                                    static_cast<Qubit>(2));

  ASSERT_EQ(
      std::hash<Permutation>{}(expectedOutputPermutation),
      std::hash<Permutation>{}(quantumComputationInstance->outputPermutation));
}
