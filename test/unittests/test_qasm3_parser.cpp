#include "Definitions.hpp"
#include "QuantumComputation.hpp"
#include "parsers/qasm3_parser/Exception.hpp"
#include "parsers/qasm3_parser/Parser.hpp"
#include "parsers/qasm3_parser/Scanner.hpp"
#include "parsers/qasm3_parser/passes/ConstEvalPass.hpp"

#include "gtest/gtest.h"
#include <cstddef>
#include <iostream>
#include <sstream>
#include <string>

using namespace qc;

class Qasm3ParserTest : public testing::TestWithParam<std::size_t> {
protected:
  void SetUp() override {}
};

TEST_F(Qasm3ParserTest, ImportQasm3) {
  std::stringstream ss{};
  const std::string testfile = "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit[3] q;\n"
                               "/* this is a comment, which can span multiple\n"
                               "\n"
                               "\n"
                               "// lines */\n"
                               "bit[3] c;\n";

  ss << testfile;
  auto qc = QuantumComputation();
  qc.import(ss, Format::OpenQASM3);

  EXPECT_EQ(qc.getNqubits(), 3);
  EXPECT_EQ(qc.getNcbits(), 3);
}

TEST_F(Qasm3ParserTest, ImportQasm3OldSyntax) {
  std::stringstream ss{};
  const std::string testfile = "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qreg q[3];\n"
                               "creg r[3];\n";

  ss << testfile;
  auto qc = QuantumComputation();
  qc.import(ss, Format::OpenQASM3);

  EXPECT_EQ(qc.getNqubits(), 3);
  EXPECT_EQ(qc.getNcbits(), 3);
}

TEST_F(Qasm3ParserTest, ImportQasm3GateDecl) {
  std::stringstream ss{};
  const std::string testfile = "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit[2] q;\n"
                               "gate my_x q1, q2 {\n"
                               "  x q1;\n"
                               "  x q2;\n"
                               "}\n"
                               "my_x q[0], q[1];\n";

  ss << testfile;
  auto qc = QuantumComputation();
  qc.import(ss, Format::OpenQASM3);

  std::stringstream out{};
  qc.dump(out, Format::OpenQASM3);

  const std::string expected = "// i 0 1\n"
                               "// o 0 1\n"
                               "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit[2] q;\n"
                               "x q[0];\n"
                               "x q[1];\n";

  EXPECT_EQ(out.str(), expected);
}

TEST_F(Qasm3ParserTest, ImportQasm3CtrlModifier) {
  std::stringstream ss{};
  const std::string testfile = "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit[3] q;\n"
                               "ctrl @ x q[0], q[1];\n"
                               "ctrl(2) @ x q[0], q[1], q[2];\n";

  ss << testfile;
  auto qc = QuantumComputation();
  qc.import(ss, Format::OpenQASM3);

  std::stringstream out{};
  qc.dump(out, Format::OpenQASM3);

  const std::string expected = "// i 0 1 2\n"
                               "// o 0 1 2\n"
                               "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit[3] q;\n"
                               "ctrl @ x q[0], q[1];\n"
                               "ctrl(2) @ x q[0], q[1], q[2];\n";

  EXPECT_EQ(out.str(), expected);
}

TEST_F(Qasm3ParserTest, ImportQasm3InvModifier) {
  std::stringstream ss{};
  const std::string testfile = "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit[1] q;\n"
                               "inv @ s q[0];\n";

  ss << testfile;
  auto qc = QuantumComputation();
  qc.import(ss, Format::OpenQASM3);

  std::stringstream out{};
  qc.dump(out, Format::OpenQASM3);

  const std::string expected = "// i 0\n"
                               "// o 0\n"
                               "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit[1] q;\n"
                               "sdg q[0];\n";

  EXPECT_EQ(out.str(), expected);
}

TEST_F(Qasm3ParserTest, ImportQasm3CompoundGate) {
  std::stringstream ss{};
  const std::string testfile = "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit q;\n"
                               "gate my_compound_gate q {\n"
                               " // comment\n"
                               "  x /* nested comment */ q;\n"
                               "  h q;\n"
                               "}\n"
                               "my_compound_gate q;";

  ss << testfile;
  auto qc = QuantumComputation();
  qc.import(ss, Format::OpenQASM3);

  std::stringstream out{};
  qc.dump(out, Format::OpenQASM3);

  const std::string expected = "// i 0\n"
                               "// o 0\n"
                               "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit[1] q;\n"
                               "x q[0];\n"
                               "h q[0];\n";

  EXPECT_EQ(out.str(), expected);
}

TEST_F(Qasm3ParserTest, ImportQasm3ControlledCompoundGate) {
  std::stringstream ss{};
  const std::string testfile = "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit[2] q;\n"
                               "gate my_compound_gate q {\n"
                               "  x q;\n"
                               "}\n"
                               "ctrl @ my_compound_gate q[0], q[1];\n";

  ss << testfile;
  auto qc = QuantumComputation();
  qc.import(ss, Format::OpenQASM3);

  std::stringstream out{};
  qc.dump(out, Format::OpenQASM3);

  const std::string expected = "// i 0 1\n"
                               "// o 0 1\n"
                               "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit[2] q;\n"
                               "ctrl @ x q[0], q[1];\n";

  EXPECT_EQ(out.str(), expected);
}

TEST_F(Qasm3ParserTest, ImportQasm3ParamCompoundGate) {
  std::stringstream ss{};
  const std::string testfile = "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit[2] q;\n"
                               "gate my_compound_gate(a) q {\n"
                               "  rz(a) q;\n"
                               "}\n"
                               "my_compound_gate(1.0 * pi) q[0];\n";

  ss << testfile;
  auto qc = QuantumComputation();
  qc.import(ss, Format::OpenQASM3);

  std::stringstream out{};
  qc.dump(out, Format::OpenQASM3);

  const std::string expected = "// i 0 1\n"
                               "// o 0 1\n"
                               "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit[2] q;\n"
                               "rz(3.14159265358979) q[0];\n";

  EXPECT_EQ(out.str(), expected);
}

TEST_F(Qasm3ParserTest, ImportQasm3Measure) {
  std::stringstream ss{};
  const std::string testfile = "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit[2] q;\n"
                               "bit r1;\n"
                               "bit[2] r2;\n"
                               "h q;\n"
                               "r1[0] = measure q[0];\n"
                               "r1 = measure q[0];\n"
                               "r2 = measure q;\n"
                               "measure q[1] -> r1;\n"
                               "measure q[1] -> r2[0];\n";

  ss << testfile;
  auto qc = QuantumComputation();
  qc.import(ss, Format::OpenQASM3);

  std::stringstream out{};
  qc.dump(out, Format::OpenQASM3);

  const std::string expected = "// i 0 1\n"
                               "// o 0 1\n"
                               "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit[2] q;\n"
                               "bit[1] r1;\n"
                               "bit[2] r2;\n"
                               "h q[0];\n"
                               "h q[1];\n"
                               "r1[0] = measure q[0];\n"
                               "r1[0] = measure q[0];\n"
                               "r2 = measure q;\n"
                               "r1[0] = measure q[1];\n"
                               "r2[0] = measure q[1];\n";

  EXPECT_EQ(out.str(), expected);
}

TEST_F(Qasm3ParserTest, ImportQasm3InitialLayout) {
  std::stringstream ss{};
  const std::string testfile = "// i 1 0\n"
                               "// o 1 0\n"
                               "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit[2] q;\n";

  ss << testfile;
  auto qc = QuantumComputation();
  qc.import(ss, Format::OpenQASM3);

  std::stringstream out{};
  qc.dump(out, Format::OpenQASM3);

  const std::string expected = "// i 1 0\n"
                               "// o 1 0\n"
                               "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit[2] q;\n";

  EXPECT_EQ(out.str(), expected);
}

TEST_F(Qasm3ParserTest, ImportQasm3ConstEval) {
  std::stringstream ss{};
  const std::string testfile =
      "OPENQASM 3.0;\n"
      "include \"stdgates.inc\";\n"
      "const uint N = (0x4 + 8 - 0b10 - (0o10 / 4)) / 2;\n"
      "qubit[N * 2] q;\n"
      "ctrl @ x q[0], q[N * 2 - 1];\n"
      "x q;";

  ss << testfile;
  auto qc = QuantumComputation();
  qc.import(ss, Format::OpenQASM3);

  std::stringstream out{};
  qc.dump(out, Format::OpenQASM3);

  const std::string expected = "// i 0 1 2 3 4 5 6 7\n"
                               "// o 0 1 2 3 4 5 6 7\n"
                               "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit[8] q;\n"
                               "bit[32] N;\n"
                               "ctrl @ x q[0], q[7];\n"
                               "x q[0];\n"
                               "x q[1];\n"
                               "x q[2];\n"
                               "x q[3];\n"
                               "x q[4];\n"
                               "x q[5];\n"
                               "x q[6];\n"
                               "x q[7];\n";

  EXPECT_EQ(out.str(), expected);
}

TEST_F(Qasm3ParserTest, ImportQasm3NonUnitary) {
  std::stringstream ss{};
  const std::string testfile = "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit[2] q1;\n"
                               "qubit[2] q2;\n"
                               "reset q1[0];\n"
                               "barrier q1, q2;\n"
                               "reset q1;\n"
                               "bit c = measure q1[0];\n";

  ss << testfile;
  auto qc = QuantumComputation();
  qc.import(ss, Format::OpenQASM3);

  std::stringstream out{};
  qc.dump(out, Format::OpenQASM3);

  const std::string expected = "// i 0 1 2 3\n"
                               "// o 0\n"
                               "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit[2] q1;\n"
                               "qubit[2] q2;\n"
                               "bit[1] c;\n"
                               "reset q1[0];\n"
                               "barrier q1[0];\n"
                               "barrier q1[1];\n"
                               "barrier q2[0];\n"
                               "barrier q2[1];\n"
                               "reset q1;\n"
                               "c[0] = measure q1[0];\n";

  EXPECT_EQ(out.str(), expected);
}

TEST_F(Qasm3ParserTest, ImportQasm3IfStatement) {
  std::stringstream ss{};
  const std::string testfile = "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit[2] q;\n"
                               "h q[0];\n"
                               "bit c = measure q[0];\n"
                               "if (c == 1) {\n"
                               "  x q[1];\n"
                               "}";

  ss << testfile;
  auto qc = QuantumComputation();
  qc.import(ss, Format::OpenQASM3);

  std::stringstream out{};
  qc.dump(out, Format::OpenQASM3);

  const std::string expected = "// i 0 1\n"
                               "// o 0\n"
                               "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit[2] q;\n"
                               "bit[1] c;\n"
                               "h q[0];\n"
                               "c[0] = measure q[0];\n"
                               "if (c == 1) {\n"
                               "  x q[1];\n"
                               "}\n"
                               "";

  EXPECT_EQ(out.str(), expected);
}

TEST_F(Qasm3ParserTest, ImportQasm3ImplicitInclude) {
  std::stringstream ss{};
  const std::string testfile = "qubit q;\n"
                               "h q[0];\n"
                               "";

  ss << testfile;
  auto qc = QuantumComputation();
  qc.import(ss, Format::OpenQASM3);

  std::stringstream out{};
  qc.dump(out, Format::OpenQASM3);

  const std::string expected = "// i 0\n"
                               "// o 0\n"
                               "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit[1] q;\n"
                               "h q[0];\n"
                               "";

  EXPECT_EQ(out.str(), expected);
}

TEST_F(Qasm3ParserTest, ImportQasm3Qelib1) {
  std::stringstream ss{};
  const std::string testfile = "OPENQASM 2.0;\n"
                               "include \"qelib1.inc\";\n"
                               "qubit q;\n"
                               "h q[0];\n"
                               "";

  ss << testfile;
  auto qc = QuantumComputation();
  qc.import(ss, Format::OpenQASM3);

  std::stringstream out{};
  qc.dump(out, Format::OpenQASM3);

  const std::string expected = "// i 0\n"
                               "// o 0\n"
                               "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit[1] q;\n"
                               "h q[0];\n"
                               "";

  EXPECT_EQ(out.str(), expected);
}

TEST_F(Qasm3ParserTest, ImportQasm3Teleportation) {
  std::stringstream ss{};
  const std::string testfile = "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "opaque teleport src, anc, tgt;\n"
                               "qubit[3] q;\n"
                               "teleport q[0], q[1], q[2];\n"
                               "";

  ss << testfile;
  auto qc = QuantumComputation();
  qc.import(ss, Format::OpenQASM3);

  std::stringstream out{};
  qc.dump(out, Format::OpenQASM3);

  const std::string expected =
      "// i 0 1 2\n"
      "// o 0 1 2\n"
      "OPENQASM 3.0;\n"
      "include \"stdgates.inc\";\n"
      "opaque teleport src, anc, tgt;\n"
      "qubit[3] q;\n"
      "// teleport q_0, a_0, a_1; q_0 --> a_1  via a_0\n"
      "teleport q[0], q[1], q[2];\n"
      "";

  EXPECT_EQ(out.str(), expected);
}

TEST_F(Qasm3ParserTest, ImportQasm3NestedGates) {
  std::stringstream ss{};
  const std::string testfile = "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "gate my_x q { x q; }\n"
                               "gate my_x2 q1 { x q1; }\n"
                               "qubit[1] q;\n"
                               "my_x2 q[0];\n"
                               "";

  ss << testfile;
  auto qc = QuantumComputation();
  qc.import(ss, Format::OpenQASM3);

  EXPECT_EQ(qc.getNops(), 1);
  EXPECT_EQ(qc.at(0)->getType(), OpType::X);
}

TEST_F(Qasm3ParserTest, ImportQasm3AlternatingControl) {
  std::stringstream ss{};
  const std::string testfile = "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit[7] q;\n"
                               "ctrl @ negctrl(2) @ negctrl @ ctrl @ ctrl @ x "
                               "q[0], q[1], q[2], q[3], q[4], q[5], q[6];\n"
                               "";

  ss << testfile;
  auto qc = QuantumComputation();
  qc.import(ss, Format::OpenQASM3);

  std::stringstream out{};
  qc.dump(out, Format::OpenQASM3);

  const std::string expected = "// i 0 1 2 3 4 5 6\n"
                               "// o 0 1 2 3 4 5 6\n"
                               "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit[7] q;\n"
                               "ctrl @ negctrl(3) @ ctrl(2) @ x q[0], q[1], "
                               "q[2], q[3], q[4], q[5], q[6];\n"
                               "";

  EXPECT_EQ(out.str(), expected);
}

TEST_F(Qasm3ParserTest, ImportQasmConstEval) {
  std::stringstream ss{};
  const std::string testfile = "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "const uint N_1 = 0xa;\n"
                               "const uint N_2 = 8;\n"
                               "qubit[N_1 - N_2] q;\n"
                               "";

  ss << testfile;
  auto qc = QuantumComputation();
  qc.import(ss, Format::OpenQASM3);

  std::stringstream out{};
  qc.dump(out, Format::OpenQASM3);

  const std::string expected = "// i 0 1\n"
                               "// o 0 1\n"
                               "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit[2] q;\n"
                               "bit[32] N_1;\n"
                               "bit[32] N_2;\n"
                               "";

  EXPECT_EQ(out.str(), expected);
}

TEST_F(Qasm3ParserTest, ImportQasmBroadcasting) {
  std::stringstream ss{};
  const std::string testfile = "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit[2] q1;\n"
                               "qubit[2] q2;\n"
                               "h q1;\n"
                               "reset q2;\n"
                               "cx q1, q2;\n";

  ss << testfile;
  auto qc = QuantumComputation();
  qc.import(ss, Format::OpenQASM3);

  std::stringstream out{};
  qc.dump(out, Format::OpenQASM3);

  const std::string expected = "// i 0 1 2 3\n"
                               "// o 0 1 2 3\n"
                               "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit[2] q1;\n"
                               "qubit[2] q2;\n"
                               "h q1[0];\n"
                               "h q1[1];\n"
                               "reset q2;\n"
                               "ctrl @ x q1[0], q2[0];\n"
                               "ctrl @ x q1[1], q2[1];\n"
                               "";

  EXPECT_EQ(out.str(), expected);
}

TEST_F(Qasm3ParserTest, ImportQasmComparison) {
  std::stringstream ss{};
  const std::string testfile = "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit[2] q;\n"
                               "bit c1;\n"
                               "h q;\n"
                               "c1 = measure q[0];\n"
                               "if (c1 < 0) { x q[0]; }\n"
                               "if (c1 <= 0) { x q[0]; }\n"
                               "if (c1 > 0) { x q[0]; }\n"
                               "if (c1 >= 0) { x q[0]; }\n"
                               "if (c1 == 0) { x q[0]; }\n"
                               "if (c1 != 0) { x q[0]; }\n";

  ss << testfile;
  auto qc = QuantumComputation();
  qc.import(ss, Format::OpenQASM3);

  std::stringstream out{};
  qc.dump(out, Format::OpenQASM3);

  const std::string expected = "// i 0 1\n"
                               "// o 0\n"
                               "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit[2] q;\n"
                               "bit[1] c1;\n"
                               "h q[0];\n"
                               "h q[1];\n"
                               "c1[0] = measure q[0];\n"
                               "if (c1 < 0) {\n"
                               "  x q[0];\n"
                               "}\n"
                               "if (c1 <= 0) {\n"
                               "  x q[0];\n"
                               "}\n"
                               "if (c1 > 0) {\n"
                               "  x q[0];\n"
                               "}\n"
                               "if (c1 >= 0) {\n"
                               "  x q[0];\n"
                               "}\n"
                               "if (c1 == 0) {\n"
                               "  x q[0];\n"
                               "}\n"
                               "if (c1 != 0) {\n"
                               "  x q[0];\n"
                               "}\n"
                               "";

  EXPECT_EQ(out.str(), expected);
}

TEST_F(Qasm3ParserTest, ImportQasmScanner) {
  std::stringstream ss{};
  const std::string testfile =
      "$1 : . .5 -1. -= += ++ *= **= ** /= % %= |= || | &= "
      "&& & ^= ^ ~= ~ ! <= <<= << < >= >>= >> >";
  const auto tokens = std::vector{
      qasm3::Token::Kind::HardwareQubit,
      qasm3::Token::Kind::Colon,
      qasm3::Token::Kind::Dot,
      qasm3::Token::Kind::FloatLiteral,
      qasm3::Token::Kind::FloatLiteral,
      qasm3::Token::Kind::MinusEquals,
      qasm3::Token::Kind::PlusEquals,
      qasm3::Token::Kind::DoublePlus,
      qasm3::Token::Kind::AsteriskEquals,
      qasm3::Token::Kind::DoubleAsteriskEquals,
      qasm3::Token::Kind::DoubleAsterisk,
      qasm3::Token::Kind::SlashEquals,
      qasm3::Token::Kind::Percent,
      qasm3::Token::Kind::PercentEquals,
      qasm3::Token::Kind::PipeEquals,
      qasm3::Token::Kind::DoublePipe,
      qasm3::Token::Kind::Pipe,
      qasm3::Token::Kind::AmpersandEquals,
      qasm3::Token::Kind::DoubleAmpersand,
      qasm3::Token::Kind::Ampersand,
      qasm3::Token::Kind::CaretEquals,
      qasm3::Token::Kind::Caret,
      qasm3::Token::Kind::TildeEquals,
      qasm3::Token::Kind::Tilde,
      qasm3::Token::Kind::ExclamationPoint,
      qasm3::Token::Kind::LessThanEquals,
      qasm3::Token::Kind::LeftShitEquals,
      qasm3::Token::Kind::LeftShift,
      qasm3::Token::Kind::LessThan,
      qasm3::Token::Kind::GreaterThanEquals,
      qasm3::Token::Kind::RightShiftEquals,
      qasm3::Token::Kind::RightShift,
      qasm3::Token::Kind::GreaterThan,
      qasm3::Token::Kind::Eof,
  };

  ss << testfile;
  qasm3::Scanner scanner(&ss);

  for (const auto& expected : tokens) {
    auto token = scanner.next();
    EXPECT_EQ(token.kind, expected);
  }
}

TEST_F(Qasm3ParserTest, ImportQasmParseOperators) {
  std::stringstream ss{};
  const std::string testfile = "x += 1;\n"
                               "x -= 1;\n"
                               "x *= 1;\n"
                               "x /= 1;\n"
                               "x &= 1;\n"
                               "x |= 1;\n"
                               "x ~= 1;\n"
                               "x ^= 1;\n"
                               "x <<= 1;\n"
                               "x >>= 1;\n"
                               "x %= 1;\n"
                               "x **= 1;\n"
                               "";

  ss << testfile;
  qasm3::Parser parser(&ss, false);

  const auto expectedTypes = std::vector{
      qasm3::AssignmentStatement::Type::PlusAssignment,
      qasm3::AssignmentStatement::Type::MinusAssignment,
      qasm3::AssignmentStatement::Type::TimesAssignment,
      qasm3::AssignmentStatement::Type::DivAssignment,
      qasm3::AssignmentStatement::Type::BitwiseAndAssignment,
      qasm3::AssignmentStatement::Type::BitwiseOrAssignment,
      qasm3::AssignmentStatement::Type::BitwiseNotAssignment,
      qasm3::AssignmentStatement::Type::BitwiseXorAssignment,
      qasm3::AssignmentStatement::Type::LeftShiftAssignment,
      qasm3::AssignmentStatement::Type::RightShiftAssignment,
      qasm3::AssignmentStatement::Type::ModuloAssignment,
      qasm3::AssignmentStatement::Type::PowerAssignment,
  };

  for (const auto& expected : expectedTypes) {
    const auto stmt = parser.parseAssignmentStatement();
    EXPECT_EQ(stmt->type, expected);
  }
}

TEST_F(Qasm3ParserTest, ImportQasmParseUnaryExpressions) {
  std::stringstream ss{};
  const std::string testfile = "sin(x)\n"
                               "cos(x)\n"
                               "tan(x)\n"
                               "exp(x)\n"
                               "ln(x)\n"
                               "sqrt(x)\n"
                               "";

  ss << testfile;
  qasm3::Parser parser(&ss, false);

  const auto expectedTypes = std::vector{
      qasm3::UnaryExpression::Op::Sin, qasm3::UnaryExpression::Op::Cos,
      qasm3::UnaryExpression::Op::Tan, qasm3::UnaryExpression::Op::Exp,
      qasm3::UnaryExpression::Op::Ln,  qasm3::UnaryExpression::Op::Sqrt,
  };

  for (const auto& expected : expectedTypes) {
    const auto expr = parser.parseExpression();
    const auto unaryExpr =
        std::dynamic_pointer_cast<qasm3::UnaryExpression>(expr);
    EXPECT_NE(unaryExpr, nullptr);
    EXPECT_EQ(unaryExpr->op, expected);
  }
}

TEST_F(Qasm3ParserTest, ImportQasmParseBinaryExpressions) {
  std::stringstream ss{};
  const std::string testfile = "x^5\n"
                               "x == 5\n"
                               "x != 5\n"
                               "x <= 5\n"
                               "x < 5\n"
                               "x >= 5\n"
                               "x > 5\n"
                               "";

  ss << testfile;
  qasm3::Parser parser(&ss, false);

  const auto expectedTypes = std::vector{
      qasm3::BinaryExpression::Op::Power,
      qasm3::BinaryExpression::Op::Equal,
      qasm3::BinaryExpression::Op::NotEqual,
      qasm3::BinaryExpression::Op::LessThanOrEqual,
      qasm3::BinaryExpression::Op::LessThan,
      qasm3::BinaryExpression::Op::GreaterThanOrEqual,
      qasm3::BinaryExpression::Op::GreaterThan,
  };

  for (const auto& expected : expectedTypes) {
    const auto expr = parser.parseExpression();
    const auto binaryExpr =
        std::dynamic_pointer_cast<qasm3::BinaryExpression>(expr);
    EXPECT_NE(binaryExpr, nullptr);
    EXPECT_EQ(binaryExpr->op, expected);
  }
}

TEST_F(Qasm3ParserTest, ImportQasmUnknownQreg) {
  std::stringstream ss{};
  const std::string testfile = "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "x q;\n"
                               "";

  ss << testfile;
  auto qc = QuantumComputation();
  EXPECT_THROW(
      {
        try {
          qc.import(ss, Format::OpenQASM3);
        } catch (const qasm3::CompilerError& e) {
          EXPECT_EQ(e.message, "Usage of unknown quantum register.");
          throw;
        }
      },
      qasm3::CompilerError);
}

TEST_F(Qasm3ParserTest, ImportQasmIndexOutOfBounds) {
  std::stringstream ss{};
  const std::string testfile = "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit[2] q;\n"
                               "x q[2];\n"
                               "";

  ss << testfile;
  auto qc = QuantumComputation();
  EXPECT_THROW(
      {
        try {
          qc.import(ss, Format::OpenQASM3);
        } catch (const qasm3::CompilerError& e) {
          EXPECT_EQ(e.message, "Index expression must be smaller than the "
                               "width of the quantum register.");
          throw;
        }
      },
      qasm3::CompilerError);
}

TEST_F(Qasm3ParserTest, ImportQasmIndexOutOfBoundsClassical) {
  std::stringstream ss{};
  const std::string testfile = "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit[2] q;\n"
                               "bit[2] c;\n"
                               "c[2] = measure q[0];\n"
                               "";

  ss << testfile;
  auto qc = QuantumComputation();
  EXPECT_THROW(
      {
        try {
          qc.import(ss, Format::OpenQASM3);
        } catch (const qasm3::CompilerError& e) {
          EXPECT_EQ(e.message, "Index expression must be smaller than the "
                               "width of the classical register.");
          throw;
        }
      },
      qasm3::CompilerError);
}

TEST_F(Qasm3ParserTest, ImportQasmDuplicateDeclaration) {
  std::stringstream ss{};
  const std::string testfile = "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit[2] q;\n"
                               "qubit[2] q;\n"
                               "";

  ss << testfile;
  auto qc = QuantumComputation();
  EXPECT_THROW(
      {
        try {
          qc.import(ss, Format::OpenQASM3);
        } catch (const qasm3::CompilerError& e) {
          EXPECT_EQ(e.message, "Identifier 'q' already declared.");
          throw;
        }
      },
      qasm3::CompilerError);
}

TEST_F(Qasm3ParserTest, ImportQasmInitConstRegWithMeasure) {
  std::stringstream ss{};
  const std::string testfile = "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit q;\n"
                               "const bit c = measure q;\n"
                               "";

  ss << testfile;
  auto qc = QuantumComputation();
  EXPECT_THROW(
      {
        try {
          qc.import(ss, Format::OpenQASM3);
        } catch (const qasm3::CompilerError& e) {
          EXPECT_EQ(e.message, "Constant Evaluation: Constant declaration "
                               "initialization expression must be const.");
          throw;
        }
      },
      qasm3::CompilerError);
}

TEST_F(Qasm3ParserTest, ImportQasmAssignmentUnknownIdentifier) {
  std::stringstream ss{};
  const std::string testfile = "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit q;\n"
                               "c = measure q;\n"
                               "";

  ss << testfile;
  auto qc = QuantumComputation();
  EXPECT_THROW(
      {
        try {
          qc.import(ss, Format::OpenQASM3);
        } catch (const qasm3::CompilerError& e) {
          EXPECT_EQ(e.message, "Type check failed.");
          throw;
        }
      },
      qasm3::CompilerError);
}

TEST_F(Qasm3ParserTest, ImportQasmAssignmentConstVar) {
  std::stringstream ss{};
  const std::string testfile = "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit q;\n"
                               "const bit c = 0;\n"
                               "c = measure q;\n"
                               "";

  ss << testfile;
  auto qc = QuantumComputation();
  EXPECT_THROW(
      {
        try {
          qc.import(ss, Format::OpenQASM3);
        } catch (const qasm3::CompilerError& e) {
          EXPECT_EQ(e.message, "Type check failed.");
          throw;
        }
      },
      qasm3::CompilerError);
}

TEST_F(Qasm3ParserTest, ImportQasmMultipleInputPermutations) {
  std::stringstream ss{};
  const std::string testfile = "// i 0\n"
                               "// i 0\n"
                               "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit q;";

  ss << testfile;
  auto qc = QuantumComputation();
  EXPECT_THROW(
      {
        try {
          qc.import(ss, Format::OpenQASM3);
        } catch (const qasm3::CompilerError& e) {
          EXPECT_EQ(e.message, "Multiple initial layout specifications found.");
          throw;
        }
      },
      qasm3::CompilerError);
}

TEST_F(Qasm3ParserTest, ImportQasmMultipleOutputPermutations) {
  std::stringstream ss{};
  const std::string testfile = "// o 0\n"
                               "// o 0\n"
                               "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit q;";

  ss << testfile;
  auto qc = QuantumComputation();
  EXPECT_THROW(
      {
        try {
          qc.import(ss, Format::OpenQASM3);
        } catch (const qasm3::CompilerError& e) {
          EXPECT_EQ(e.message,
                    "Multiple output permutation specifications found.");
          throw;
        }
      },
      qasm3::CompilerError);
}

TEST_F(Qasm3ParserTest, ImportQasmInvalidOpaqueGate) {
  std::stringstream ss{};
  const std::string testfile = "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "opaque asdf q;";

  ss << testfile;
  auto qc = QuantumComputation();
  EXPECT_THROW(
      {
        try {
          qc.import(ss, Format::OpenQASM3);
        } catch (const qasm3::CompilerError& e) {
          EXPECT_EQ(e.message, "Unsupported opaque gate 'asdf'.");
          throw;
        }
      },
      qasm3::CompilerError);
}

TEST_F(Qasm3ParserTest, ImportQasmDuplicateGateDecl) {
  std::stringstream ss{};
  const std::string testfile = "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "gate my_x q { x q; }\n"
                               "gate my_x q { x q; }\n";

  ss << testfile;
  auto qc = QuantumComputation();
  EXPECT_THROW(
      {
        try {
          qc.import(ss, Format::OpenQASM3);
        } catch (const qasm3::CompilerError& e) {
          EXPECT_EQ(e.message, "Gate 'my_x' already declared.");
          throw;
        }
      },
      qasm3::CompilerError);
}

TEST_F(Qasm3ParserTest, ImportQasmDuplicateQubitArgGate) {
  std::stringstream ss{};
  const std::string testfile = "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "gate my_gate q, q { }\n";

  ss << testfile;
  auto qc = QuantumComputation();
  EXPECT_THROW(
      {
        try {
          qc.import(ss, Format::OpenQASM3);
        } catch (const qasm3::CompilerError& e) {
          EXPECT_EQ(e.message, "Qubit 'q' already declared.");
          throw;
        }
      },
      qasm3::CompilerError);
}

TEST_F(Qasm3ParserTest, ImportQasmUndeclaredGate) {
  std::stringstream ss{};
  const std::string testfile = "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit q;\n"
                               "my_x q;";

  ss << testfile;
  auto qc = QuantumComputation();
  EXPECT_THROW(
      {
        try {
          qc.import(ss, Format::OpenQASM3);
        } catch (const qasm3::CompilerError& e) {
          EXPECT_EQ(e.message, "Usage of unknown gate 'my_x'.");
          throw;
        }
      },
      qasm3::CompilerError);
}

TEST_F(Qasm3ParserTest, ImportQasmInvalidGateTargets) {
  std::stringstream ss{};
  const std::string testfile = "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit[2] q;\n"
                               "gate my_x q { x q; }\n"
                               "my_x q[0], q[1];\n";

  ss << testfile;
  auto qc = QuantumComputation();
  EXPECT_THROW(
      {
        try {
          qc.import(ss, Format::OpenQASM3);
        } catch (const qasm3::CompilerError& e) {
          EXPECT_EQ(e.message,
                    "Gate 'my_x' takes 1 targets, but 2 were supplied.");
          throw;
        }
      },
      qasm3::CompilerError);
}

TEST_F(Qasm3ParserTest, ImportQasmInvalidGateControls) {
  std::stringstream ss{};
  const std::string testfile = "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit[3] q;\n"
                               "cx q[0];\n";

  ss << testfile;
  auto qc = QuantumComputation();
  EXPECT_THROW(
      {
        try {
          qc.import(ss, Format::OpenQASM3);
        } catch (const qasm3::CompilerError& e) {
          EXPECT_EQ(e.message,
                    "Gate 'cx' takes 1 targets, but 0 were supplied.");
          throw;
        }
      },
      qasm3::CompilerError);
}

TEST_F(Qasm3ParserTest, ImportQasmInvalidGateModifiers) {
  std::stringstream ss{};
  const std::string testfile = "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit[2] q;\n"
                               "ctrl(2) @ x q[0];\n";

  ss << testfile;
  auto qc = QuantumComputation();
  EXPECT_THROW(
      {
        try {
          qc.import(ss, Format::OpenQASM3);
        } catch (const qasm3::CompilerError& e) {
          EXPECT_EQ(e.message,
                    "Gate 'x' takes 2 controls, but only 1 were supplied.");
          throw;
        }
      },
      qasm3::CompilerError);
}

TEST_F(Qasm3ParserTest, ImportQasmGateCallNonConst) {
  std::stringstream ss{};
  const std::string testfile = "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit[2] q;\n"
                               "bit[2] c = measure q;\n"
                               "rz(c) q[0];\n";

  ss << testfile;
  auto qc = QuantumComputation();
  EXPECT_THROW(
      {
        try {
          qc.import(ss, Format::OpenQASM3);
        } catch (const qasm3::CompilerError& e) {
          EXPECT_EQ(e.message, "Only const expressions are supported as gate "
                               "parameters, but found 'IdentifierExpr (c)'.");
          throw;
        }
      },
      qasm3::CompilerError);
}

TEST_F(Qasm3ParserTest, ImportQasmGateCallBroadcastingInvalidWidth) {
  std::stringstream ss{};
  const std::string testfile = "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit[2] q1;\n"
                               "qubit[3] q2;\n"
                               "cx q1, q2;\n";

  ss << testfile;
  auto qc = QuantumComputation();
  EXPECT_THROW(
      {
        try {
          qc.import(ss, Format::OpenQASM3);
        } catch (const qasm3::CompilerError& e) {
          EXPECT_EQ(
              e.message,
              "When broadcasting, all registers must be of the same width.");
          throw;
        }
      },
      qasm3::CompilerError);
}

TEST_F(Qasm3ParserTest, ImportQasmGateCallIndexingGateBody) {
  std::stringstream ss{};
  const std::string testfile = "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "gate my_x q { x q[0]; }\n"
                               "qubit q;\n"
                               "my_x q;";

  ss << testfile;
  auto qc = QuantumComputation();
  EXPECT_THROW(
      {
        try {
          qc.import(ss, Format::OpenQASM3);
        } catch (const qasm3::CompilerError& e) {
          EXPECT_EQ(e.message,
                    "Gate arguments cannot be indexed within gate body.");
          throw;
        }
      },
      qasm3::CompilerError);
}

TEST_F(Qasm3ParserTest, ImportQasmGateMeasureInvalidSizes) {
  std::stringstream ss{};
  const std::string testfile = "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit[2] q;\n"
                               "bit[3] c = measure q;";

  ss << testfile;
  auto qc = QuantumComputation();
  EXPECT_THROW(
      {
        try {
          qc.import(ss, Format::OpenQASM3);
        } catch (const qasm3::CompilerError& e) {
          EXPECT_EQ(e.message,
                    "Classical and quantum register must have the same width "
                    "in measure statement. Classical register 'c' has 3 bits, "
                    "but quantum register 'q' has 2 qubits.");
          throw;
        }
      },
      qasm3::CompilerError);
}

TEST_F(Qasm3ParserTest, ImportQasmGateOldStyleDesignator) {
  std::stringstream ss{};
  const std::string testfile = "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit q[2];\n"
                               "";

  ss << testfile;
  auto qc = QuantumComputation();
  EXPECT_THROW(
      {
        try {
          qc.import(ss, Format::OpenQASM3);
        } catch (const qasm3::CompilerError& e) {
          EXPECT_EQ(e.message, "In OpenQASM 3.0, the designator has been "
                               "changed to `type[designator] identifier;`");
          throw;
        }
      },
      qasm3::CompilerError);
}

TEST_F(Qasm3ParserTest, ImportQasmGateExpectStatement) {
  std::stringstream ss{};
  const std::string testfile = "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "+\n"
                               "";

  ss << testfile;
  auto qc = QuantumComputation();
  EXPECT_THROW(
      {
        try {
          qc.import(ss, Format::OpenQASM3);
        } catch (const qasm3::CompilerError& e) {
          EXPECT_EQ(e.message, "Expected statement, got '+'.");
          throw;
        }
      },
      qasm3::CompilerError);
}

TEST_F(Qasm3ParserTest, ImportQasmGateVersionDeclaration) {
  std::stringstream ss{};
  const std::string testfile = "qubit q;\n"
                               "OPENQASM 3.0;\n"
                               "";

  ss << testfile;
  auto qc = QuantumComputation();
  EXPECT_THROW(
      {
        try {
          qc.import(ss, Format::OpenQASM3);
        } catch (const qasm3::CompilerError& e) {
          EXPECT_EQ(
              e.message,
              "Version declaration must be at the beginning of the file.");
          throw;
        }
      },
      qasm3::CompilerError);
}

TEST_F(Qasm3ParserTest, ImportQasmInvalidExpected) {
  std::stringstream ss{};
  const std::string testfile = "OPENQASM 3.0;\n"
                               "qubit[2] q;\n"
                               "cx q[0] q[1];"; // missing comma

  ss << testfile;
  auto qc = QuantumComputation();
  EXPECT_THROW(
      {
        try {
          qc.import(ss, Format::OpenQASM3);
        } catch (const qasm3::CompilerError& e) {
          EXPECT_EQ(e.message, "Expected ',', got 'Identifier'.");
          throw;
        }
      },
      qasm3::CompilerError);
}

TEST_F(Qasm3ParserTest, ImportQasmTypeMismatchAssignment) {
  std::stringstream ss{};
  const std::string testfile = "OPENQASM 3.0;\n"
                               "bit x;\n"
                               "x = 10;";

  ss << testfile;
  auto qc = QuantumComputation();
  EXPECT_THROW(
      {
        try {
          qc.import(ss, Format::OpenQASM3);
        } catch (const qasm3::CompilerError& e) {
          EXPECT_EQ(e.message, "Type check failed.");
          throw;
        }
      },
      qasm3::CompilerError);
}

TEST_F(Qasm3ParserTest, ImportQasmTypeMismatchBinaryExpr) {
  std::stringstream ss{};
  const std::string testfile = "OPENQASM 3.0;\n"
                               "const bit x = 0;\n"
                               "const int y = 10 + x;\n";

  ss << testfile;
  auto qc = QuantumComputation();
  EXPECT_THROW(
      {
        try {
          qc.import(ss, Format::OpenQASM3);
        } catch (const qasm3::CompilerError& e) {
          EXPECT_EQ(e.message, "Type check failed.");
          throw;
        }
      },
      qasm3::CompilerError);
}

TEST_F(Qasm3ParserTest, ImportQasmConstNotInitialized) {
  std::stringstream ss{};
  const std::string testfile = "OPENQASM 3.0;\n"
                               "const bit x;\n";

  ss << testfile;
  auto qc = QuantumComputation();
  EXPECT_THROW(
      {
        try {
          qc.import(ss, Format::OpenQASM3);
        } catch (const qasm3::CompilerError& e) {
          EXPECT_EQ(e.message,
                    "Constant Evaluation: Constant declaration initialization "
                    "expression must be initialized.");
          throw;
        }
      },
      qasm3::CompilerError);
}

TEST_F(Qasm3ParserTest, ImportQasmUnaryTypeMismatchLogicalNot) {
  std::stringstream ss{};
  const std::string testfile = "OPENQASM 3.0;\n"
                               "int x = !0;\n";

  ss << testfile;
  auto qc = QuantumComputation();
  EXPECT_THROW(
      {
        try {
          qc.import(ss, Format::OpenQASM3);
        } catch (const qasm3::CompilerError& e) {
          EXPECT_EQ(e.message, "Type check failed.");
          throw;
        }
      },
      qasm3::CompilerError);
}

TEST_F(Qasm3ParserTest, ImportQasmUnaryTypeMismatchBitwiseNot) {
  std::stringstream ss{};
  const std::string testfile = "OPENQASM 3.0;\n"
                               "bool x = ~false;\n"
                               "bool y = !true;\n";

  ss << testfile;
  auto qc = QuantumComputation();
  EXPECT_THROW(
      {
        try {
          qc.import(ss, Format::OpenQASM3);
        } catch (const qasm3::CompilerError& e) {
          EXPECT_EQ(e.message, "Type check failed.");
          throw;
        }
      },
      qasm3::CompilerError);
}

TEST_F(Qasm3ParserTest, ImportQasmBinaryTypeMismatch) {
  std::stringstream ss{};
  const std::string testfile = "OPENQASM 3.0;\n"
                               "int x = 1 + false;\n";

  ss << testfile;
  auto qc = QuantumComputation();
  EXPECT_THROW(
      {
        try {
          qc.import(ss, Format::OpenQASM3);
        } catch (const qasm3::CompilerError& e) {
          EXPECT_EQ(e.message, "Type check failed.");
          throw;
        }
      },
      qasm3::CompilerError);
}

TEST_F(Qasm3ParserTest, ImportQasmAssignmentIndexType) {
  std::stringstream ss{};
  const std::string testfile = "OPENQASM 3.0;\n"
                               "bit[16] x;\n"
                               "x[-1] = 0;\n";

  ss << testfile;
  auto qc = QuantumComputation();
  EXPECT_THROW(
      {
        try {
          qc.import(ss, Format::OpenQASM3);
        } catch (const qasm3::CompilerError& e) {
          EXPECT_EQ(e.message, "Type check failed.");
          throw;
        }
      },
      qasm3::CompilerError);
}

TEST_F(Qasm3ParserTest, ImportQasmUnknownIdentifier) {
  std::stringstream ss{};
  const std::string testfile = "OPENQASM 3.0;\n"
                               "int x = y;\n";

  ss << testfile;
  auto qc = QuantumComputation();
  EXPECT_THROW(
      {
        try {
          qc.import(ss, Format::OpenQASM3);
        } catch (const qasm3::CompilerError& e) {
          EXPECT_EQ(e.message, "Type check failed.");
          throw;
        }
      },
      qasm3::CompilerError);
}

TEST_F(Qasm3ParserTest, ImportQasmUnknownQubit) {
  std::stringstream ss{};
  const std::string testfile = "OPENQASM 3.0;\n"
                               "bit x = measure q;\n";

  ss << testfile;
  auto qc = QuantumComputation();
  EXPECT_THROW(
      {
        try {
          qc.import(ss, Format::OpenQASM3);
        } catch (const qasm3::CompilerError& e) {
          EXPECT_EQ(e.message, "Type check failed.");
          throw;
        }
      },
      qasm3::CompilerError);
}

TEST_F(Qasm3ParserTest, ImportQasmNegativeTypeDesignator) {
  std::stringstream ss{};
  const std::string testfile = "OPENQASM 3.0;\n"
                               "bit[-1] c;\n";

  ss << testfile;
  auto qc = QuantumComputation();
  EXPECT_THROW(
      {
        try {
          qc.import(ss, Format::OpenQASM3);
        } catch (const qasm3::CompilerError& e) {
          EXPECT_EQ(e.message, "Type check failed.");
          throw;
        }
      },
      qasm3::CompilerError);
}

TEST_F(Qasm3ParserTest, TestPrintTokens) {
  // This test is to print all tokens and make the coverage report happy.
  const auto tokens = std::vector{
      qasm3::Token(qasm3::Token::Kind::None, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Comment, 0, 0),
      qasm3::Token(qasm3::Token::Kind::OpenQasm, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Include, 0, 0),
      qasm3::Token(qasm3::Token::Kind::DefCalGrammar, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Def, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Cal, 0, 0),
      qasm3::Token(qasm3::Token::Kind::DefCal, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Gate, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Opaque, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Extern, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Box, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Let, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Break, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Continue, 0, 0),
      qasm3::Token(qasm3::Token::Kind::If, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Else, 0, 0),
      qasm3::Token(qasm3::Token::Kind::End, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Return, 0, 0),
      qasm3::Token(qasm3::Token::Kind::For, 0, 0),
      qasm3::Token(qasm3::Token::Kind::While, 0, 0),
      qasm3::Token(qasm3::Token::Kind::In, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Pragma, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Input, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Output, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Const, 0, 0),
      qasm3::Token(qasm3::Token::Kind::ReadOnly, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Mutable, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Qreg, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Qubit, 0, 0),
      qasm3::Token(qasm3::Token::Kind::CReg, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Bool, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Bit, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Int, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Uint, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Float, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Angle, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Complex, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Array, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Void, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Duration, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Stretch, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Gphase, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Inv, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Pow, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Ctrl, 0, 0),
      qasm3::Token(qasm3::Token::Kind::NegCtrl, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Dim, 0, 0),
      qasm3::Token(qasm3::Token::Kind::DurationOf, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Delay, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Reset, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Measure, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Barrier, 0, 0),
      qasm3::Token(qasm3::Token::Kind::True, 0, 0),
      qasm3::Token(qasm3::Token::Kind::False, 0, 0),
      qasm3::Token(qasm3::Token::Kind::LBracket, 0, 0),
      qasm3::Token(qasm3::Token::Kind::RBracket, 0, 0),
      qasm3::Token(qasm3::Token::Kind::LBrace, 0, 0),
      qasm3::Token(qasm3::Token::Kind::RBrace, 0, 0),
      qasm3::Token(qasm3::Token::Kind::LParen, 0, 0),
      qasm3::Token(qasm3::Token::Kind::RParen, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Colon, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Semicolon, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Eof, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Dot, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Comma, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Equals, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Arrow, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Plus, 0, 0),
      qasm3::Token(qasm3::Token::Kind::DoublePlus, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Minus, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Asterisk, 0, 0),
      qasm3::Token(qasm3::Token::Kind::DoubleAsterisk, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Slash, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Percent, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Pipe, 0, 0),
      qasm3::Token(qasm3::Token::Kind::DoublePipe, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Ampersand, 0, 0),
      qasm3::Token(qasm3::Token::Kind::DoubleAmpersand, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Caret, 0, 0),
      qasm3::Token(qasm3::Token::Kind::At, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Tilde, 0, 0),
      qasm3::Token(qasm3::Token::Kind::ExclamationPoint, 0, 0),
      qasm3::Token(qasm3::Token::Kind::DoubleEquals, 0, 0),
      qasm3::Token(qasm3::Token::Kind::NotEquals, 0, 0),
      qasm3::Token(qasm3::Token::Kind::PlusEquals, 0, 0),
      qasm3::Token(qasm3::Token::Kind::MinusEquals, 0, 0),
      qasm3::Token(qasm3::Token::Kind::AsteriskEquals, 0, 0),
      qasm3::Token(qasm3::Token::Kind::SlashEquals, 0, 0),
      qasm3::Token(qasm3::Token::Kind::AmpersandEquals, 0, 0),
      qasm3::Token(qasm3::Token::Kind::PipeEquals, 0, 0),
      qasm3::Token(qasm3::Token::Kind::TildeEquals, 0, 0),
      qasm3::Token(qasm3::Token::Kind::CaretEquals, 0, 0),
      qasm3::Token(qasm3::Token::Kind::LeftShitEquals, 0, 0),
      qasm3::Token(qasm3::Token::Kind::RightShiftEquals, 0, 0),
      qasm3::Token(qasm3::Token::Kind::PercentEquals, 0, 0),
      qasm3::Token(qasm3::Token::Kind::DoubleAsteriskEquals, 0, 0),
      qasm3::Token(qasm3::Token::Kind::LessThan, 0, 0),
      qasm3::Token(qasm3::Token::Kind::LessThanEquals, 0, 0),
      qasm3::Token(qasm3::Token::Kind::GreaterThan, 0, 0),
      qasm3::Token(qasm3::Token::Kind::GreaterThanEquals, 0, 0),
      qasm3::Token(qasm3::Token::Kind::LeftShift, 0, 0),
      qasm3::Token(qasm3::Token::Kind::RightShift, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Imag, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Underscore, 0, 0),
      qasm3::Token(qasm3::Token::Kind::TimeUnitDt, 0, 0),
      qasm3::Token(qasm3::Token::Kind::TimeUnitNs, 0, 0),
      qasm3::Token(qasm3::Token::Kind::TimeUnitUs, 0, 0),
      qasm3::Token(qasm3::Token::Kind::TimeUnitMys, 0, 0),
      qasm3::Token(qasm3::Token::Kind::TimeUnitMs, 0, 0),
      qasm3::Token(qasm3::Token::Kind::S, 0, 0),
      qasm3::Token(qasm3::Token::Kind::DoubleQuote, 0, 0),
      qasm3::Token(qasm3::Token::Kind::SingleQuote, 0, 0),
      qasm3::Token(qasm3::Token::Kind::BackSlash, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Identifier, 0, 0),
      qasm3::Token(qasm3::Token::Kind::HardwareQubit, 0, 0),
      qasm3::Token(qasm3::Token::Kind::StringLiteral, 0, 0),
      qasm3::Token(qasm3::Token::Kind::IntegerLiteral, 0, 0),
      qasm3::Token(qasm3::Token::Kind::FloatLiteral, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Sin, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Cos, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Tan, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Exp, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Ln, 0, 0),
      qasm3::Token(qasm3::Token::Kind::Sqrt, 0, 0),
  };

  // Print all tokens.
  std::stringstream ss{};
  for (const auto& token : tokens) {
    ss << token << "\n";
  }

  // We expect all tokens to look like this.
  // These are just all tokens joined by a newline.
  const std::string expected =
      "None\n"
      "Comment\n"
      "OPENQASM\n"
      "include\n"
      "DefCalGrammar\n"
      "Def\n"
      "Cal\n"
      "DefCal\n"
      "gate\n"
      "opaque\n"
      "extern\n"
      "box\n"
      "let\n"
      "break\n"
      "continue\n"
      "if\n"
      "else\n"
      "end\n"
      "return\n"
      "for\n"
      "while\n"
      "in\n"
      "pragma\n"
      "input\n"
      "output\n"
      "const\n"
      "readOnly\n"
      "mutable\n"
      "qreg\n"
      "qubit\n"
      "cReg\n"
      "bool\n"
      "bit\n"
      "int\n"
      "uint\n"
      "float\n"
      "angle\n"
      "complex\n"
      "array\n"
      "void\n"
      "duration\n"
      "stretch\n"
      "gphase\n"
      "inv\n"
      "pow\n"
      "ctrl\n"
      "negCtrl\n"
      "#dim\n"
      "durationof\n"
      "delay\n"
      "reset\n"
      "measure\n"
      "barrier\n"
      "true\n"
      "false\n"
      "[\n"
      "]\n"
      "{\n"
      "}\n"
      "(\n"
      ")\n"
      ":\n"
      ";\n"
      "Eof\n"
      ".\n"
      ",\n"
      "=\n"
      "->\n"
      "+\n"
      "++\n"
      "-\n"
      "*\n"
      "**\n"
      "/\n"
      "%\n"
      "|\n"
      "||\n"
      "&\n"
      "&&\n"
      "^\n"
      "@\n"
      "~\n"
      "!\n"
      "==\n"
      "!=\n"
      "+=\n"
      "-=\n"
      "*=\n"
      "/=\n"
      "&=\n"
      "|=\n"
      "~=\n"
      "^=\n"
      "<<=\n"
      ">>=\n"
      "%=\n"
      "**=\n"
      "<\n"
      "<=\n"
      ">\n"
      ">=\n"
      "<<\n"
      ">>\n"
      "imag\n"
      "underscore\n"
      "dt\n"
      "ns\n"
      "us\n"
      "mys\n"
      "ms\n"
      "s\n"
      "\"\n"
      "'\n"
      "\\\n"
      // These tokens are not keywords, but have a value associated
      "Identifier ()\n"
      "HardwareQubit\n"
      "StringLiteral (\"\")\n"
      "IntegerLiteral (0)\n"
      "FloatLiteral (0)\n"
      "sin\n"
      "cos\n"
      "tan\n"
      "exp\n"
      "ln\n"
      "sqrt\n";

  // Now we check if they are the same.
  EXPECT_EQ(ss.str(), expected);
}

TEST_F(Qasm3ParserTest, TestConstEval) {
  qasm3::const_eval::ConstEvalPass constEvalPass{};

  // Test constant eval.
  // The first element of the pair is the expression to be evaluated, the second
  // element is the expected result.
  const auto inputs = std::vector<std::pair<std::shared_ptr<qasm3::Expression>,
                                            qasm3::const_eval::ConstEvalValue>>{
      // integer unsigned
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::Power,
                    std::make_shared<qasm3::Constant>(2, false),
                    std::make_shared<qasm3::Constant>(2, false)),
                qasm3::const_eval::ConstEvalValue(4, false)},
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::Add,
                    std::make_shared<qasm3::Constant>(1, false),
                    std::make_shared<qasm3::Constant>(2, false)),
                qasm3::const_eval::ConstEvalValue(3, false)},
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::Subtract,
                    std::make_shared<qasm3::Constant>(5, false),
                    std::make_shared<qasm3::Constant>(2, false)),
                qasm3::const_eval::ConstEvalValue(3, false)},
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::Multiply,
                    std::make_shared<qasm3::Constant>(1, false),
                    std::make_shared<qasm3::Constant>(2, false)),
                qasm3::const_eval::ConstEvalValue(2, false)},
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::Divide,
                    std::make_shared<qasm3::Constant>(6, false),
                    std::make_shared<qasm3::Constant>(2, false)),
                qasm3::const_eval::ConstEvalValue(3, false)},
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::Modulo,
                    std::make_shared<qasm3::Constant>(1, false),
                    std::make_shared<qasm3::Constant>(2, false)),
                qasm3::const_eval::ConstEvalValue(1, false)},
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::LeftShift,
                    std::make_shared<qasm3::Constant>(2, false),
                    std::make_shared<qasm3::Constant>(1, false)),
                qasm3::const_eval::ConstEvalValue(4, false)},
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::RightShift,
                    std::make_shared<qasm3::Constant>(2, false),
                    std::make_shared<qasm3::Constant>(1, false)),
                qasm3::const_eval::ConstEvalValue(1, false)},
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::LessThan,
                    std::make_shared<qasm3::Constant>(1, false),
                    std::make_shared<qasm3::Constant>(2, false)),
                qasm3::const_eval::ConstEvalValue(true)},
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::LessThanOrEqual,
                    std::make_shared<qasm3::Constant>(1, false),
                    std::make_shared<qasm3::Constant>(2, false)),
                qasm3::const_eval::ConstEvalValue(true)},
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::GreaterThan,
                    std::make_shared<qasm3::Constant>(1, false),
                    std::make_shared<qasm3::Constant>(2, false)),
                qasm3::const_eval::ConstEvalValue(false)},
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::GreaterThanOrEqual,
                    std::make_shared<qasm3::Constant>(1, false),
                    std::make_shared<qasm3::Constant>(2, false)),
                qasm3::const_eval::ConstEvalValue(false)},
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::Equal,
                    std::make_shared<qasm3::Constant>(1, false),
                    std::make_shared<qasm3::Constant>(2, false)),
                qasm3::const_eval::ConstEvalValue(false)},
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::NotEqual,
                    std::make_shared<qasm3::Constant>(1, false),
                    std::make_shared<qasm3::Constant>(2, false)),
                qasm3::const_eval::ConstEvalValue(true)},
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::BitwiseAnd,
                    std::make_shared<qasm3::Constant>(1, false),
                    std::make_shared<qasm3::Constant>(3, false)),
                qasm3::const_eval::ConstEvalValue(1, false)},
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::BitwiseXor,
                    std::make_shared<qasm3::Constant>(1, false),
                    std::make_shared<qasm3::Constant>(2, false)),
                qasm3::const_eval::ConstEvalValue(3, false)},
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::BitwiseOr,
                    std::make_shared<qasm3::Constant>(1, false),
                    std::make_shared<qasm3::Constant>(2, false)),
                qasm3::const_eval::ConstEvalValue(3, false)},

      // integer signed
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::Power,
                    std::make_shared<qasm3::Constant>(2, true),
                    std::make_shared<qasm3::Constant>(2, true)),
                qasm3::const_eval::ConstEvalValue(4, true)},
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::Add,
                    std::make_shared<qasm3::Constant>(1, true),
                    std::make_shared<qasm3::Constant>(2, true)),
                qasm3::const_eval::ConstEvalValue(3, true)},
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::Subtract,
                    std::make_shared<qasm3::Constant>(5, true),
                    std::make_shared<qasm3::Constant>(2, true)),
                qasm3::const_eval::ConstEvalValue(3, true)},
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::Multiply,
                    std::make_shared<qasm3::Constant>(1, true),
                    std::make_shared<qasm3::Constant>(2, true)),
                qasm3::const_eval::ConstEvalValue(2, true)},
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::Divide,
                    std::make_shared<qasm3::Constant>(6, true),
                    std::make_shared<qasm3::Constant>(2, true)),
                qasm3::const_eval::ConstEvalValue(3, true)},
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::Modulo,
                    std::make_shared<qasm3::Constant>(1, true),
                    std::make_shared<qasm3::Constant>(2, true)),
                qasm3::const_eval::ConstEvalValue(1, true)},
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::LeftShift,
                    std::make_shared<qasm3::Constant>(2, true),
                    std::make_shared<qasm3::Constant>(1, true)),
                qasm3::const_eval::ConstEvalValue(4, true)},
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::RightShift,
                    std::make_shared<qasm3::Constant>(2, true),
                    std::make_shared<qasm3::Constant>(1, true)),
                qasm3::const_eval::ConstEvalValue(1, true)},
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::LessThan,
                    std::make_shared<qasm3::Constant>(1, true),
                    std::make_shared<qasm3::Constant>(2, true)),
                qasm3::const_eval::ConstEvalValue(true)},
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::LessThanOrEqual,
                    std::make_shared<qasm3::Constant>(1, true),
                    std::make_shared<qasm3::Constant>(2, true)),
                qasm3::const_eval::ConstEvalValue(true)},
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::GreaterThan,
                    std::make_shared<qasm3::Constant>(1, true),
                    std::make_shared<qasm3::Constant>(2, true)),
                qasm3::const_eval::ConstEvalValue(false)},
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::GreaterThanOrEqual,
                    std::make_shared<qasm3::Constant>(1, true),
                    std::make_shared<qasm3::Constant>(2, true)),
                qasm3::const_eval::ConstEvalValue(false)},
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::Equal,
                    std::make_shared<qasm3::Constant>(1, true),
                    std::make_shared<qasm3::Constant>(2, true)),
                qasm3::const_eval::ConstEvalValue(false)},
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::NotEqual,
                    std::make_shared<qasm3::Constant>(1, true),
                    std::make_shared<qasm3::Constant>(2, true)),
                qasm3::const_eval::ConstEvalValue(true)},
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::BitwiseAnd,
                    std::make_shared<qasm3::Constant>(1, true),
                    std::make_shared<qasm3::Constant>(3, true)),
                qasm3::const_eval::ConstEvalValue(1, true)},
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::BitwiseXor,
                    std::make_shared<qasm3::Constant>(1, true),
                    std::make_shared<qasm3::Constant>(2, true)),
                qasm3::const_eval::ConstEvalValue(3, true)},
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::BitwiseOr,
                    std::make_shared<qasm3::Constant>(1, true),
                    std::make_shared<qasm3::Constant>(2, true)),
                qasm3::const_eval::ConstEvalValue(3, true)},

      // float
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::Power,
                    std::make_shared<qasm3::Constant>(2.0),
                    std::make_shared<qasm3::Constant>(2.0)),
                qasm3::const_eval::ConstEvalValue(4.0)},
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::Add,
                    std::make_shared<qasm3::Constant>(1.0),
                    std::make_shared<qasm3::Constant>(2.0)),
                qasm3::const_eval::ConstEvalValue(3.0)},
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::Subtract,
                    std::make_shared<qasm3::Constant>(5.0),
                    std::make_shared<qasm3::Constant>(2.0)),
                qasm3::const_eval::ConstEvalValue(3.0)},
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::Multiply,
                    std::make_shared<qasm3::Constant>(1.0),
                    std::make_shared<qasm3::Constant>(2.0)),
                qasm3::const_eval::ConstEvalValue(2.0)},
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::Divide,
                    std::make_shared<qasm3::Constant>(6.0),
                    std::make_shared<qasm3::Constant>(2.0)),
                qasm3::const_eval::ConstEvalValue(3.0)},
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::Modulo,
                    std::make_shared<qasm3::Constant>(1.0),
                    std::make_shared<qasm3::Constant>(2.0)),
                qasm3::const_eval::ConstEvalValue(1.0)},
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::LessThan,
                    std::make_shared<qasm3::Constant>(1.0),
                    std::make_shared<qasm3::Constant>(2.0)),
                qasm3::const_eval::ConstEvalValue(true)},
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::LessThanOrEqual,
                    std::make_shared<qasm3::Constant>(1.0),
                    std::make_shared<qasm3::Constant>(2.0)),
                qasm3::const_eval::ConstEvalValue(true)},
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::GreaterThan,
                    std::make_shared<qasm3::Constant>(1.0),
                    std::make_shared<qasm3::Constant>(2.0)),
                qasm3::const_eval::ConstEvalValue(false)},
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::GreaterThanOrEqual,
                    std::make_shared<qasm3::Constant>(1.0),
                    std::make_shared<qasm3::Constant>(2.0)),
                qasm3::const_eval::ConstEvalValue(false)},
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::Equal,
                    std::make_shared<qasm3::Constant>(1.0),
                    std::make_shared<qasm3::Constant>(2.0)),
                qasm3::const_eval::ConstEvalValue(false)},
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::NotEqual,
                    std::make_shared<qasm3::Constant>(1.0),
                    std::make_shared<qasm3::Constant>(2.0)),
                qasm3::const_eval::ConstEvalValue(true)},

      // bool
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::Equal,
                    std::make_shared<qasm3::Constant>(true),
                    std::make_shared<qasm3::Constant>(false)),
                qasm3::const_eval::ConstEvalValue(false)},
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::NotEqual,
                    std::make_shared<qasm3::Constant>(true),
                    std::make_shared<qasm3::Constant>(false)),
                qasm3::const_eval::ConstEvalValue(true)},
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::BitwiseAnd,
                    std::make_shared<qasm3::Constant>(true),
                    std::make_shared<qasm3::Constant>(false)),
                qasm3::const_eval::ConstEvalValue(false)},
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::BitwiseXor,
                    std::make_shared<qasm3::Constant>(true),
                    std::make_shared<qasm3::Constant>(false)),
                qasm3::const_eval::ConstEvalValue(true)},
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::BitwiseOr,
                    std::make_shared<qasm3::Constant>(true),
                    std::make_shared<qasm3::Constant>(false)),
                qasm3::const_eval::ConstEvalValue(true)},
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::LogicalAnd,
                    std::make_shared<qasm3::Constant>(true),
                    std::make_shared<qasm3::Constant>(false)),
                qasm3::const_eval::ConstEvalValue(false)},
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::LogicalOr,
                    std::make_shared<qasm3::Constant>(true),
                    std::make_shared<qasm3::Constant>(false)),
                qasm3::const_eval::ConstEvalValue(true)},

      // coercion
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::Add,
                    std::make_shared<qasm3::Constant>(2, true),
                    std::make_shared<qasm3::Constant>(3.0)),
                qasm3::const_eval::ConstEvalValue(5.0)},
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::Add,
                    std::make_shared<qasm3::Constant>(2, false),
                    std::make_shared<qasm3::Constant>(3.0)),
                qasm3::const_eval::ConstEvalValue(5.0)},
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::Subtract,
                    std::make_shared<qasm3::Constant>(6U, false),
                    std::make_shared<qasm3::Constant>(2, true)),
                qasm3::const_eval::ConstEvalValue(4, true)},
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::Add,
                    std::make_shared<qasm3::Constant>(3.0),
                    std::make_shared<qasm3::Constant>(2, true)),
                qasm3::const_eval::ConstEvalValue(5.0)},
      std::pair{std::make_shared<qasm3::BinaryExpression>(
                    qasm3::BinaryExpression::Op::Add,
                    std::make_shared<qasm3::Constant>(3.0),
                    std::make_shared<qasm3::Constant>(2, false)),
                qasm3::const_eval::ConstEvalValue(5.0)},

      // unary expr
      std::pair{
          std::make_shared<qasm3::UnaryExpression>(
              qasm3::UnaryExpression::Op::BitwiseNot,
              std::make_shared<qasm3::Constant>(0xFFFF'FFFF'FFFF'FFFF, false)),
          qasm3::const_eval::ConstEvalValue(0, false)},
      std::pair{std::make_shared<qasm3::UnaryExpression>(
                    qasm3::UnaryExpression::Op::LogicalNot,
                    std::make_shared<qasm3::Constant>(false)),
                qasm3::const_eval::ConstEvalValue(true)},
      std::pair{std::make_shared<qasm3::UnaryExpression>(
                    qasm3::UnaryExpression::Op::Negate,
                    std::make_shared<qasm3::Constant>(1.0)),
                qasm3::const_eval::ConstEvalValue(-1.0)},
      std::pair{std::make_shared<qasm3::UnaryExpression>(
                    qasm3::UnaryExpression::Op::Negate,
                    std::make_shared<qasm3::Constant>(1, true)),
                qasm3::const_eval::ConstEvalValue(-1, true)},
      std::pair{std::make_shared<qasm3::UnaryExpression>(
                    qasm3::UnaryExpression::Op::Sin,
                    std::make_shared<qasm3::Constant>(1.0)),
                qasm3::const_eval::ConstEvalValue(std::sin(1.0))},
      std::pair{std::make_shared<qasm3::UnaryExpression>(
                    qasm3::UnaryExpression::Op::Cos,
                    std::make_shared<qasm3::Constant>(1.0)),
                qasm3::const_eval::ConstEvalValue(std::cos(1.0))},
      std::pair{std::make_shared<qasm3::UnaryExpression>(
                    qasm3::UnaryExpression::Op::Tan,
                    std::make_shared<qasm3::Constant>(1.0)),
                qasm3::const_eval::ConstEvalValue(std::tan(1.0))},
      std::pair{std::make_shared<qasm3::UnaryExpression>(
                    qasm3::UnaryExpression::Op::Exp,
                    std::make_shared<qasm3::Constant>(1.0)),
                qasm3::const_eval::ConstEvalValue(std::exp(1.0))},
      std::pair{std::make_shared<qasm3::UnaryExpression>(
                    qasm3::UnaryExpression::Op::Ln,
                    std::make_shared<qasm3::Constant>(1.0)),
                qasm3::const_eval::ConstEvalValue(std::log(1.0))},
      std::pair{std::make_shared<qasm3::UnaryExpression>(
                    qasm3::UnaryExpression::Op::Sqrt,
                    std::make_shared<qasm3::Constant>(1.0)),
                qasm3::const_eval::ConstEvalValue(std::sqrt(1.0))},

  };

  for (const auto& [expr, expected] : inputs) {
    auto result = constEvalPass.visit(expr);
    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(result, expected);
  }
}
