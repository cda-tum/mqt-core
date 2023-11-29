#include "Definitions.hpp"
#include "QuantumComputation.hpp"
#include "parsers/qasm3_parser/Exception.hpp"
#include "parsers/qasm3_parser/Parser.hpp"
#include "parsers/qasm3_parser/Scanner.hpp"

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
                               "const int N_2 = -8;\n"
                               "qubit[N_1 - -N_2] q;\n"
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
        } catch (const qasm3::CompilerError e) {
          std::cerr << e.message << "\n";
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
        } catch (const qasm3::CompilerError e) {
          std::cerr << e.message << "\n";
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
        } catch (const qasm3::CompilerError e) {
          std::cerr << e.message << "\n";
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
        } catch (const qasm3::CompilerError e) {
          std::cerr << e.message << "\n";
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
        } catch (const qasm3::CompilerError e) {
          std::cerr << e.message << "\n";
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
        } catch (const qasm3::CompilerError e) {
          std::cerr << e.message << "\n";
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
        } catch (const qasm3::CompilerError e) {
          std::cerr << e.message << "\n";
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
        } catch (const qasm3::CompilerError e) {
          std::cerr << e.message << "\n";
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
        } catch (const qasm3::CompilerError e) {
          std::cerr << e.message << "\n";
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
        } catch (const qasm3::CompilerError e) {
          std::cerr << e.message << "\n";
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
        } catch (const qasm3::CompilerError e) {
          std::cerr << e.message << "\n";
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
        } catch (const qasm3::CompilerError e) {
          std::cerr << e.message << "\n";
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
        } catch (const qasm3::CompilerError e) {
          std::cerr << e.message << "\n";
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
        } catch (const qasm3::CompilerError e) {
          std::cerr << e.message << "\n";
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
        } catch (const qasm3::CompilerError e) {
          std::cerr << e.message << "\n";
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
        } catch (const qasm3::CompilerError e) {
          std::cerr << e.message << "\n";
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
        } catch (const qasm3::CompilerError e) {
          std::cerr << e.message << "\n";
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
        } catch (const qasm3::CompilerError e) {
          std::cerr << e.message << "\n";
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
        } catch (const qasm3::CompilerError e) {
          std::cerr << e.message << "\n";
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
        } catch (const qasm3::CompilerError e) {
          std::cerr << e.message << "\n";
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
        } catch (const qasm3::CompilerError e) {
          std::cerr << e.message << "\n";
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
        } catch (const qasm3::CompilerError e) {
          std::cerr << e.message << "\n";
          EXPECT_EQ(e.message, "Expected statement, got Plus.");
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
        } catch (const qasm3::CompilerError e) {
          std::cerr << e.message << "\n";
          EXPECT_EQ(
              e.message,
              "Version declaration must be at the beginning of the file.");
          throw;
        }
      },
      qasm3::CompilerError);
}
