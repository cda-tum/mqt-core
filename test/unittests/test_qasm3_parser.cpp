#include "Definitions.hpp"
#include "QuantumComputation.hpp"
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
                               "  x q;\n"
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
