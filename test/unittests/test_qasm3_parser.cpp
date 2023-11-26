#include "Definitions.hpp"
#include "QuantumComputation.hpp"

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
