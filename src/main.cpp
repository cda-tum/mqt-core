#include "QuantumComputation.hpp"

#include <iostream>

// TODO: this whole file is just for testing and debugging the parser;
//      remove it before the PR will be merged.

int main() {
  qc::QuantumComputation q;

  //    std::string const s = " OPENQASM 3.0;\n"
  //                    "include \"qelib1.inc\";\n"
  //                    "qubit q[2];\n"
  //                    "h q;\n"
  //                    "cx q[0], q[1];\n"
  //                    "measure q -> c;\n";
  //    std::string const  s = "OPENQASM 3.0;\n"
  //                           "// i 0 1 2 3 4 5 6 7 9 8 10\n"
  //                           "// o 0 1 2 3 4 5 6 7 8 10 9\n"
  //                           "const int x = 1;\n"
  //                           "const int y = 2 * x;\n"
  //                           "const int z = y + 5 + y;\n"
  //                           "creg c[11];\n"
  //                           "qubit[2] q;\n"
  //                           "qubit[z] r;";
  //    std::string const s = "OPENQASM 3.0;\n"
  //                           "gate g a {\n"
  //                           "  U(0, 0, 0) a;\n"
  //                           "}\n"
  //                           "qubit[2] q;\n"
  //                           "g q[0];\n";
  //    std::string const s = "OPENQASM 3.0;\n"
  //                           "qubit[5] q1;\n"
  //                           "const float x = 0;\n"
  //                           "gate asdf q, p, r {\n"
  //                           "  U(π/2, 0, π) q;\n"
  //                           "  U(π/2, x, π) q;\n"
  //                           "  U(π/2, 0, π) r;\n"
  //                           "  gphase -π/3;\n"
  //                           "}\n"
  //                           "qubit[3] q;\n"
  //                           "asdf q[0] q[1] q[2];\n";
  //    std::string const s = "OPENQASM 3.0;\n"
  //                          "gate h q {\n"
  //                          "  U(π/2, 0, π) q;\n"
  //                          "  gphase π;\n"
  //                          "}\n"
  //                          "gate X q {\n"
  //                          "  U(π, 0, π) q;\n"
  //                          "  gphase -π/2;\n"
  //                          "}\n"
  //                          "gate CX c, t {\n"
  //                          "  ctrl @ X c, t;\n"
  //                          "}\n"
  //                          "gate neg_cx q1, q2 {\n"
  //                          "  negctrl @ X q1, q2;\n"
  //                          "}"
  //                          "qubit[4] q;\n"
  //                          "h q[0];\n"
  //                          "CX q[0], q[1];\n"
  //                          "neg_cx q[2], q[3];\n"
  //                          "";

  std::string const s = "OPENQASM 3.0;\n"
                        "include \"stdgates.inc\";\n"
                        "const uint N_QUBITS = 4;\n"
                        "const uint[64] CTRL_INDEX = 0;\n"
                        "const int a = -1;\n"
                        "qubit[N_QUBITS] q;\n"
                        "qubit[N_QUBITS] p;\n"
                        "bit[N_QUBITS] c;\n"
                        "ctrl @ cx q[0], q[1], q[2];\n"
                        "h q[0];\n"
                        "bit r = measure q[0];\n"
                        "bit[4] r1 = measure q;\n"
                        "h q[0];\n"
                        "reset q[0];\n"
                        "barrier q;\n"
                        "barrier q[1];\n"
                        "measure q[3] -> r[0];\n"
                        "measure q[3] -> r;\n"
                        "measure q -> r1;\n"
                        "r1 = measure q;\n"
                        "r = measure q[1];\n"
                        "x q[1];\n"
                        "reset q;\n"
                        "gate mycx(alpha) c, t {\n"
                        "  crz(alpha) c, t;\n"
                        "  ctrl @ x c, t;\n"
                        "}\n"
                        "inv @ mycx(0) q[0], p[0];\n"
                        "mycx(0) q[0], p[0];\n"
                        "ctrl(2) @ x q[CTRL_INDEX], q[CTRL_INDEX + 1], p;\n"
                        "c[0] = measure q[0];\n"
                        "c = measure q;\n"
                        "if (c == 1) {\n"
                        "  x q[0];\n"
                        "}"
                        "";

  std::cout << s << '\n';

  std::istringstream iss(s);

  q.import(iss, qc::Format::OpenQASM3);

  std::stringstream ss;
  q.dump(ss, qc::Format::OpenQASM);

  std::cout << ss.str() << '\n';

  return 0;
}
