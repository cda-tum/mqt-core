/*
 * This file is part of JKQ QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#include "QuantumComputation.hpp"

#include "gtest/gtest.h"
#include <fstream>
#include <iostream>
#include <streambuf>
#include <string>

class IO: public testing::TestWithParam<std::tuple<std::string, qc::Format>> {
protected:
    void TearDown() override {
    }

    void SetUp() override {
        qc = std::make_unique<qc::QuantumComputation>();
    }

    dd::QubitCount                          nqubits = 0;
    unsigned int                            seed    = 0;
    std::string                             output  = "tmp.txt";
    std::string                             output2 = "tmp2.txt";
    std::unique_ptr<qc::QuantumComputation> qc;
};

void compare_files(const std::string& file1, const std::string& file2, bool strip_whitespaces = false) {
    std::ifstream fstream1(file1);
    std::string   str1((std::istreambuf_iterator<char>(fstream1)),
                     std::istreambuf_iterator<char>());
    std::ifstream fstream2(file2);
    std::string   str2((std::istreambuf_iterator<char>(fstream2)),
                     std::istreambuf_iterator<char>());
    if (strip_whitespaces) {
        str1.erase(std::remove_if(str1.begin(), str1.end(), isspace), str1.end());
        str2.erase(std::remove_if(str2.begin(), str2.end(), isspace), str2.end());
    }
    ASSERT_EQ(str1, str2);
}

INSTANTIATE_TEST_SUITE_P(IO,
                         IO,
                         testing::Values(std::make_tuple("./circuits/test.qasm", qc::OpenQASM)), //std::make_tuple("circuits/test.real", qc::Real
                         [](const testing::TestParamInfo<IO::ParamType>& info) {
                             qc::Format format = std::get<1>(info.param);

                             switch (format) {
                                 case qc::Real:
                                     return "Real";
                                 case qc::OpenQASM:
                                     return "OpenQasm";
                                 case qc::GRCS:
                                     return "GRCS";
                                 default: return "Unknown format";
                             }
                         });

TEST_P(IO, importAndDump) {
    std::string input;
    qc::Format  format;
    std::tie(input, format) = GetParam();
    std::cout << "FILE: " << input << std::endl;

    ASSERT_NO_THROW(qc->import(input, format));
    ASSERT_NO_THROW(qc->dump(output, format));
    ASSERT_NO_THROW(qc->reset());
    ASSERT_NO_THROW(qc->import(output, format));
    ASSERT_NO_THROW(qc->dump(output2, format));

    compare_files(output, output2, true);
}

TEST_F(IO, importFromString) {
    std::string       bell_circuit_qasm = "OPENQASM 2.0;\nqreg q[2];\nU(pi/2,0,pi) q[0];\nCX q[0],q[1];\n";
    std::stringstream ss{bell_circuit_qasm};
    ASSERT_NO_THROW(qc->import(ss, qc::OpenQASM));
    std::cout << *qc << std::endl;
    std::string bell_circuit_real = ".numvars 2\n.variables q0 q1\n.begin\nh1 q0\nt2 q0 q1\n.end\n";
    ss.clear();
    ss.str(bell_circuit_real);
    ASSERT_NO_THROW(qc->import(ss, qc::Real));
    std::cout << *qc << std::endl;
}

TEST_F(IO, controlled_op_acting_on_whole_register) {
    std::string       circuit_qasm = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[2];\ncx q,q[1];\n";
    std::stringstream ss{circuit_qasm};
    EXPECT_THROW(qc->import(ss, qc::OpenQASM), qasm::QASMParserException);
}

TEST_F(IO, invalid_real_header) {
    std::string       circuit_real = ".numvars 2\nvariables q0 q1\n.begin\nh1 q0\nt2 q0 q1\n.end\n";
    std::stringstream ss{circuit_real};
    EXPECT_THROW(qc->import(ss, qc::Real), qc::QFRException);
}

TEST_F(IO, invalid_real_command) {
    std::string       circuit_real = ".numvars 2\n.var q0 q1\n.begin\nh1 q0\n# test comment\nt2 q0 q1\n.end\n";
    std::stringstream ss{circuit_real};
    EXPECT_THROW(qc->import(ss, qc::Real), qc::QFRException);
}

TEST_F(IO, insufficient_registers_qelib) {
    std::string       circuit_qasm = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[2];\ncx q[0];\n";
    std::stringstream ss{circuit_qasm};
    EXPECT_THROW(qc->import(ss, qc::OpenQASM), qasm::QASMParserException);
}

TEST_F(IO, insufficient_registers_enhanced_qelib) {
    std::string       circuit_qasm = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[4];\ncccz q[0], q[1], q[2];\n";
    std::stringstream ss{circuit_qasm};
    EXPECT_THROW(qc->import(ss, qc::OpenQASM), qasm::QASMParserException);
}

TEST_F(IO, superfluous_registers_qelib) {
    std::string       circuit_qasm = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[3];\ncx q[0], q[1], q[2];\n";
    std::stringstream ss{circuit_qasm};
    EXPECT_THROW(qc->import(ss, qc::OpenQASM), qasm::QASMParserException);
}

TEST_F(IO, superfluous_registers_enhanced_qelib) {
    std::string       circuit_qasm = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[5];\ncccz q[0], q[1], q[2], q[3], q[4];\n";
    std::stringstream ss{circuit_qasm};
    EXPECT_THROW(qc->import(ss, qc::OpenQASM), qasm::QASMParserException);
}

TEST_F(IO, dump_negative_control) {
    std::string       circuit_real = ".numvars 2\n.variables a b\n.begin\nt2 -a b\n.end";
    std::stringstream ss{circuit_real};
    qc->import(ss, qc::Real);
    qc->dump("testdump.qasm");
    qc->import("testdump.qasm");
    ASSERT_EQ(qc->getNops(), 3);
    auto it = qc->begin();
    EXPECT_EQ((*it)->getType(), qc::X);
    EXPECT_EQ((*it)->getControls().size(), 0);
    ++it;
    EXPECT_EQ((*it)->getType(), qc::X);
    EXPECT_EQ((*it)->getControls().size(), 1);
    ++it;
    EXPECT_EQ((*it)->getType(), qc::X);
    EXPECT_EQ((*it)->getControls().size(), 0);
}

TEST_F(IO, qiskit_mcx_gray) {
    std::stringstream ss{};
    ss << "OPENQASM 2.0;"
       << "include \"qelib1.inc\";"
       << "qreg q[4];"
       << "mcx_gray q[0], q[1], q[2], q[3];"
       << std::endl;
    qc->import(ss, qc::OpenQASM);
    auto& gate = *(qc->begin());
    std::cout << *qc << std::endl;
    EXPECT_EQ(gate->getType(), qc::X);
    EXPECT_EQ(gate->getNcontrols(), 3);
    EXPECT_EQ(gate->getTargets().at(0), 3);
}

TEST_F(IO, qiskit_mcx_recursive) {
    std::stringstream ss{};
    ss << "OPENQASM 2.0;"
       << "include \"qelib1.inc\";"
       << "qreg q[6];"
       << "qreg anc[1];"
       << "mcx_recursive q[0], q[1], q[2], q[3], q[4];"
       << "mcx_recursive q[0], q[1], q[2], q[3], q[4], q[5], anc[0];"
       << std::endl;
    qc->import(ss, qc::OpenQASM);
    auto& gate = *(qc->begin());
    std::cout << *qc << std::endl;
    EXPECT_EQ(gate->getType(), qc::X);
    EXPECT_EQ(gate->getNcontrols(), 4);
    EXPECT_EQ(gate->getTargets().at(0), 4);
    auto& second = *(++qc->begin());
    EXPECT_EQ(second->getType(), qc::X);
    EXPECT_EQ(second->getNcontrols(), 5);
    EXPECT_EQ(second->getTargets().at(0), 5);
}

TEST_F(IO, qiskit_mcx_vchain) {
    std::stringstream ss{};
    ss << "OPENQASM 2.0;"
       << "include \"qelib1.inc\";"
       << "qreg q[4];"
       << "qreg anc[1];"
       << "mcx_vchain q[0], q[1], q[2], q[3], anc[0];"
       << std::endl;
    qc->import(ss, qc::OpenQASM);
    auto& gate = *(qc->begin());
    std::cout << *qc << std::endl;
    EXPECT_EQ(gate->getType(), qc::X);
    EXPECT_EQ(gate->getNcontrols(), 3);
    EXPECT_EQ(gate->getTargets().at(0), 3);
}

TEST_F(IO, qiskit_mcx_duplicate_qubit) {
    std::stringstream ss{};
    ss << "OPENQASM 2.0;"
       << "include \"qelib1.inc\";"
       << "qreg q[4];"
       << "qreg anc[1];"
       << "mcx_vchain q[0], q[0], q[2], q[3], anc[0];"
       << std::endl;
    EXPECT_THROW(qc->import(ss, qc::OpenQASM), qasm::QASMParserException);
}

TEST_F(IO, qiskit_mcx_qubit_register) {
    std::stringstream ss{};
    ss << "OPENQASM 2.0;"
       << "include \"qelib1.inc\";"
       << "qreg q[4];"
       << "qreg anc[1];"
       << "mcx_vchain q, q[0], q[2], q[3], anc[0];"
       << std::endl;
    EXPECT_THROW(qc->import(ss, qc::OpenQASM), qasm::QASMParserException);
}

TEST_F(IO, tfc_input) {
    qc->import("./circuits/test.tfc");
    std::cout << *qc << std::endl;
}

TEST_F(IO, qc_input) {
    qc->import("./circuits/test.qc");
    std::cout << *qc << std::endl;
}

TEST_F(IO, grcs_input) {
    qc->import("./circuits/grcs/bris_4_40_9_v2.txt");
    std::cout << *qc << std::endl;
    qc->import("./circuits/grcs/inst_4x4_80_9_v2.txt");
    std::cout << *qc << std::endl;
}

TEST_F(IO, classic_controlled) {
    std::stringstream ss{};
    ss << "OPENQASM 2.0;"
       << "include \"qelib1.inc\";"
       << "qreg q[1];"
       << "creg c[1];"
       << "h q[0];"
       << "measure q->c;"
       << "// test classic controlled operation\n"
       << "if (c==1) x q[0];"
       << std::endl;
    EXPECT_NO_THROW(qc->import(ss, qc::OpenQASM););
    std::cout << *qc << std::endl;
}
