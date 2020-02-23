/*
 * This file is part of IIC-JKU QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#include "gtest/gtest.h"

#include "QuantumComputation.hpp"

#include <iostream>
#include <string>
#include <fstream>
#include <streambuf>

class IO : public testing::TestWithParam<std::tuple<std::string, qc::Format>> {
protected:
	void TearDown() override {

	}

	void SetUp() override {
        qc = std::make_unique<qc::QuantumComputation>();
	}

	unsigned short nqubits = 0;
	unsigned int   seed    = 0;
    std::string    output  = "tmp.txt";
    std::string    output2 = "tmp2.txt";
	std::unique_ptr<qc::QuantumComputation> qc;
};

void compare_files(const std::string& file1, const std::string& file2, bool strip_whitespaces = false) {
    std::ifstream fstream1(file1);
    std::string str1((std::istreambuf_iterator<char>(fstream1)),
                     std::istreambuf_iterator<char>());
    std::ifstream fstream2(file2);
    std::string str2((std::istreambuf_iterator<char>(fstream2)),
                     std::istreambuf_iterator<char>());
    if(strip_whitespaces) {
        str1.erase(std::remove_if(str1.begin(), str1.end(), isspace), str1.end());
        str2.erase(std::remove_if(str2.begin(), str2.end(), isspace), str2.end());
    }
    ASSERT_EQ(str1, str2);
}

INSTANTIATE_TEST_SUITE_P(IO,
                         IO,
                         testing::Values(std::make_tuple("circuits/test.qasm", qc::OpenQASM)), //std::make_tuple("circuits/test.real", qc::Real
                         [](const testing::TestParamInfo<IO::ParamType>& info) {
                             qc::Format format = std::get<1>(info.param);
	                         
                             switch(format) {
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

    ASSERT_NO_THROW(qc->import(input,  format));
    ASSERT_NO_THROW(qc->dump(output,   format));
    ASSERT_NO_THROW(qc->reset());
    ASSERT_NO_THROW(qc->import(output, format));
    ASSERT_NO_THROW(qc->dump(output2,  format));

    compare_files(output, output2, true);
}

TEST_F(IO, importFromString) {
	std::string bell_circuit_qasm = "OPENQASM 2.0;\nqreg q[2];\nU(pi/2,0,pi) q[0];\nCX q[0],q[1];\n";
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
	std::string circuit_qasm = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[3];\nccx q,q[1];\n";
	std::stringstream ss{circuit_qasm};
	EXPECT_THROW(qc->import(ss, qc::OpenQASM), qasm::QASMParserException);
}

TEST_F(IO, invalid_real_header) {
	std::string circuit_real = ".numvars 2\nvariables q0 q1\n.begin\nh1 q0\nt2 q0 q1\n.end\n";
	std::stringstream ss{circuit_real};
	EXPECT_THROW(qc->import(ss, qc::Real), qc::QFRException);
}

TEST_F(IO, invalid_real_command) {
	std::string circuit_real = ".numvars 2\n.var q0 q1\n.begin\nh1 q0\n# test comment\nt2 q0 q1\n.end\n";
	std::stringstream ss{circuit_real};
	EXPECT_THROW(qc->import(ss, qc::Real), qc::QFRException);
}
