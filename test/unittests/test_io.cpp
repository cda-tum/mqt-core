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
        //std::remove(output.c_str()); 
        //std::remove(output2.c_str()); 
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

void compare_files(std::string file1, std::string file2, bool strip_whitespaces = false) {
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


    compare_files(output, output2);
}