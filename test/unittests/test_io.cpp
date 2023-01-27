/*
 * This file is part of MQT QFR library which is released under the MIT license.
 * See file README.md or go to https://www.cda.cit.tum.de/research/quantum/ for more information.
 */

#include "QuantumComputation.hpp"

#include "gtest/gtest.h"
#include <iostream>
#include <string>

class IO: public testing::TestWithParam<std::tuple<std::string, qc::Format>> {
protected:
    void TearDown() override {
    }

    void SetUp() override {
        qc = std::make_unique<qc::QuantumComputation>();
    }

    std::size_t                             nqubits = 0;
    unsigned int                            seed    = 0;
    std::string                             output  = "tmp.txt";
    std::string                             output2 = "tmp2.txt";
    std::unique_ptr<qc::QuantumComputation> qc;
};

void compareFiles(const std::string& file1, const std::string& file2, bool stripWhitespaces = false) {
    std::ifstream fstream1(file1);
    std::string   str1((std::istreambuf_iterator<char>(fstream1)),
                       std::istreambuf_iterator<char>());
    std::ifstream fstream2(file2);
    std::string   str2((std::istreambuf_iterator<char>(fstream2)),
                       std::istreambuf_iterator<char>());
    if (stripWhitespaces) {
        str1.erase(std::remove_if(str1.begin(), str1.end(), isspace), str1.end());
        str2.erase(std::remove_if(str2.begin(), str2.end(), isspace), str2.end());
    }
    ASSERT_EQ(str1, str2);
}

INSTANTIATE_TEST_SUITE_P(IO,
                         IO,
                         testing::Values(std::make_tuple("./circuits/test.qasm", qc::Format::OpenQASM)), //std::make_tuple("circuits/test.real", qc::Format::Real
                         [](const testing::TestParamInfo<IO::ParamType>& inf) {
                             const qc::Format format = std::get<1>(inf.param);

                             switch (format) {
                                 case qc::Format::Real:
                                     return "Real";
                                 case qc::Format::OpenQASM:
                                     return "OpenQasm";
                                 case qc::Format::GRCS:
                                     return "GRCS";
                                 default: return "Unknown format";
                             }
                         });

TEST_P(IO, importAndDump) {
    const auto& [input, format] = GetParam();
    std::cout << "FILE: " << input << std::endl;

    ASSERT_NO_THROW(qc->import(input, format));
    ASSERT_NO_THROW(qc->dump(output, format));
    ASSERT_NO_THROW(qc->reset());
    ASSERT_NO_THROW(qc->import(output, format));
    ASSERT_NO_THROW(qc->dump(output2, format));

    compareFiles(output, output2, true);
}

TEST_F(IO, importFromString) {
    const std::string bellCircuitQasm = "OPENQASM 2.0;\nqreg q[2];\nU(pi/2,0,pi) q[0];\nCX q[0],q[1];\n";
    std::stringstream ss{bellCircuitQasm};
    ASSERT_NO_THROW(qc->import(ss, qc::Format::OpenQASM));
    std::cout << *qc << std::endl;
    const std::string bellCircuitReal = ".numvars 2\n.variables q0 q1\n.begin\nh1 q0\nt2 q0 q1\n.end\n";
    ss.clear();
    ss.str(bellCircuitReal);
    ASSERT_NO_THROW(qc->import(ss, qc::Format::Real));
    std::cout << *qc << std::endl;
}

TEST_F(IO, controlledOpActingOnWholeRegister) {
    const std::string circuitQasm = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[2];\ncx q,q[1];\n";
    std::stringstream ss{circuitQasm};
    EXPECT_THROW(qc->import(ss, qc::Format::OpenQASM), qasm::QASMParserException);
}

TEST_F(IO, invalidRealHeader) {
    const std::string circuitReal = ".numvars 2\nvariables q0 q1\n.begin\nh1 q0\nt2 q0 q1\n.end\n";
    std::stringstream ss{circuitReal};
    EXPECT_THROW(qc->import(ss, qc::Format::Real), qc::QFRException);
}

TEST_F(IO, invalidRealCommand) {
    const std::string circuitReal = ".numvars 2\n.var q0 q1\n.begin\nh1 q0\n# test comment\nt2 q0 q1\n.end\n";
    std::stringstream ss{circuitReal};
    EXPECT_THROW(qc->import(ss, qc::Format::Real), qc::QFRException);
}

TEST_F(IO, insufficientRegistersQelib) {
    const std::string circuitQasm = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[2];\ncx q[0];\n";
    std::stringstream ss{circuitQasm};
    EXPECT_THROW(qc->import(ss, qc::Format::OpenQASM), qasm::QASMParserException);
}

TEST_F(IO, insufficientRegistersEnhancedQelib) {
    const std::string circuitQasm = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[4];\ncccz q[0], q[1], q[2];\n";
    std::stringstream ss{circuitQasm};
    EXPECT_THROW(qc->import(ss, qc::Format::OpenQASM), qasm::QASMParserException);
}

TEST_F(IO, superfluousRegistersQelib) {
    const std::string circuitQasm = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[3];\ncx q[0], q[1], q[2];\n";
    std::stringstream ss{circuitQasm};
    EXPECT_THROW(qc->import(ss, qc::Format::OpenQASM), qasm::QASMParserException);
}

TEST_F(IO, superfluousRegistersEnhancedQelib) {
    const std::string circuitQasm = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[5];\ncccz q[0], q[1], q[2], q[3], q[4];\n";
    std::stringstream ss{circuitQasm};
    EXPECT_THROW(qc->import(ss, qc::Format::OpenQASM), qasm::QASMParserException);
}

TEST_F(IO, dumpNegativeControl) {
    const std::string circuitReal = ".numvars 2\n.variables a b\n.begin\nt2 -a b\n.end";
    std::stringstream ss{circuitReal};
    qc->import(ss, qc::Format::Real);
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

TEST_F(IO, qiskitMcxGray) {
    std::stringstream ss{};
    ss << "OPENQASM 2.0;"
       << "include \"qelib1.inc\";"
       << "qreg q[4];"
       << "mcx_gray q[0], q[1], q[2], q[3];"
       << std::endl;
    qc->import(ss, qc::Format::OpenQASM);
    auto& gate = *(qc->begin());
    std::cout << *qc << std::endl;
    EXPECT_EQ(gate->getType(), qc::X);
    EXPECT_EQ(gate->getNcontrols(), 3);
    EXPECT_EQ(gate->getTargets().at(0), 3);
}

TEST_F(IO, qiskitMcxSkipGateDefinition) {
    std::stringstream ss{};
    ss << "OPENQASM 2.0;"
       << "include \"qelib1.inc\";"
       << "qreg q[4];"
       << "gate mcx q0,q1,q2,q3 { cccx q0,q1,q2,q3; }"
       << "mcx q[0], q[1], q[2], q[3];"
       << std::endl;
    qc->import(ss, qc::Format::OpenQASM);
    auto& gate = *(qc->begin());
    std::cout << *qc << std::endl;
    EXPECT_EQ(gate->getType(), qc::X);
    EXPECT_EQ(gate->getNcontrols(), 3);
    EXPECT_EQ(gate->getTargets().at(0), 3);
}

TEST_F(IO, qiskitMcphase) {
    std::stringstream ss{};
    ss << "OPENQASM 2.0;"
       << "include \"qelib1.inc\";"
       << "qreg q[4];"
       << "mcphase(pi) q[0], q[1], q[2], q[3];"
       << std::endl;
    qc->import(ss, qc::Format::OpenQASM);
    auto& gate = *(qc->begin());
    std::cout << *qc << std::endl;
    EXPECT_EQ(gate->getType(), qc::Z);
    EXPECT_EQ(gate->getNcontrols(), 3);
    EXPECT_EQ(gate->getTargets().at(0), 3);
}

TEST_F(IO, qiskitMcphaseInDeclaration) {
    std::stringstream ss{};
    ss << "OPENQASM 2.0;"
       << "include \"qelib1.inc\";"
       << "qreg q[4];"
       << "gate foo q0, q1, q2, q3 { mcphase(pi) q0, q1, q2, q3; }"
       << "foo q[0], q[1], q[2], q[3];"
       << std::endl;
    qc->import(ss, qc::Format::OpenQASM);
    auto&       gate     = *(qc->begin());
    auto*       compound = dynamic_cast<qc::CompoundOperation*>(gate.get());
    const auto& op       = compound->at(0);
    std::cout << *op << std::endl;
    EXPECT_EQ(op->getType(), qc::Z);
    EXPECT_EQ(op->getNcontrols(), 3);
    EXPECT_EQ(op->getTargets().at(0), 3);
}

TEST_F(IO, qiskitMcxRecursive) {
    std::stringstream ss{};
    ss << "OPENQASM 2.0;"
       << "include \"qelib1.inc\";"
       << "qreg q[6];"
       << "qreg anc[1];"
       << "mcx_recursive q[0], q[1], q[2], q[3], q[4];"
       << "mcx_recursive q[0], q[1], q[2], q[3], q[4], q[5], anc[0];"
       << std::endl;
    qc->import(ss, qc::Format::OpenQASM);
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

TEST_F(IO, qiskitMcxVchain) {
    std::stringstream ss{};
    ss << "OPENQASM 2.0;"
       << "include \"qelib1.inc\";"
       << "qreg q[4];"
       << "qreg anc[1];"
       << "mcx_vchain q[0], q[1], q[2], q[3], anc[0];"
       << std::endl;
    qc->import(ss, qc::Format::OpenQASM);
    auto& gate = *(qc->begin());
    std::cout << *qc << std::endl;
    EXPECT_EQ(gate->getType(), qc::X);
    EXPECT_EQ(gate->getNcontrols(), 3);
    EXPECT_EQ(gate->getTargets().at(0), 3);
}

TEST_F(IO, qiskitMcxDuplicateQubit) {
    std::stringstream ss{};
    ss << "OPENQASM 2.0;"
       << "include \"qelib1.inc\";"
       << "qreg q[4];"
       << "qreg anc[1];"
       << "mcx_vchain q[0], q[0], q[2], q[3], anc[0];"
       << std::endl;
    EXPECT_THROW(qc->import(ss, qc::Format::OpenQASM), qasm::QASMParserException);
}

TEST_F(IO, qiskitMcxQubitRegister) {
    std::stringstream ss{};
    ss << "OPENQASM 2.0;"
       << "include \"qelib1.inc\";"
       << "qreg q[4];"
       << "qreg anc[1];"
       << "mcx_vchain q, q[0], q[2], q[3], anc[0];"
       << std::endl;
    EXPECT_THROW(qc->import(ss, qc::Format::OpenQASM), qasm::QASMParserException);
}

TEST_F(IO, tfcInput) {
    qc->import("./circuits/test.tfc");
    std::cout << *qc << std::endl;
}

TEST_F(IO, qcInput) {
    qc->import("./circuits/test.qc");
    std::cout << *qc << std::endl;
}

TEST_F(IO, grcsInput) {
    qc->import("./circuits/grcs/bris_4_40_9_v2.txt");
    std::cout << *qc << std::endl;
    qc->import("./circuits/grcs/inst_4x4_80_9_v2.txt");
    std::cout << *qc << std::endl;
}

TEST_F(IO, classicControlled) {
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
    EXPECT_NO_THROW(qc->import(ss, qc::Format::OpenQASM););
    std::cout << *qc << std::endl;
}

TEST_F(IO, iSWAPDumpIsValid) {
    qc->addQubitRegister(2);
    qc->iswap(0, 1);
    std::cout << *qc << std::endl;
    std::stringstream ss{};
    qc->dumpOpenQASM(ss);
    EXPECT_NO_THROW(qc->import(ss, qc::Format::OpenQASM););
    std::cout << *qc << std::endl;
}

TEST_F(IO, PeresDumpIsValid) {
    qc->addQubitRegister(2);
    qc->peres(0, 1);
    std::cout << *qc << std::endl;
    std::stringstream ss{};
    qc->dumpOpenQASM(ss);
    EXPECT_NO_THROW(qc->import(ss, qc::Format::OpenQASM););
    std::cout << *qc << std::endl;
}

TEST_F(IO, PeresdagDumpIsValid) {
    qc->addQubitRegister(2);
    qc->peresdag(0, 1);
    std::cout << *qc << std::endl;
    std::stringstream ss{};
    qc->dumpOpenQASM(ss);
    EXPECT_NO_THROW(qc->import(ss, qc::Format::OpenQASM););
    std::cout << *qc << std::endl;
}

TEST_F(IO, printingNonUnitary) {
    std::stringstream ss{};
    ss << "OPENQASM 2.0;"
       << "include \"qelib1.inc\";"
       << "qreg q[2];"
       << "creg c[2];"
       << "h q[0];"
       << "reset q[0];"
       << "h q[0];"
       << "snapshot(1) q[0];"
       << "barrier q;"
       << "measure q -> c;"
       << std::endl;
    EXPECT_NO_THROW(qc->import(ss, qc::Format::OpenQASM));
    std::cout << *qc << std::endl;
    for (const auto& op: *qc) {
        op->print(std::cout);
        std::cout << std::endl;
    }
}

TEST_F(IO, sxAndSxdag) {
    std::stringstream ss{};
    ss << "OPENQASM 2.0;"
       << "include \"qelib1.inc\";"
       << "qreg q[1];"
       << "creg c[1];"
       << "gate test q0 { sx q0; sxdg q0;}"
       << "sx q[0];"
       << "sxdg q[0];"
       << "test q[0];"
       << std::endl;
    EXPECT_NO_THROW(qc->import(ss, qc::Format::OpenQASM));
    std::cout << *qc << std::endl;
    auto& op1 = *(qc->begin());
    EXPECT_EQ(op1->getType(), qc::OpType::SX);
    auto& op2 = *(++qc->begin());
    EXPECT_EQ(op2->getType(), qc::OpType::SXdag);
    auto& op3 = *(++(++qc->begin()));
    ASSERT_TRUE(op3->isCompoundOperation());
    auto* compOp  = dynamic_cast<qc::CompoundOperation*>(op3.get());
    auto& compOp1 = *(compOp->begin());
    EXPECT_EQ(compOp1->getType(), qc::OpType::SX);
    auto& compOp2 = *(++compOp->begin());
    EXPECT_EQ(compOp2->getType(), qc::OpType::SXdag);
}

TEST_F(IO, unifyRegisters) {
    std::stringstream ss{};
    ss << "OPENQASM 2.0;"
       << "include \"qelib1.inc\";"
       << "qreg q[1];"
       << "qreg r[1];"
       << "x q[0];"
       << "x r[0];"
       << std::endl;
    qc->import(ss, qc::Format::OpenQASM);
    std::cout << *qc << std::endl;
    qc->unifyQuantumRegisters();
    std::cout << *qc << std::endl;
    std::ostringstream oss{};
    qc->dump(oss, qc::Format::OpenQASM);
    EXPECT_STREQ(oss.str().c_str(),
                 "// i 0 1\n"
                 "// o 0 1\n"
                 "OPENQASM 2.0;\n"
                 "include \"qelib1.inc\";\n"
                 "qreg q[2];\n"
                 "x q[0];\n"
                 "x q[1];\n");
}

TEST_F(IO, appendMeasurementsAccordingToOutputPermutation) {
    std::stringstream ss{};
    ss << "// o 1\n"
       << "OPENQASM 2.0;"
       << "include \"qelib1.inc\";"
       << "qreg q[2];"
       << "x q[1];"
       << std::endl;
    qc->import(ss, qc::Format::OpenQASM);
    qc->appendMeasurementsAccordingToOutputPermutation();
    std::cout << *qc << std::endl;
    const auto& op = *(qc->rbegin());
    ASSERT_EQ(op->getType(), qc::OpType::Measure);
    const auto& meas = dynamic_cast<const qc::NonUnitaryOperation*>(op.get());
    EXPECT_EQ(meas->getTargets().size(), 1U);
    EXPECT_EQ(meas->getTargets().front(), 1U);
    EXPECT_EQ(meas->getClassics().size(), 1U);
    EXPECT_EQ(meas->getClassics().front(), 0U);
}

TEST_F(IO, appendMeasurementsAccordingToOutputPermutationAugmentRegister) {
    std::stringstream ss{};
    ss << "// o 0 1\n"
       << "OPENQASM 2.0;"
       << "include \"qelib1.inc\";"
       << "qreg q[2];"
       << "creg c[1];"
       << "x q;"
       << std::endl;
    qc->import(ss, qc::Format::OpenQASM);
    qc->appendMeasurementsAccordingToOutputPermutation();
    std::cout << *qc << std::endl;
    EXPECT_EQ(qc->getNcbits(), 2U);
    const auto& op = *(qc->rbegin());
    ASSERT_EQ(op->getType(), qc::OpType::Measure);
    const auto& meas = dynamic_cast<const qc::NonUnitaryOperation*>(op.get());
    EXPECT_EQ(meas->getTargets().size(), 1U);
    EXPECT_EQ(meas->getTargets().front(), 1U);
    EXPECT_EQ(meas->getClassics().size(), 1U);
    EXPECT_EQ(meas->getClassics().front(), 1U);
    const auto& op2 = *(++qc->rbegin());
    ASSERT_EQ(op2->getType(), qc::OpType::Measure);
    const auto& meas2 = dynamic_cast<const qc::NonUnitaryOperation*>(op2.get());
    EXPECT_EQ(meas2->getTargets().size(), 1U);
    EXPECT_EQ(meas2->getTargets().front(), 0U);
    EXPECT_EQ(meas2->getClassics().size(), 1U);
    EXPECT_EQ(meas2->getClassics().front(), 0U);
    std::ostringstream oss{};
    qc->dump(oss, qc::Format::OpenQASM);
    std::cout << oss.str() << std::endl;
    EXPECT_STREQ(oss.str().c_str(),
                 "// i 0 1\n"
                 "// o 0 1\n"
                 "OPENQASM 2.0;\n"
                 "include \"qelib1.inc\";\n"
                 "qreg q[2];\n"
                 "creg c[2];\n"
                 "x q[0];\n"
                 "x q[1];\n"
                 "barrier q;\n"
                 "measure q[0] -> c[0];\n"
                 "measure q[1] -> c[1];\n");
}

TEST_F(IO, appendMeasurementsAccordingToOutputPermutationAddRegister) {
    std::stringstream ss{};
    ss << "// o 0 1\n"
       << "OPENQASM 2.0;"
       << "include \"qelib1.inc\";"
       << "qreg q[2];"
       << "creg d[1];"
       << "x q;"
       << std::endl;
    qc->import(ss, qc::Format::OpenQASM);
    qc->appendMeasurementsAccordingToOutputPermutation();
    std::cout << *qc << std::endl;
    EXPECT_EQ(qc->getNcbits(), 2U);
    const auto& op = *(qc->rbegin());
    ASSERT_EQ(op->getType(), qc::OpType::Measure);
    const auto& meas = dynamic_cast<const qc::NonUnitaryOperation*>(op.get());
    EXPECT_EQ(meas->getTargets().size(), 1U);
    EXPECT_EQ(meas->getTargets().front(), 1U);
    EXPECT_EQ(meas->getClassics().size(), 1U);
    EXPECT_EQ(meas->getClassics().front(), 1U);
    const auto& op2 = *(++qc->rbegin());
    ASSERT_EQ(op2->getType(), qc::OpType::Measure);
    const auto& meas2 = dynamic_cast<const qc::NonUnitaryOperation*>(op2.get());
    EXPECT_EQ(meas2->getTargets().size(), 1U);
    EXPECT_EQ(meas2->getTargets().front(), 0U);
    EXPECT_EQ(meas2->getClassics().size(), 1U);
    EXPECT_EQ(meas2->getClassics().front(), 0U);
    std::ostringstream oss{};
    qc->dump(oss, qc::Format::OpenQASM);
    std::cout << oss.str() << std::endl;
    EXPECT_STREQ(oss.str().c_str(),
                 "// i 0 1\n"
                 "// o 0 1\n"
                 "OPENQASM 2.0;\n"
                 "include \"qelib1.inc\";\n"
                 "qreg q[2];\n"
                 "creg d[1];\n"
                 "creg c[1];\n"
                 "x q[0];\n"
                 "x q[1];\n"
                 "barrier q;\n"
                 "measure q[0] -> d[0];\n"
                 "measure q[1] -> c[0];\n");
}
