/*
 * This file is part of MQT QFR library which is released under the MIT license.
 * See file README.md or go to https://www.cda.cit.tum.de/research/quantum/ for more information.
 */

#include "QuantumComputation.hpp"
#include "dd/FunctionalityConstruction.hpp"
#include "dd/Simulation.hpp"

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

TEST_F(IO, qiskit_mcx_skip_gate_definition) {
    std::stringstream ss{};
    ss << "OPENQASM 2.0;"
       << "include \"qelib1.inc\";"
       << "qreg q[4];"
       << "gate mcx q0,q1,q2,q3 { cccx q0,q1,q2,q3; }"
       << "mcx q[0], q[1], q[2], q[3];"
       << std::endl;
    qc->import(ss, qc::OpenQASM);
    auto& gate = *(qc->begin());
    std::cout << *qc << std::endl;
    EXPECT_EQ(gate->getType(), qc::X);
    EXPECT_EQ(gate->getNcontrols(), 3);
    EXPECT_EQ(gate->getTargets().at(0), 3);
}

TEST_F(IO, qiskit_mcphase) {
    std::stringstream ss{};
    ss << "OPENQASM 2.0;"
       << "include \"qelib1.inc\";"
       << "qreg q[4];"
       << "mcphase(pi) q[0], q[1], q[2], q[3];"
       << std::endl;
    qc->import(ss, qc::OpenQASM);
    auto& gate = *(qc->begin());
    std::cout << *qc << std::endl;
    EXPECT_EQ(gate->getType(), qc::Z);
    EXPECT_EQ(gate->getNcontrols(), 3);
    EXPECT_EQ(gate->getTargets().at(0), 3);
}

TEST_F(IO, qiskit_mcphase_in_declaration) {
    std::stringstream ss{};
    ss << "OPENQASM 2.0;"
       << "include \"qelib1.inc\";"
       << "qreg q[4];"
       << "gate foo q0, q1, q2, q3 { mcphase(pi) q0, q1, q2, q3; }"
       << "foo q[0], q[1], q[2], q[3];"
       << std::endl;
    qc->import(ss, qc::OpenQASM);
    auto&       gate     = *(qc->begin());
    auto*       compound = dynamic_cast<qc::CompoundOperation*>(gate.get());
    const auto& op       = compound->at(0);
    std::cout << *op << std::endl;
    EXPECT_EQ(op->getType(), qc::Z);
    EXPECT_EQ(op->getNcontrols(), 3);
    EXPECT_EQ(op->getTargets().at(0), 3);
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

TEST_F(IO, changePermutation) {
    std::stringstream ss{};
    ss << "// o 1 0\n"
       << "OPENQASM 2.0;"
       << "include \"qelib1.inc\";"
       << "qreg q[2];"
       << "x q[0];"
       << std::endl;
    qc->import(ss, qc::OpenQASM);
    auto dd  = std::make_unique<dd::Package<>>();
    auto sim = simulate(qc.get(), dd->makeZeroState(qc->getNqubits()), dd);
    EXPECT_TRUE(sim.p->e[0].isZeroTerminal());
    EXPECT_TRUE(sim.p->e[1].w.approximatelyOne());
    EXPECT_TRUE(sim.p->e[1].p->e[1].isZeroTerminal());
    EXPECT_TRUE(sim.p->e[1].p->e[0].w.approximatelyOne());
    auto func = buildFunctionality(qc.get(), dd);
    EXPECT_FALSE(func.p->e[0].isZeroTerminal());
    EXPECT_FALSE(func.p->e[1].isZeroTerminal());
    EXPECT_FALSE(func.p->e[2].isZeroTerminal());
    EXPECT_FALSE(func.p->e[3].isZeroTerminal());
    EXPECT_TRUE(func.p->e[0].p->e[1].w.approximatelyOne());
    EXPECT_TRUE(func.p->e[1].p->e[3].w.approximatelyOne());
    EXPECT_TRUE(func.p->e[2].p->e[0].w.approximatelyOne());
    EXPECT_TRUE(func.p->e[3].p->e[2].w.approximatelyOne());
}

TEST_F(IO, iSWAP_dump_is_valid) {
    qc->addQubitRegister(2);
    qc->iswap(0, 1);
    std::cout << *qc << std::endl;
    std::stringstream ss{};
    qc->dumpOpenQASM(ss);
    EXPECT_NO_THROW(qc->import(ss, qc::OpenQASM););
    std::cout << *qc << std::endl;
}

TEST_F(IO, Peres_dump_is_valid) {
    qc->addQubitRegister(2);
    qc->peres(0, 1);
    std::cout << *qc << std::endl;
    std::stringstream ss{};
    qc->dumpOpenQASM(ss);
    EXPECT_NO_THROW(qc->import(ss, qc::OpenQASM););
    std::cout << *qc << std::endl;
}

TEST_F(IO, Peresdag_dump_is_valid) {
    qc->addQubitRegister(2);
    qc->peresdag(0, 1);
    std::cout << *qc << std::endl;
    std::stringstream ss{};
    qc->dumpOpenQASM(ss);
    EXPECT_NO_THROW(qc->import(ss, qc::OpenQASM););
    std::cout << *qc << std::endl;
}

TEST_F(IO, printing_non_unitary) {
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
    EXPECT_NO_THROW(qc->import(ss, qc::OpenQASM));
    std::cout << *qc << std::endl;
    for (const auto& op: *qc) {
        op->print(std::cout);
        std::cout << std::endl;
    }
}

TEST_F(IO, sx_and_sxdag) {
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
    EXPECT_NO_THROW(qc->import(ss, qc::OpenQASM));
    std::cout << *qc << std::endl;
    auto& op1 = *(qc->begin());
    EXPECT_EQ(op1->getType(), qc::OpType::SX);
    auto& op2 = *(++qc->begin());
    EXPECT_EQ(op2->getType(), qc::OpType::SXdag);
    auto& op3 = *(++(++qc->begin()));
    ASSERT_TRUE(op3->isCompoundOperation());
    auto  compOp  = dynamic_cast<qc::CompoundOperation*>(op3.get());
    auto& compOp1 = *(compOp->begin());
    EXPECT_EQ(compOp1->getType(), qc::OpType::SX);
    auto& compOp2 = *(++compOp->begin());
    EXPECT_EQ(compOp2->getType(), qc::OpType::SXdag);
}

TEST_F(IO, unify_registers) {
    std::stringstream ss{};
    ss << "OPENQASM 2.0;"
       << "include \"qelib1.inc\";"
       << "qreg q[1];"
       << "qreg r[1];"
       << "x q[0];"
       << "x r[0];"
       << std::endl;
    qc->import(ss, qc::OpenQASM);
    std::cout << *qc << std::endl;
    qc->unifyQuantumRegisters();
    std::cout << *qc << std::endl;
    std::ostringstream oss{};
    qc->dump(oss, qc::OpenQASM);
    EXPECT_STREQ(oss.str().c_str(),
                 "// i 0 1\n"
                 "// o 0 1\n"
                 "OPENQASM 2.0;\n"
                 "include \"qelib1.inc\";\n"
                 "qreg q[2];\n"
                 "x q[0];\n"
                 "x q[1];\n");
}

TEST_F(IO, append_measurements_according_to_output_permutation) {
    std::stringstream ss{};
    ss << "// o 1\n"
       << "OPENQASM 2.0;"
       << "include \"qelib1.inc\";"
       << "qreg q[2];"
       << "x q[1];"
       << std::endl;
    qc->import(ss, qc::OpenQASM);
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

TEST_F(IO, append_measurements_according_to_output_permutation_augment_register) {
    std::stringstream ss{};
    ss << "// o 0 1\n"
       << "OPENQASM 2.0;"
       << "include \"qelib1.inc\";"
       << "qreg q[2];"
       << "creg c[1];"
       << "x q;"
       << std::endl;
    qc->import(ss, qc::OpenQASM);
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
    qc->dump(oss, qc::OpenQASM);
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
                 "measure q[0] -> c[0];\n"
                 "measure q[1] -> c[1];\n");
}

TEST_F(IO, append_measurements_according_to_output_permutation_add_register) {
    std::stringstream ss{};
    ss << "// o 0 1\n"
       << "OPENQASM 2.0;"
       << "include \"qelib1.inc\";"
       << "qreg q[2];"
       << "creg d[1];"
       << "x q;"
       << std::endl;
    qc->import(ss, qc::OpenQASM);
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
    qc->dump(oss, qc::OpenQASM);
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
                 "measure q[0] -> d[0];\n"
                 "measure q[1] -> c[0];\n");
}
