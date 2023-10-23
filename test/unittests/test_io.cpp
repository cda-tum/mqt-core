#include "QuantumComputation.hpp"

#include <filesystem>
#include <gtest/gtest.h>
#include <iostream>
#include <string>

class IO : public testing::TestWithParam<std::tuple<std::string, qc::Format>> {
protected:
  void TearDown() override {}

  void SetUp() override { qc = std::make_unique<qc::QuantumComputation>(); }

  std::size_t nqubits = 0;
  unsigned int seed = 0;
  std::string output = "tmp.txt";
  std::string output2 = "tmp2.txt";
  std::string output3 = "tmp";
  std::string output4 = "tmp.tmp.qasm";
  std::string output5 = "./tmpdir/circuit.qasm";
  std::string output5dir = "tmpdir";
  std::unique_ptr<qc::QuantumComputation> qc;
};

void compareFiles(const std::string& file1, const std::string& file2,
                  bool stripWhitespaces = false) {
  std::ifstream fstream1(file1);
  std::string str1((std::istreambuf_iterator<char>(fstream1)),
                   std::istreambuf_iterator<char>());
  std::ifstream fstream2(file2);
  std::string str2((std::istreambuf_iterator<char>(fstream2)),
                   std::istreambuf_iterator<char>());
  if (stripWhitespaces) {
    str1.erase(std::remove_if(str1.begin(), str1.end(), isspace), str1.end());
    str2.erase(std::remove_if(str2.begin(), str2.end(), isspace), str2.end());
  }
  ASSERT_EQ(str1, str2);
}

INSTANTIATE_TEST_SUITE_P(
    IO, IO,
    testing::Values(std::make_tuple(
        "./circuits/test.qasm",
        qc::Format::OpenQASM)), // std::make_tuple("circuits/test.real",
                                // qc::Format::Real
    [](const testing::TestParamInfo<IO::ParamType>& inf) {
      const qc::Format format = std::get<1>(inf.param);

      switch (format) {
      case qc::Format::Real:
        return "Real";
      case qc::Format::OpenQASM:
        return "OpenQasm";
      case qc::Format::GRCS:
        return "GRCS";
      default:
        return "Unknown format";
      }
    });

TEST_P(IO, importAndDump) {
  const auto& [input, format] = GetParam();
  std::cout << "FILE: " << input << "\n";

  ASSERT_NO_THROW(qc->import(input, format));
  ASSERT_NO_THROW(qc->dump(output, format));
  ASSERT_NO_THROW(qc->reset());
  ASSERT_NO_THROW(qc->import(output, format));
  ASSERT_NO_THROW(qc->dump(output2, format));

  compareFiles(output, output2, true);
  std::filesystem::remove(output);
  std::filesystem::remove(output2);
}

TEST_F(IO, dumpValidFilenames) {
  
  qc::QuantumComputation qc{4, 4};
  qc.x(0);
  qc.cx(qc::Control{3}, 2);
  ASSERT_NO_THROW(qc.dump(output3, qc::Format::OpenQASM));
  ASSERT_NO_THROW(qc.dump(output4, qc::Format::OpenQASM));
  ASSERT_NO_THROW(qc.dump(output4));
  
  std::filesystem::create_directory(output5dir);
  ASSERT_NO_THROW(qc.dump(output5, qc::Format::OpenQASM));
  ASSERT_NO_THROW(qc.dump(output5));
  
  std::filesystem::remove(output3);
  std::filesystem::remove(output4);
  std::filesystem::remove(output5);
  std::filesystem::remove(output5dir);
}

TEST_F(IO, importFromString) {
  const std::string bellCircuitQasm =
      "qreg q[2];\nU(pi/2,0,pi) q[0];\nCX q[0],q[1];\n";
  std::stringstream ss{bellCircuitQasm};
  ASSERT_NO_THROW(qc->import(ss, qc::Format::OpenQASM));
  std::cout << *qc << "\n";
  const std::string bellCircuitReal =
      ".numvars 2\n.variables q0 q1\n.begin\nh1 q0\nt2 q0 q1\n.end\n";
  ss.clear();
  ss.str(bellCircuitReal);
  ASSERT_NO_THROW(qc->import(ss, qc::Format::Real));
  std::cout << *qc << "\n";
}

TEST_F(IO, controlledOpActingOnWholeRegister) {
  const std::string circuitQasm = "qreg q[2];\ncx q,q[1];\n";
  std::stringstream ss{circuitQasm};
  EXPECT_THROW(qc->import(ss, qc::Format::OpenQASM), std::runtime_error);
}

TEST_F(IO, invalidRealHeader) {
  const std::string circuitReal =
      ".numvars 2\nvariables q0 q1\n.begin\nh1 q0\nt2 q0 q1\n.end\n";
  std::stringstream ss{circuitReal};
  EXPECT_THROW(qc->import(ss, qc::Format::Real), qc::QFRException);
}

TEST_F(IO, invalidRealCommand) {
  const std::string circuitReal =
      ".numvars 2\n.var q0 q1\n.begin\nh1 q0\n# test comment\nt2 q0 q1\n.end\n";
  std::stringstream ss{circuitReal};
  EXPECT_THROW(qc->import(ss, qc::Format::Real), qc::QFRException);
}

TEST_F(IO, insufficientRegistersQelib) {
  const std::string circuitQasm = "qreg q[2];\ncx q[0];\n";
  std::stringstream ss{circuitQasm};
  EXPECT_THROW(qc->import(ss, qc::Format::OpenQASM), std::runtime_error);
}

TEST_F(IO, insufficientRegistersEnhancedQelib) {
  const std::string circuitQasm = "qreg q[4];\ncccz q[0], q[1], q[2];\n";
  std::stringstream ss{circuitQasm};
  EXPECT_THROW(qc->import(ss, qc::Format::OpenQASM), std::runtime_error);
}

TEST_F(IO, superfluousRegistersQelib) {
  const std::string circuitQasm = "qreg q[3];\ncx q[0], q[1], q[2];\n";
  std::stringstream ss{circuitQasm};
  EXPECT_THROW(qc->import(ss, qc::Format::OpenQASM), std::runtime_error);
}

TEST_F(IO, superfluousRegistersEnhancedQelib) {
  const std::string circuitQasm =
      "qreg q[5];\ncccz q[0], q[1], q[2], q[3], q[4];\n";
  std::stringstream ss{circuitQasm};
  EXPECT_THROW(qc->import(ss, qc::Format::OpenQASM), std::runtime_error);
}

TEST_F(IO, dumpNegativeControl) {
  const std::string circuitReal =
      ".numvars 2\n.variables a b\n.begin\nt2 -a b\n.end";
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
  std::filesystem::remove("testdump.qasm");
}

TEST_F(IO, qiskitMcxGray) {
  std::stringstream ss{};
  ss << "qreg q[4];"
     << "mcx_gray q[0], q[1], q[2], q[3];\n";
  qc->import(ss, qc::Format::OpenQASM);
  auto& gate = *(qc->begin());
  std::cout << *qc << "\n";
  EXPECT_EQ(gate->getType(), qc::X);
  EXPECT_EQ(gate->getNcontrols(), 3);
  EXPECT_EQ(gate->getTargets().at(0), 3);
}

TEST_F(IO, qiskitMcxSkipGateDefinition) {
  std::stringstream ss{};
  ss << "qreg q[4];"
     << "gate mcx q0,q1,q2,q3 { cccx q0,q1,q2,q3; }"
     << "mcx q[0], q[1], q[2], q[3];\n";
  qc->import(ss, qc::Format::OpenQASM);
  auto& gate = *(qc->begin());
  std::cout << *qc << "\n";
  EXPECT_EQ(gate->getType(), qc::X);
  EXPECT_EQ(gate->getNcontrols(), 3);
  EXPECT_EQ(gate->getTargets().at(0), 3);
}

TEST_F(IO, qiskitMcphase) {
  std::stringstream ss{};
  ss << "qreg q[4];"
     << "mcphase(pi) q[0], q[1], q[2], q[3];\n";
  qc->import(ss, qc::Format::OpenQASM);
  auto& gate = *(qc->begin());
  std::cout << *qc << "\n";
  EXPECT_EQ(gate->getType(), qc::Z);
  EXPECT_EQ(gate->getNcontrols(), 3);
  EXPECT_EQ(gate->getTargets().at(0), 3);
}

TEST_F(IO, qiskitMcphaseInDeclaration) {
  std::stringstream ss{};
  ss << "qreg q[4];"
     << "gate foo q0, q1, q2, q3 { mcphase(pi) q0, q1, q2, q3; }"
     << "foo q[0], q[1], q[2], q[3];\n";
  qc->import(ss, qc::Format::OpenQASM);
  std::cout << *qc << "\n";
  auto& op = *(qc->begin());
  EXPECT_EQ(op->getType(), qc::Z);
  EXPECT_EQ(op->getNcontrols(), 3);
  EXPECT_EQ(op->getTargets().at(0), 3);
}

TEST_F(IO, qiskitMcxRecursive) {
  std::stringstream ss{};
  ss << "qreg q[6];"
     << "qreg anc[1];"
     << "mcx_recursive q[0], q[1], q[2], q[3], q[4];"
     << "mcx_recursive q[0], q[1], q[2], q[3], q[4], q[5], anc[0];"
     << "\n";
  qc->import(ss, qc::Format::OpenQASM);
  auto& gate = *(qc->begin());
  std::cout << *qc << "\n";
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
  ss << "qreg q[4];"
     << "qreg anc[1];"
     << "mcx_vchain q[0], q[1], q[2], q[3], anc[0];\n";
  qc->import(ss, qc::Format::OpenQASM);
  auto& gate = *(qc->begin());
  std::cout << *qc << "\n";
  EXPECT_EQ(gate->getType(), qc::X);
  EXPECT_EQ(gate->getNcontrols(), 3);
  EXPECT_EQ(gate->getTargets().at(0), 3);
}

TEST_F(IO, qiskitMcxRecursiveInDeclaration) {
  std::stringstream ss{};
  ss << "qreg q[7];"
     << "gate foo q0, q1, q2, q3, q4 { mcx_recursive q0, q1, q2, q3, q4; }"
     << "gate bar q0, q1, q2, q3, q4, q5, anc { mcx_recursive q0, q1, q2, q3, "
        "q4, q5, anc; }"
     << "foo q[0], q[1], q[2], q[3], q[4];"
     << "bar q[0], q[1], q[2], q[3], q[4], q[5], q[6];\n";
  qc->import(ss, qc::Format::OpenQASM);
  std::cout << *qc << "\n";
  const auto& op = qc->at(0);
  EXPECT_EQ(op->getType(), qc::X);
  EXPECT_EQ(op->getNcontrols(), 4);
  EXPECT_EQ(op->getTargets().at(0), 4);
  const auto& second = qc->at(1);
  EXPECT_EQ(second->getType(), qc::X);
  EXPECT_EQ(second->getNcontrols(), 5);
  EXPECT_EQ(second->getTargets().at(0), 5);
}

TEST_F(IO, qiskitMcxVchainInDeclaration) {
  std::stringstream ss{};
  ss << "qreg q[5];"
     << "gate foo q0, q1, q2, q3, anc { mcx_vchain q0, q1, q2, q3, anc; }"
     << "foo q[0], q[1], q[2], q[3], q[4];\n";
  qc->import(ss, qc::Format::OpenQASM);
  std::cout << *qc << "\n";
  const auto& op = qc->at(0);
  EXPECT_EQ(op->getType(), qc::X);
  EXPECT_EQ(op->getNcontrols(), 3);
  EXPECT_EQ(op->getTargets().at(0), 3);
}

TEST_F(IO, qiskitMcxDuplicateQubit) {
  std::stringstream ss{};
  ss << "qreg q[4];"
     << "qreg anc[1];"
     << "mcx_vchain q[0], q[0], q[2], q[3], anc[0];\n";
  EXPECT_THROW(qc->import(ss, qc::Format::OpenQASM), std::runtime_error);
}

TEST_F(IO, qiskitMcxQubitRegister) {
  std::stringstream ss{};
  ss << "qreg q[4];"
     << "qreg anc[1];"
     << "mcx_vchain q, q[0], q[2], q[3], anc[0];\n";
  EXPECT_THROW(qc->import(ss, qc::Format::OpenQASM), std::runtime_error);
}

TEST_F(IO, barrierInDeclaration) {
  std::stringstream ss{};
  ss << "qreg q[1];"
     << "gate foo q0 { h q0; barrier q0; h q0; }"
     << "foo q[0];\n";
  qc->import(ss, qc::Format::OpenQASM);
  std::cout << *qc << "\n";
  EXPECT_EQ(qc->getNops(), 1);
  const auto& op = qc->at(0);
  EXPECT_EQ(op->getType(), qc::Compound);
  const auto* comp = dynamic_cast<const qc::CompoundOperation*>(op.get());
  ASSERT_NE(comp, nullptr);
  EXPECT_EQ(comp->size(), 2);
  EXPECT_EQ(comp->at(0)->getType(), qc::H);
  EXPECT_EQ(comp->at(1)->getType(), qc::H);
}

TEST_F(IO, CommentInDeclaration) {
  std::stringstream ss{};
  ss << "qreg q[1];"
     << "gate foo q0 { \n"
        "  h q0;\n"
        "  //x q0;\n"
        "  h q0;\n"
        "}\n"
     << "foo q[0];\n";
  qc->import(ss, qc::Format::OpenQASM);
  std::cout << *qc << "\n";
  EXPECT_EQ(qc->getNops(), 1);
  const auto& op = qc->at(0);
  EXPECT_EQ(op->getType(), qc::Compound);
  const auto* comp = dynamic_cast<const qc::CompoundOperation*>(op.get());
  ASSERT_NE(comp, nullptr);
  EXPECT_EQ(comp->size(), 2);
  EXPECT_EQ(comp->at(0)->getType(), qc::H);
  EXPECT_EQ(comp->at(1)->getType(), qc::H);
}

TEST_F(IO, realInput) {
  qc->import("./circuits/test.real");
  std::cout << *qc << "\n";
}

TEST_F(IO, tfcInput) {
  qc->import("./circuits/test.tfc");
  std::cout << *qc << "\n";
}

TEST_F(IO, qcInput) {
  qc->import("./circuits/test.qc");
  std::cout << *qc << "\n";
}

TEST_F(IO, grcsInput) {
  qc->import("./circuits/grcs/bris_4_40_9_v2.txt");
  std::cout << *qc << "\n";
  qc->import("./circuits/grcs/inst_4x4_80_9_v2.txt");
  std::cout << *qc << "\n";
}

TEST_F(IO, classicControlled) {
  std::stringstream ss{};
  ss << "qreg q[1];"
     << "creg c[1];"
     << "h q[0];"
     << "measure q->c;"
     << "// test classic controlled operation\n"
     << "if (c==1) x q[0];\n";
  EXPECT_NO_THROW(qc->import(ss, qc::Format::OpenQASM););
  std::cout << *qc << "\n";
}

TEST_F(IO, iSWAPDumpIsValid) {
  qc->addQubitRegister(2);
  qc->iswap(0, 1);
  std::cout << *qc << "\n";
  std::stringstream ss{};
  qc->dumpOpenQASM(ss);
  EXPECT_NO_THROW(qc->import(ss, qc::Format::OpenQASM););
  std::cout << *qc << "\n";
}

TEST_F(IO, PeresDumpIsValid) {
  qc->addQubitRegister(2);
  qc->peres(0, 1);
  std::cout << *qc << "\n";
  std::stringstream ss{};
  qc->dumpOpenQASM(ss);
  EXPECT_NO_THROW(qc->import(ss, qc::Format::OpenQASM););
  std::cout << *qc << "\n";
}

TEST_F(IO, PeresdagDumpIsValid) {
  qc->addQubitRegister(2);
  qc->peresdg(0, 1);
  std::cout << *qc << "\n";
  std::stringstream ss{};
  qc->dumpOpenQASM(ss);
  EXPECT_NO_THROW(qc->import(ss, qc::Format::OpenQASM););
  std::cout << *qc << "\n";
}

TEST_F(IO, printingNonUnitary) {
  std::stringstream ss{};
  ss << "qreg q[2];"
     << "creg c[2];"
     << "h q[0];"
     << "reset q[0];"
     << "h q[0];"
     << "snapshot(1) q[0];"
     << "barrier q;"
     << "measure q -> c;\n";
  EXPECT_NO_THROW(qc->import(ss, qc::Format::OpenQASM));
  std::cout << *qc << "\n";
  for (const auto& op : *qc) {
    op->print(std::cout);
    std::cout << "\n";
  }
}

TEST_F(IO, sxAndSxdag) {
  std::stringstream ss{};
  ss << "qreg q[1];"
     << "creg c[1];"
     << "gate test q0 { sx q0; sxdg q0;}"
     << "sx q[0];"
     << "sxdg q[0];"
     << "test q[0];\n";
  EXPECT_NO_THROW(qc->import(ss, qc::Format::OpenQASM));
  std::cout << *qc << "\n";
  auto& op1 = *(qc->begin());
  EXPECT_EQ(op1->getType(), qc::OpType::SX);
  auto& op2 = *(++qc->begin());
  EXPECT_EQ(op2->getType(), qc::OpType::SXdg);
  auto& op3 = *(++(++qc->begin()));
  ASSERT_TRUE(op3->isCompoundOperation());
  auto* compOp = dynamic_cast<qc::CompoundOperation*>(op3.get());
  ASSERT_NE(compOp, nullptr);
  auto& compOp1 = *(compOp->begin());
  EXPECT_EQ(compOp1->getType(), qc::OpType::SX);
  auto& compOp2 = *(++compOp->begin());
  EXPECT_EQ(compOp2->getType(), qc::OpType::SXdg);
}

TEST_F(IO, unifyRegisters) {
  std::stringstream ss{};
  ss << "qreg q[1];"
     << "qreg r[1];"
     << "x q[0];"
     << "x r[0];\n";
  qc->import(ss, qc::Format::OpenQASM);
  std::cout << *qc << "\n";
  qc->unifyQuantumRegisters();
  std::cout << *qc << "\n";
  std::ostringstream oss{};
  qc->dump(oss, qc::Format::OpenQASM);
  EXPECT_STREQ(oss.str().c_str(), "// i 0 1\n"
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
     << "qreg q[2];"
     << "x q[1];\n";
  qc->import(ss, qc::Format::OpenQASM);
  qc->appendMeasurementsAccordingToOutputPermutation();
  std::cout << *qc << "\n";
  const auto& op = *(qc->rbegin());
  ASSERT_EQ(op->getType(), qc::OpType::Measure);
  const auto& meas = dynamic_cast<const qc::NonUnitaryOperation*>(op.get());
  ASSERT_NE(meas, nullptr);
  EXPECT_EQ(meas->getTargets().size(), 1U);
  EXPECT_EQ(meas->getTargets().front(), 1U);
  EXPECT_EQ(meas->getClassics().size(), 1U);
  EXPECT_EQ(meas->getClassics().front(), 0U);
}

TEST_F(IO, appendMeasurementsAccordingToOutputPermutationAugmentRegister) {
  std::stringstream ss{};
  ss << "// o 0 1\n"
     << "qreg q[2];"
     << "creg c[1];"
     << "x q;\n";
  qc->import(ss, qc::Format::OpenQASM);
  qc->appendMeasurementsAccordingToOutputPermutation();
  std::cout << *qc << "\n";
  EXPECT_EQ(qc->getNcbits(), 2U);
  const auto& op = *(qc->rbegin());
  ASSERT_EQ(op->getType(), qc::OpType::Measure);
  const auto& meas = dynamic_cast<const qc::NonUnitaryOperation*>(op.get());
  ASSERT_NE(meas, nullptr);
  EXPECT_EQ(meas->getTargets().size(), 1U);
  EXPECT_EQ(meas->getTargets().front(), 1U);
  EXPECT_EQ(meas->getClassics().size(), 1U);
  EXPECT_EQ(meas->getClassics().front(), 1U);
  const auto& op2 = *(++qc->rbegin());
  ASSERT_EQ(op2->getType(), qc::OpType::Measure);
  const auto& meas2 = dynamic_cast<const qc::NonUnitaryOperation*>(op2.get());
  ASSERT_NE(meas2, nullptr);
  EXPECT_EQ(meas2->getTargets().size(), 1U);
  EXPECT_EQ(meas2->getTargets().front(), 0U);
  EXPECT_EQ(meas2->getClassics().size(), 1U);
  EXPECT_EQ(meas2->getClassics().front(), 0U);
  std::ostringstream oss{};
  qc->dump(oss, qc::Format::OpenQASM);
  std::cout << oss.str() << "\n";
  EXPECT_STREQ(oss.str().c_str(), "// i 0 1\n"
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
     << "qreg q[2];"
     << "creg d[1];"
     << "x q;\n";
  qc->import(ss, qc::Format::OpenQASM);
  qc->appendMeasurementsAccordingToOutputPermutation();
  std::cout << *qc << "\n";
  EXPECT_EQ(qc->getNcbits(), 2U);
  const auto& op = *(qc->rbegin());
  ASSERT_EQ(op->getType(), qc::OpType::Measure);
  const auto& meas = dynamic_cast<const qc::NonUnitaryOperation*>(op.get());
  ASSERT_NE(meas, nullptr);
  EXPECT_EQ(meas->getTargets().size(), 1U);
  EXPECT_EQ(meas->getTargets().front(), 1U);
  EXPECT_EQ(meas->getClassics().size(), 1U);
  EXPECT_EQ(meas->getClassics().front(), 1U);
  const auto& op2 = *(++qc->rbegin());
  ASSERT_EQ(op2->getType(), qc::OpType::Measure);
  const auto& meas2 = dynamic_cast<const qc::NonUnitaryOperation*>(op2.get());
  ASSERT_NE(meas2, nullptr);
  EXPECT_EQ(meas2->getTargets().size(), 1U);
  EXPECT_EQ(meas2->getTargets().front(), 0U);
  EXPECT_EQ(meas2->getClassics().size(), 1U);
  EXPECT_EQ(meas2->getClassics().front(), 0U);
  std::ostringstream oss{};
  qc->dump(oss, qc::Format::OpenQASM);
  std::cout << oss.str() << "\n";
  EXPECT_STREQ(oss.str().c_str(), "// i 0 1\n"
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

TEST_F(IO, NativeTwoQubitGateImportAndExport) {
  const auto gates = std::vector<std::string>{"dcx",
                                              "ecr",
                                              "rxx(0.5)",
                                              "ryy(0.5)",
                                              "rzz(0.5)",
                                              "rzx(0.5)",
                                              "xx_minus_yy(0.5,0.5)",
                                              "xx_plus_yy(0.5,0.5)"};

  const std::string header = "// i 0 1\n"
                             "// o 0 1\n"
                             "OPENQASM 2.0;\n"
                             "include \"qelib1.inc\";\n"
                             "qreg q[2];\n";
  for (const auto& gate : gates) {
    std::stringstream ss{};
    ss << header << gate << " q[0], q[1];\n";
    const auto target = ss.str();
    qc->import(ss, qc::Format::OpenQASM);
    std::cout << *qc << "\n";
    std::ostringstream oss{};
    qc->dump(oss, qc::Format::OpenQASM);
    std::cout << oss.str() << "\n";
    EXPECT_STREQ(oss.str().c_str(), target.c_str());
    qc->reset();
    std::cout << "---\n";
  }
}

TEST_F(IO, UseQelib1Gate) {
  std::stringstream ss{};
  ss << "include \"qelib1.inc\";\n"
     << "qreg q[3];\n"
     << "rccx q[0], q[1], q[2];\n";
  qc->import(ss, qc::Format::OpenQASM);
  std::cout << *qc << "\n";
  EXPECT_EQ(qc->getNqubits(), 3U);
  EXPECT_EQ(qc->getNops(), 1U);
  EXPECT_EQ(qc->at(0)->getType(), qc::Compound);
  const auto& op = dynamic_cast<const qc::CompoundOperation*>(qc->at(0).get());
  ASSERT_NE(op, nullptr);
  EXPECT_EQ(op->size(), 9U);
}

TEST_F(IO, ParametrizedGateDefinition) {
  std::stringstream ss{};
  ss << "qreg q[1];\n"
     << "gate foo(theta, beta) q { rz(theta) q; rx(beta) q; }\n"
     << "foo(2*cos(pi/4), 0.5*sin(pi/2)) q[0];\n";
  qc->import(ss, qc::Format::OpenQASM);
  std::cout << *qc << "\n";
  EXPECT_EQ(qc->getNqubits(), 1U);
  EXPECT_EQ(qc->getNops(), 1U);
  EXPECT_EQ(qc->at(0)->getType(), qc::Compound);
  const auto& op = dynamic_cast<const qc::CompoundOperation*>(qc->at(0).get());
  ASSERT_NE(op, nullptr);
  EXPECT_EQ(op->size(), 2U);
  EXPECT_EQ(op->at(0)->getType(), qc::RZ);
  EXPECT_EQ(op->at(1)->getType(), qc::RX);
  const auto& rz = dynamic_cast<const qc::StandardOperation*>(op->at(0).get());
  ASSERT_NE(rz, nullptr);
  const auto& rx = dynamic_cast<const qc::StandardOperation*>(op->at(1).get());
  ASSERT_NE(rx, nullptr);
  EXPECT_EQ(rz->getParameter().at(0), 2 * std::cos(qc::PI_4));
  EXPECT_EQ(rx->getParameter().at(0), 0.5 * std::sin(qc::PI_2));
}

TEST_F(IO, NonExistingInclude) {
  std::stringstream ss{};
  ss << "include \"qelib.inc\";\n";
  EXPECT_THROW(qc->import(ss, qc::Format::OpenQASM), std::runtime_error);
}

TEST_F(IO, NonStandardInclude) {
  std::ofstream ofs{"defs.inc"};
  ofs << "gate foo q { h q; }\n";
  ofs.close();
  std::stringstream ss{};
  ss << "include \"defs.inc\";\n"
     << "qreg q[1];\n"
     << "foo q[0];\n";
  qc->import(ss, qc::Format::OpenQASM);
  std::cout << *qc << "\n";
  EXPECT_EQ(qc->getNqubits(), 1U);
  EXPECT_EQ(qc->getNops(), 1U);
  EXPECT_EQ(qc->at(0)->getType(), qc::H);
  std::filesystem::remove("defs.inc");
}
