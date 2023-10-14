#include "CircuitOptimizer.hpp"
#include "QuantumComputation.hpp"
#include "algorithms/RandomCliffordCircuit.hpp"

#include "gtest/gtest.h"
#include <iostream>
#include <random>

using namespace qc;

class QFRFunctionality : public testing::TestWithParam<std::size_t> {
protected:
  void SetUp() override {
    std::array<std::mt19937_64::result_type, std::mt19937_64::state_size>
        randomData{};
    std::random_device rd;
    std::generate(begin(randomData), end(randomData), [&]() { return rd(); });
    std::seed_seq seeds(begin(randomData), end(randomData));
    mt.seed(seeds);
    dist = std::uniform_real_distribution<fp>(0.0, 2 * PI);
  }

  std::mt19937_64 mt;
  std::uniform_real_distribution<fp> dist;
};

TEST_F(QFRFunctionality, fuseCxToSwap) {
  const std::size_t nqubits = 2;
  QuantumComputation qc(nqubits);
  qc.cx(0, 1);
  qc.cx(1, 0);
  qc.cx(0, 1);
  CircuitOptimizer::swapReconstruction(qc);
  const auto& op = qc.front();
  EXPECT_TRUE(op->isStandardOperation());
  EXPECT_EQ(op->getType(), SWAP);
  EXPECT_EQ(op->getTargets().at(0), 0);
  EXPECT_EQ(op->getTargets().at(1), 1);
}

TEST_F(QFRFunctionality, replaceCxToSwapAtEnd) {
  const std::size_t nqubits = 2;
  QuantumComputation qc(nqubits);
  qc.cx(0, 1);
  qc.cx(1, 0);
  CircuitOptimizer::swapReconstruction(qc);
  auto it = qc.begin();
  const auto& op = *it;
  EXPECT_TRUE(op->isStandardOperation());
  EXPECT_EQ(op->getType(), SWAP);
  EXPECT_EQ(op->getTargets().at(0), 0);
  EXPECT_EQ(op->getTargets().at(1), 1);

  ++it;
  const auto& op2 = *it;
  EXPECT_TRUE(op2->isStandardOperation());
  EXPECT_EQ(op2->getType(), X);
  EXPECT_EQ(op2->getControls().begin()->qubit, 0);
  EXPECT_EQ(op2->getTargets().at(0), 1);
}

TEST_F(QFRFunctionality, replaceCxToSwap) {
  const std::size_t nqubits = 2;
  QuantumComputation qc(nqubits);
  qc.cx(0, 1);
  qc.cx(1, 0);
  qc.h(0);
  CircuitOptimizer::swapReconstruction(qc);
  auto it = qc.begin();
  const auto& op = *it;
  EXPECT_TRUE(op->isStandardOperation());
  EXPECT_EQ(op->getType(), SWAP);
  EXPECT_EQ(op->getTargets().at(0), 0);
  EXPECT_EQ(op->getTargets().at(1), 1);
  ++it;
  const auto& op2 = *it;
  EXPECT_TRUE(op2->isStandardOperation());
  EXPECT_EQ(op2->getType(), X);
  EXPECT_EQ(op2->getControls().begin()->qubit, 0);
  EXPECT_EQ(op2->getTargets().at(0), 1);
}

TEST_F(QFRFunctionality, removeTrailingIdleQubits) {
  const std::size_t nqubits = 4;
  QuantumComputation qc(nqubits, nqubits);
  qc.x(0);
  qc.x(2);
  std::cout << qc;
  qc::QuantumComputation::printPermutation(qc.outputPermutation);
  qc.printRegisters();

  qc.outputPermutation.erase(1);
  qc.outputPermutation.erase(3);

  qc.stripIdleQubits();
  EXPECT_EQ(qc.getNqubits(), 2);
  std::cout << qc;
  qc::QuantumComputation::printPermutation(qc.outputPermutation);
  qc.printRegisters();

  qc.pop_back();
  qc.outputPermutation.erase(2);
  std::cout << qc;
  qc::QuantumComputation::printPermutation(qc.outputPermutation);
  qc.printRegisters();

  qc.stripIdleQubits();
  EXPECT_EQ(qc.getNqubits(), 1);
}

TEST_F(QFRFunctionality, ancillaryQubitAtEnd) {
  const std::size_t nqubits = 2;
  QuantumComputation qc(nqubits);
  qc.x(0);
  qc.addAncillaryRegister(1);
  EXPECT_EQ(qc.getNancillae(), 1);
  EXPECT_EQ(qc.getNqubitsWithoutAncillae(), nqubits);
  EXPECT_EQ(qc.getNqubits(), 3);
  qc.x(2);
  qc.printRegisters();
  auto p = qc.removeQubit(2);
  EXPECT_EQ(p.first, nqubits);
  EXPECT_EQ(p.second, nqubits);
  EXPECT_EQ(qc.getNancillae(), 0);
  EXPECT_EQ(qc.getNqubitsWithoutAncillae(), nqubits);
  EXPECT_EQ(qc.getNqubits(), nqubits);
  EXPECT_TRUE(qc.getANCregs().empty());
  qc.printRegisters();
  qc.addAncillaryQubit(p.first, p.second);
  EXPECT_EQ(qc.getNancillae(), 1);
  EXPECT_EQ(qc.getNqubitsWithoutAncillae(), nqubits);
  EXPECT_EQ(qc.getNqubits(), nqubits + 1);
  EXPECT_FALSE(qc.getANCregs().empty());
  qc.printRegisters();
  auto q = qc.removeQubit(2);
  EXPECT_EQ(q.first, nqubits);
  EXPECT_EQ(q.second, nqubits);
  EXPECT_EQ(qc.getNancillae(), 0);
  EXPECT_EQ(qc.getNqubitsWithoutAncillae(), nqubits);
  EXPECT_EQ(qc.getNqubits(), nqubits);
  EXPECT_TRUE(qc.getANCregs().empty());
  qc.printRegisters();
  auto rm = qc.removeQubit(1);
  EXPECT_EQ(rm.first, 1);
  EXPECT_EQ(rm.second, 1);
  EXPECT_EQ(qc.getNancillae(), 0);
  EXPECT_EQ(qc.getNqubitsWithoutAncillae(), 1);
  EXPECT_EQ(qc.getNqubits(), 1);
  qc.printRegisters();
  auto empty = qc.removeQubit(0);
  EXPECT_EQ(empty.first, 0);
  EXPECT_EQ(empty.second, 0);
  EXPECT_EQ(qc.getNancillae(), 0);
  EXPECT_EQ(qc.getNqubitsWithoutAncillae(), 0);
  EXPECT_EQ(qc.getNqubits(), 0);
  EXPECT_TRUE(qc.getQregs().empty());
  qc.printRegisters();
  qc.printStatistics(std::cout);
}

TEST_F(QFRFunctionality, ancillaryQubitRemoveMiddle) {
  const std::size_t nqubits = 2;
  QuantumComputation qc(nqubits);
  qc.x(0);
  qc.addAncillaryRegister(3);
  auto p = qc.removeQubit(3);
  EXPECT_EQ(p.first, 3);
  EXPECT_EQ(p.second, 3);
  EXPECT_EQ(qc.getNancillae(), 2);
  EXPECT_EQ(qc.getNqubitsWithoutAncillae(), 2);
  EXPECT_EQ(qc.getNqubits(), 4);
  qc.printRegisters();
}

TEST_F(QFRFunctionality, splitQreg) {
  const std::size_t nqubits = 3;
  QuantumComputation qc(nqubits);
  qc.x(0);
  auto p = qc.removeQubit(1);
  EXPECT_EQ(p.first, 1);
  EXPECT_EQ(p.second, 1);
  EXPECT_EQ(qc.getNancillae(), 0);
  EXPECT_EQ(qc.getNqubitsWithoutAncillae(), 2);
  EXPECT_EQ(qc.getNqubits(), 2);
  qc.printRegisters();
}

TEST_F(QFRFunctionality, StripIdleAndDump) {
  std::stringstream ss{};
  const std::string testfile = "OPENQASM 2.0;\n"
                               "include \"qelib1.inc\";\n"
                               "qreg q[5];\n"
                               "creg c[3];\n"
                               "x q[0];\n"
                               "x q[2];\n"
                               "barrier q;\n"
                               "barrier q[0];\n"
                               "reset q;\n"
                               "reset q[2];\n"
                               "cx q[0],q[4];\n";

  ss << testfile;
  auto qc = qc::QuantumComputation();
  qc.import(ss, qc::Format::OpenQASM);
  qc.print(std::cout);
  qc.stripIdleQubits();
  qc.print(std::cout);
  std::stringstream goal{};
  qc.print(goal);
  std::stringstream test{};
  qc.dump(test, qc::Format::OpenQASM);
  std::cout << test.str() << "\n";
  qc.reset();
  qc.import(test, qc::Format::OpenQASM);
  qc.print(std::cout);
  qc.stripIdleQubits();
  qc.print(std::cout);
  std::stringstream actual{};
  qc.print(actual);
  EXPECT_EQ(goal.str(), actual.str());
}

TEST_F(QFRFunctionality, CollapseCompoundOperationToStandard) {
  const std::size_t nqubits = 1;
  QuantumComputation qc(nqubits);
  qc.x(0);
  qc.i(0);
  std::cout << "-----------------------------\n";
  qc.print(std::cout);
  CircuitOptimizer::singleQubitGateFusion(qc);
  std::cout << "-----------------------------\n";
  qc.print(std::cout);
  EXPECT_EQ(qc.getNops(), 1);
  EXPECT_TRUE(qc.begin()->get()->isStandardOperation());
}

TEST_F(QFRFunctionality, eliminateCompoundOperation) {
  const std::size_t nqubits = 1;
  QuantumComputation qc(nqubits);
  qc.i(0);
  qc.i(0);
  std::cout << "-----------------------------\n";
  qc.print(std::cout);
  CircuitOptimizer::singleQubitGateFusion(qc);
  std::cout << "-----------------------------\n";
  qc.print(std::cout);
  EXPECT_EQ(qc.getNops(), 0);
  EXPECT_TRUE(qc.empty());
}

TEST_F(QFRFunctionality, eliminateInverseInCompoundOperation) {
  const std::size_t nqubits = 1;
  QuantumComputation qc(nqubits);
  qc.s(0);
  qc.sdg(0);
  std::cout << "-----------------------------\n";
  qc.print(std::cout);
  CircuitOptimizer::singleQubitGateFusion(qc);
  std::cout << "-----------------------------\n";
  qc.print(std::cout);
  EXPECT_EQ(qc.getNops(), 0);
  EXPECT_TRUE(qc.empty());
}

TEST_F(QFRFunctionality, unknownInverseInCompoundOperation) {
  const std::size_t nqubits = 1;
  QuantumComputation qc(nqubits);
  qc.p(1., 0);
  qc.p(-1., 0);
  std::cout << "-----------------------------\n";
  qc.print(std::cout);
  CircuitOptimizer::singleQubitGateFusion(qc);
  std::cout << "-----------------------------\n";
  qc.print(std::cout);
  EXPECT_EQ(qc.getNops(), 1);
}

TEST_F(QFRFunctionality, removeDiagonalSingleQubitBeforeMeasure) {
  const std::size_t nqubits = 1;
  QuantumComputation qc(nqubits, nqubits);
  qc.z(0);
  qc.measure(0, 0);
  std::cout << "-----------------------------\n";
  qc.print(std::cout);
  CircuitOptimizer::removeDiagonalGatesBeforeMeasure(qc);
  std::cout << "-----------------------------\n";
  qc.print(std::cout);
  EXPECT_EQ(qc.getNops(), 1);
  EXPECT_EQ(qc.begin()->get()->getType(), qc::Measure);
}

TEST_F(QFRFunctionality, removeDiagonalCompoundOpBeforeMeasure) {
  const std::size_t nqubits = 1;
  QuantumComputation qc(nqubits, nqubits);
  qc.z(0);
  qc.t(0);
  qc.measure(0, 0);
  std::cout << "-----------------------------\n";
  qc.print(std::cout);
  CircuitOptimizer::singleQubitGateFusion(qc);
  CircuitOptimizer::removeDiagonalGatesBeforeMeasure(qc);
  std::cout << "-----------------------------\n";
  qc.print(std::cout);
  EXPECT_EQ(qc.getNops(), 1);
  EXPECT_EQ(qc.begin()->get()->getType(), qc::Measure);
}

TEST_F(QFRFunctionality, removeDiagonalTwoQubitGateBeforeMeasure) {
  const std::size_t nqubits = 2;
  QuantumComputation qc(nqubits, nqubits);
  qc.cz(0, 1);
  qc.measure({0, 1}, {0, 1});
  std::cout << "-----------------------------\n";
  qc.print(std::cout);
  CircuitOptimizer::removeDiagonalGatesBeforeMeasure(qc);
  std::cout << "-----------------------------\n";
  qc.print(std::cout);
  EXPECT_EQ(qc.getNops(), 1);
  EXPECT_EQ(qc.begin()->get()->getType(), qc::Measure);
}

TEST_F(QFRFunctionality, leaveGateBeforeMeasure) {
  const std::size_t nqubits = 2;
  QuantumComputation qc(nqubits, nqubits);
  qc.cz(0, 1);
  qc.x(0);
  qc.measure({0, 1}, {0, 1});
  std::cout << "-----------------------------\n";
  qc.print(std::cout);
  CircuitOptimizer::removeDiagonalGatesBeforeMeasure(qc);
  std::cout << "-----------------------------\n";
  qc.print(std::cout);
  EXPECT_EQ(qc.getNops(), 3);
}

TEST_F(QFRFunctionality, removeComplexGateBeforeMeasure) {
  const std::size_t nqubits = 4;
  QuantumComputation qc(nqubits, nqubits);
  qc.cz(0, 1);
  qc.x(0);
  qc.cz(1, 2);
  qc.cz(0, 1);
  qc.z(0);
  qc.cz(1, 2);
  qc.x(3);
  qc.t(3);
  qc.mcz({0, 1, 2}, 3);
  qc.measure({0, 1, 2, 3}, {0, 1, 2, 3});
  std::cout << "-----------------------------\n";
  qc.print(std::cout);
  CircuitOptimizer::removeDiagonalGatesBeforeMeasure(qc);
  std::cout << "-----------------------------\n";
  qc.print(std::cout);
  EXPECT_EQ(qc.getNops(), 4);
}

TEST_F(QFRFunctionality, removeSimpleCompoundOpBeforeMeasure) {
  const std::size_t nqubits = 1;
  QuantumComputation qc(nqubits, nqubits);
  qc.x(0);
  qc.t(0);
  qc.measure(0, 0);
  std::cout << "-----------------------------\n";
  qc.print(std::cout);
  CircuitOptimizer::singleQubitGateFusion(qc);
  CircuitOptimizer::removeDiagonalGatesBeforeMeasure(qc);
  std::cout << "-----------------------------\n";
  qc.print(std::cout);
  EXPECT_EQ(qc.getNops(), 2);
}

TEST_F(QFRFunctionality, removePartOfCompoundOpBeforeMeasure) {
  const std::size_t nqubits = 1;
  QuantumComputation qc(nqubits, nqubits);
  qc.t(0);
  qc.x(0);
  qc.t(0);
  qc.measure(0, 0);
  std::cout << "-----------------------------\n";
  qc.print(std::cout);
  CircuitOptimizer::singleQubitGateFusion(qc);
  CircuitOptimizer::removeDiagonalGatesBeforeMeasure(qc);
  std::cout << "-----------------------------\n";
  qc.print(std::cout);
  EXPECT_EQ(qc.getNops(), 2);
}

TEST_F(QFRFunctionality, decomposeSWAPsUndirectedArchitecture) {
  const std::size_t nqubits = 2;
  QuantumComputation qc(nqubits, nqubits);
  qc.swap(0, 1);
  std::cout << "-----------------------------\n";
  qc.print(std::cout);
  CircuitOptimizer::decomposeSWAP(qc, false);
  std::cout << "-----------------------------\n";
  qc.print(std::cout);
  auto it = qc.begin();
  const auto& op = *it;
  EXPECT_TRUE(op->isStandardOperation());
  EXPECT_EQ(op->getType(), qc::X);
  EXPECT_EQ(op->getControls().begin()->qubit, 0);
  EXPECT_EQ(op->getTargets().at(0), 1);
  ++it;
  const auto& op2 = *it;
  EXPECT_TRUE(op2->isStandardOperation());
  EXPECT_EQ(op2->getType(), qc::X);
  EXPECT_EQ(op2->getControls().begin()->qubit, 1);
  EXPECT_EQ(op2->getTargets().at(0), 0);
  ++it;
  const auto& op3 = *it;
  EXPECT_TRUE(op3->isStandardOperation());
  EXPECT_EQ(op3->getType(), qc::X);
  EXPECT_EQ(op3->getControls().begin()->qubit, 0);
  EXPECT_EQ(op3->getTargets().at(0), 1);
}
TEST_F(QFRFunctionality, decomposeSWAPsDirectedArchitecture) {
  const std::size_t nqubits = 2;
  QuantumComputation qc(nqubits);
  qc.swap(0, 1);
  std::cout << "-----------------------------\n";
  qc.print(std::cout);
  CircuitOptimizer::decomposeSWAP(qc, true);
  std::cout << "-----------------------------\n";
  qc.print(std::cout);
  auto it = qc.begin();
  const auto& op = *it;
  EXPECT_TRUE(op->isStandardOperation());
  EXPECT_EQ(op->getType(), qc::X);
  EXPECT_EQ(op->getControls().begin()->qubit, 0);
  EXPECT_EQ(op->getTargets().at(0), 1);

  ++it;
  const auto& op2 = *it;
  EXPECT_TRUE(op2->isStandardOperation());
  EXPECT_EQ(op2->getType(), qc::H);
  EXPECT_EQ(op2->getTargets().at(0), 1);

  ++it;
  const auto& op3 = *it;
  EXPECT_TRUE(op3->isStandardOperation());
  EXPECT_EQ(op3->getType(), qc::H);
  EXPECT_EQ(op3->getTargets().at(0), 0);

  ++it;
  const auto& op4 = *it;
  EXPECT_TRUE(op4->isStandardOperation());
  EXPECT_EQ(op4->getType(), qc::X);
  EXPECT_EQ(op4->getControls().begin()->qubit, 0);
  EXPECT_EQ(op4->getTargets().at(0), 1);

  ++it;
  const auto& op5 = *it;
  EXPECT_TRUE(op5->isStandardOperation());
  EXPECT_EQ(op5->getType(), qc::H);
  EXPECT_EQ(op5->getTargets().at(0), 1);

  ++it;
  const auto& op6 = *it;
  EXPECT_TRUE(op6->isStandardOperation());
  EXPECT_EQ(op6->getType(), qc::H);
  EXPECT_EQ(op6->getTargets().at(0), 0);

  ++it;
  const auto& op7 = *it;
  EXPECT_TRUE(op7->isStandardOperation());
  EXPECT_EQ(op7->getType(), qc::X);
  EXPECT_EQ(op7->getControls().begin()->qubit, 0);
  EXPECT_EQ(op7->getTargets().at(0), 1);
}

TEST_F(QFRFunctionality, decomposeSWAPsCompound) {
  const std::size_t nqubits = 2;
  QuantumComputation qc(nqubits);
  QuantumComputation comp(nqubits);
  comp.swap(0, 1);
  comp.swap(0, 1);
  comp.swap(0, 1);
  qc.emplace_back(comp.asOperation());
  std::cout << "-----------------------------\n";
  qc.print(std::cout);
  qc::CircuitOptimizer::decomposeSWAP(qc, false);
  std::cout << "-----------------------------\n";
  qc.print(std::cout);
  auto it = qc.begin();
  const auto& op = *it;
  EXPECT_TRUE(op->isCompoundOperation());

  const auto* cop = dynamic_cast<qc::CompoundOperation*>(op.get());
  ASSERT_NE(cop, nullptr);
  EXPECT_EQ(cop->size(), 9);
}

TEST_F(QFRFunctionality, decomposeSWAPsCompoundDirected) {
  const std::size_t nqubits = 2;
  QuantumComputation qc(nqubits);
  QuantumComputation comp(nqubits);
  comp.swap(0, 1);
  comp.swap(0, 1);
  comp.swap(0, 1);
  qc.emplace_back(comp.asOperation());
  std::cout << "-----------------------------\n";
  qc.print(std::cout);
  qc::CircuitOptimizer::decomposeSWAP(qc, true);
  std::cout << "-----------------------------\n";
  qc.print(std::cout);
  auto it = qc.begin();
  const auto& op = *it;
  EXPECT_TRUE(op->isCompoundOperation());

  const auto* cop = dynamic_cast<qc::CompoundOperation*>(op.get());
  ASSERT_NE(cop, nullptr);
  EXPECT_EQ(cop->size(), 21);
}

TEST_F(QFRFunctionality, removeFinalMeasurements) {
  const std::size_t nqubits = 2;
  QuantumComputation qc(nqubits, nqubits);
  qc.h(0);
  qc.h(1);
  qc.measure(0, 0);
  qc.measure(1, 1);
  qc.h(1);
  std::cout << "-----------------------------\n";
  qc.print(std::cout);
  CircuitOptimizer::removeFinalMeasurements(qc);
  std::cout << "-----------------------------\n";
  qc.print(std::cout);
  auto it = qc.begin();
  ++it;
  ++it; // skip first two H
  const auto& op = *it;
  EXPECT_TRUE(op->isNonUnitaryOperation());
  EXPECT_EQ(op->getType(), qc::Measure);
  ++it;
  const auto& op2 = *it;
  EXPECT_TRUE(op2->isStandardOperation());
  EXPECT_EQ(op2->getType(), qc::H);
  EXPECT_EQ(op2->getTargets().at(0), 1);
}

TEST_F(QFRFunctionality, removeFinalMeasurementsTwoQubitMeasurement) {
  const std::size_t nqubits = 2;
  QuantumComputation qc(nqubits, nqubits);
  qc.h(0);
  qc.h(1);
  qc.measure({0, 1}, {0, 1});
  qc.h(1);
  std::cout << "-----------------------------\n";
  qc.print(std::cout);
  CircuitOptimizer::removeFinalMeasurements(qc);
  std::cout << "-----------------------------\n";
  qc.print(std::cout);
  auto it = qc.begin();
  ++it;
  ++it; // skip first two H
  const auto& op = *it;
  EXPECT_TRUE(op->isNonUnitaryOperation());
  EXPECT_EQ(op->getType(), qc::Measure);
  ++it;
  const auto& op2 = *it;
  EXPECT_TRUE(op2->isStandardOperation());
  EXPECT_EQ(op2->getType(), qc::H);
  EXPECT_EQ(op2->getTargets().at(0), 1);
}

TEST_F(QFRFunctionality, removeFinalMeasurementsCompound) {
  const std::size_t nqubits = 2;
  QuantumComputation qc(nqubits, nqubits);
  QuantumComputation comp(nqubits, nqubits);
  comp.measure(0, 0);
  comp.measure(1, 1);
  comp.h(1);
  qc.emplace_back(comp.asOperation());
  qc.h(1);
  std::cout << "-----------------------------\n";
  qc.print(std::cout);
  CircuitOptimizer::removeFinalMeasurements(qc);
  std::cout << "-----------------------------\n";
  qc.print(std::cout);
  auto it = qc.begin();
  const auto& op = *it;
  EXPECT_TRUE(op->isCompoundOperation());

  const auto* cop = dynamic_cast<qc::CompoundOperation*>(op.get());
  ASSERT_NE(cop, nullptr);
  EXPECT_EQ(cop->size(), 2);
  ++it;
  const auto& op2 = *it;
  EXPECT_TRUE(op2->isStandardOperation());
  EXPECT_EQ(op2->getType(), qc::H);
  EXPECT_EQ(op2->getTargets().at(0), 1);
}

TEST_F(QFRFunctionality, removeFinalMeasurementsCompoundDegraded) {
  const std::size_t nqubits = 2;
  QuantumComputation qc(nqubits, nqubits);
  QuantumComputation comp(nqubits, nqubits);
  comp.measure(0, 0);
  comp.h(1);
  qc.emplace_back(comp.asOperation());
  qc.h(1);
  std::cout << "-----------------------------\n";
  qc.print(std::cout);
  CircuitOptimizer::removeFinalMeasurements(qc);
  std::cout << "-----------------------------\n";
  qc.print(std::cout);
  auto it = qc.begin();
  const auto& op = *it;
  EXPECT_TRUE(op->isStandardOperation());
  EXPECT_EQ(op->getType(), qc::H);
  EXPECT_EQ(op->getTargets().at(0), 1);
  ++it;
  const auto& op2 = *it;
  EXPECT_TRUE(op2->isStandardOperation());
  EXPECT_EQ(op2->getType(), qc::H);
  EXPECT_EQ(op2->getTargets().at(0), 1);
}

TEST_F(QFRFunctionality, removeFinalMeasurementsCompoundEmpty) {
  const std::size_t nqubits = 2;
  QuantumComputation qc(nqubits, nqubits);
  QuantumComputation comp(nqubits, nqubits);
  comp.measure(0, 0);
  qc.emplace_back(comp.asCompoundOperation());
  qc.h(1);
  std::cout << "-----------------------------\n";
  qc.print(std::cout);
  CircuitOptimizer::removeFinalMeasurements(qc);
  std::cout << "-----------------------------\n";
  qc.print(std::cout);
  auto it = qc.begin();
  const auto& op = *it;
  EXPECT_TRUE(op->isStandardOperation());
  EXPECT_EQ(op->getType(), qc::H);
  EXPECT_EQ(op->getTargets().at(0), 1);
}

TEST_F(QFRFunctionality, removeFinalMeasurementsWithOperationsInFront) {
  const std::string circ =
      "OPENQASM 2.0;include \"qelib1.inc\";qreg q[3];qreg r[3];h q;cx q, "
      "r;creg c[3];creg d[3];barrier q;measure q->c;measure r->d;\n";
  std::stringstream ss{};
  ss << circ;
  QuantumComputation qc{};
  qc.import(ss, qc::Format::OpenQASM);
  std::cout << "-----------------------------\n";
  qc.print(std::cout);
  CircuitOptimizer::removeFinalMeasurements(qc);
  std::cout << "-----------------------------\n";
  qc.print(std::cout);
  ASSERT_EQ(qc.getNops(), 2);
  ASSERT_EQ(qc.getNindividualOps(), 6);
}

TEST_F(QFRFunctionality, removeFinalMeasurementsWithBarrier) {
  const std::size_t nqubits = 2;
  QuantumComputation qc(nqubits, nqubits);
  qc.barrier({0, 1});
  qc.measure(0, 0);
  qc.measure(1, 1);
  std::cout << "-----------------------------\n";
  qc.print(std::cout);
  CircuitOptimizer::removeFinalMeasurements(qc);
  std::cout << "-----------------------------\n";
  qc.print(std::cout);
  EXPECT_TRUE(qc.empty());
}

TEST_F(QFRFunctionality, gateShortCutsAndCloning) {
  // This test checks if the gate shortcuts are working correctly
  // and if the cloning of gates is working correctly.
  // To this end, we create a circuit with every possible gate in the following
  // variants:
  //  - without controls,
  //  - with a single control,
  //  - with multiple controls.
  // Then, we clone the circuit and check if the resulting circuit contains the
  // same number of gates.
  QuantumComputation qc(5, 5);
  qc.gphase(PI);
  qc.i(0);
  qc.ci(1, 0);
  qc.mci({1, 2_nc}, 0);
  qc.h(0);
  qc.ch(1, 0);
  qc.mch({1, 2_nc}, 0);
  qc.x(0);
  qc.cx(1, 0);
  qc.mcx({1, 2_nc}, 0);
  qc.y(0);
  qc.cy(1, 0);
  qc.mcy({1, 2_nc}, 0);
  qc.z(0);
  qc.cz(1, 0);
  qc.mcz({1, 2_nc}, 0);
  qc.s(0);
  qc.cs(1, 0);
  qc.mcs({1, 2_nc}, 0);
  qc.sdg(0);
  qc.csdg(1, 0);
  qc.mcsdg({1, 2_nc}, 0);
  qc.t(0);
  qc.ct(1, 0);
  qc.mct({1, 2_nc}, 0);
  qc.tdg(0);
  qc.ctdg(1, 0);
  qc.mctdg({1, 2_nc}, 0);
  qc.v(0);
  qc.cv(1, 0);
  qc.mcv({1, 2_nc}, 0);
  qc.vdg(0);
  qc.cvdg(1, 0);
  qc.mcvdg({1, 2_nc}, 0);
  qc.u(PI, PI, PI, 0);
  qc.cu(PI, PI, PI, 1, 0);
  qc.mcu(PI, PI, PI, {1, 2_nc}, 0);
  qc.u2(PI, PI, 0);
  qc.cu2(PI, PI, 1, 0);
  qc.mcu2(PI, PI, {1, 2_nc}, 0);
  qc.p(PI, 0);
  qc.cp(PI, 1, 0);
  qc.mcp(PI, {1, 2_nc}, 0);
  qc.sx(0);
  qc.csx(1, 0);
  qc.mcsx({1, 2_nc}, 0);
  qc.sxdg(0);
  qc.csxdg(1, 0);
  qc.mcsxdg({1, 2_nc}, 0);
  qc.rx(PI, 0);
  qc.crx(PI, 1, 0);
  qc.mcrx(PI, {1, 2_nc}, 0);
  qc.ry(PI, 0);
  qc.cry(PI, 1, 0);
  qc.mcry(PI, {1, 2_nc}, 0);
  qc.rz(PI, 0);
  qc.crz(PI, 1, 0);
  qc.mcrz(PI, {1, 2_nc}, 0);
  qc.swap(0, 1);
  qc.cswap(2, 0, 1);
  qc.mcswap({2, 3_nc}, 0, 1);
  qc.iswap(0, 1);
  qc.ciswap(2, 0, 1);
  qc.mciswap({2, 3_nc}, 0, 1);
  qc.peres(0, 1);
  qc.cperes(2, 0, 1);
  qc.mcperes({2, 3_nc}, 0, 1);
  qc.peresdg(0, 1);
  qc.cperesdg(2, 0, 1);
  qc.mcperesdg({2, 3_nc}, 0, 1);
  qc.dcx(0, 1);
  qc.cdcx(2, 0, 1);
  qc.mcdcx({2, 3_nc}, 0, 1);
  qc.ecr(0, 1);
  qc.cecr(2, 0, 1);
  qc.mcecr({2, 3_nc}, 0, 1);
  qc.rxx(PI, 0, 1);
  qc.crxx(PI, 2, 0, 1);
  qc.mcrxx(PI, {2, 3_nc}, 0, 1);
  qc.ryy(PI, 0, 1);
  qc.cryy(PI, 2, 0, 1);
  qc.mcryy(PI, {2, 3_nc}, 0, 1);
  qc.rzz(PI, 0, 1);
  qc.crzz(PI, 2, 0, 1);
  qc.mcrzz(PI, {2, 3_nc}, 0, 1);
  qc.rzx(PI, 0, 1);
  qc.crzx(PI, 2, 0, 1);
  qc.mcrzx(PI, {2, 3_nc}, 0, 1);
  qc.xx_minus_yy(PI, PI, 0, 1);
  qc.cxx_minus_yy(PI, PI, 2, 0, 1);
  qc.mcxx_minus_yy(PI, PI, {2, 3_nc}, 0, 1);
  qc.xx_plus_yy(PI, PI, 0, 1);
  qc.cxx_plus_yy(PI, PI, 2, 0, 1);
  qc.mcxx_plus_yy(PI, PI, {2, 3_nc}, 0, 1);
  qc.measure(0, 0);
  qc.measure({1, 2}, {1, 2});
  qc.barrier(0);
  qc.barrier({1, 2});
  qc.reset(0);
  qc.reset({1, 2});

  const auto qcCloned = qc;
  ASSERT_EQ(qc.size(), qcCloned.size());
  ASSERT_EQ(qcCloned.getGlobalPhase(), PI);
}

TEST_F(QFRFunctionality, cloningDifferentOperations) {
  const std::size_t nqubits = 5;
  QuantumComputation qc(nqubits, nqubits);
  QuantumComputation comp(nqubits);
  comp.barrier(0);
  comp.h(0);
  qc.emplace_back(comp.asOperation());
  qc.classicControlled(qc::X, 0, qc.getCregs().at("c"), 1);

  const auto qcCloned = qc;
  ASSERT_EQ(qc.size(), qcCloned.size());
}

TEST_F(QFRFunctionality, wrongRegisterSizes) {
  const std::size_t nqubits = 5;
  QuantumComputation qc(nqubits);
  ASSERT_THROW(qc.measure({0}, {1, 2}), std::invalid_argument);
}

TEST_F(QFRFunctionality, eliminateResetsBasicTest) {
  QuantumComputation qc{};
  qc.addQubitRegister(1);
  qc.addClassicalRegister(2);
  qc.h(0);
  qc.measure(0, 0U);
  qc.reset(0);
  qc.h(0);
  qc.measure(0, 1U);

  std::cout << qc << "\n";

  EXPECT_TRUE(CircuitOptimizer::isDynamicCircuit(qc));

  EXPECT_NO_THROW(CircuitOptimizer::eliminateResets(qc););

  std::cout << qc << "\n";

  ASSERT_EQ(qc.getNqubits(), 2);
  ASSERT_EQ(qc.getNindividualOps(), 4);
  const auto& op0 = qc.at(0);
  const auto& op1 = qc.at(1);
  const auto& op2 = qc.at(2);
  const auto& op3 = qc.at(3);

  EXPECT_EQ(op0->getNqubits(), 2);
  EXPECT_TRUE(op0->getType() == qc::H);
  const auto& targets0 = op0->getTargets();
  EXPECT_EQ(targets0.size(), 1);
  EXPECT_EQ(targets0.at(0), static_cast<Qubit>(0));
  EXPECT_TRUE(op0->getControls().empty());

  EXPECT_EQ(op1->getNqubits(), 2);
  EXPECT_TRUE(op1->getType() == qc::Measure);
  const auto& targets1 = op1->getTargets();
  EXPECT_EQ(targets1.size(), 1);
  EXPECT_EQ(targets1.at(0), static_cast<Qubit>(0));
  const auto* measure0 = dynamic_cast<qc::NonUnitaryOperation*>(op1.get());
  ASSERT_NE(measure0, nullptr);
  const auto& classics0 = measure0->getClassics();
  EXPECT_EQ(classics0.size(), 1);
  EXPECT_EQ(classics0.at(0), 0);

  EXPECT_EQ(op2->getNqubits(), 2);
  EXPECT_TRUE(op2->getType() == qc::H);
  const auto& targets2 = op2->getTargets();
  EXPECT_EQ(targets2.size(), 1);
  EXPECT_EQ(targets2.at(0), static_cast<Qubit>(1));
  EXPECT_TRUE(op2->getControls().empty());

  EXPECT_EQ(op3->getNqubits(), 2);
  EXPECT_TRUE(op3->getType() == qc::Measure);
  const auto& targets3 = op3->getTargets();
  EXPECT_EQ(targets3.size(), 1);
  EXPECT_EQ(targets3.at(0), static_cast<Qubit>(1));
  auto* measure1 = dynamic_cast<qc::NonUnitaryOperation*>(op3.get());
  ASSERT_NE(measure1, nullptr);
  const auto& classics1 = measure1->getClassics();
  EXPECT_EQ(classics1.size(), 1);
  EXPECT_EQ(classics1.at(0), 1);
}

TEST_F(QFRFunctionality, eliminateResetsClassicControlled) {
  QuantumComputation qc{};
  qc.addQubitRegister(1);
  qc.addClassicalRegister(2);
  qc.h(0);
  qc.measure(0, 0U);
  qc.reset(0);
  qc.classicControlled(qc::X, 0, {0, 1U}, 1U);
  std::cout << qc << "\n";

  EXPECT_TRUE(CircuitOptimizer::isDynamicCircuit(qc));

  EXPECT_NO_THROW(CircuitOptimizer::eliminateResets(qc););

  std::cout << qc << "\n";

  ASSERT_EQ(qc.getNqubits(), 2);
  ASSERT_EQ(qc.getNindividualOps(), 3);
  const auto& op0 = qc.at(0);
  const auto& op1 = qc.at(1);
  const auto& op2 = qc.at(2);

  EXPECT_EQ(op0->getNqubits(), 2);
  EXPECT_TRUE(op0->getType() == qc::H);
  const auto& targets0 = op0->getTargets();
  EXPECT_EQ(targets0.size(), 1);
  EXPECT_EQ(targets0.at(0), static_cast<Qubit>(0));
  EXPECT_TRUE(op0->getControls().empty());

  EXPECT_EQ(op1->getNqubits(), 2);
  EXPECT_TRUE(op1->getType() == qc::Measure);
  const auto& targets1 = op1->getTargets();
  EXPECT_EQ(targets1.size(), 1);
  EXPECT_EQ(targets1.at(0), static_cast<Qubit>(0));
  auto* measure0 = dynamic_cast<qc::NonUnitaryOperation*>(op1.get());
  ASSERT_NE(measure0, nullptr);
  const auto& classics0 = measure0->getClassics();
  EXPECT_EQ(classics0.size(), 1);
  EXPECT_EQ(classics0.at(0), 0);

  EXPECT_EQ(op2->getNqubits(), 2);
  EXPECT_TRUE(op2->isClassicControlledOperation());
  auto* classicControlled =
      dynamic_cast<qc::ClassicControlledOperation*>(op2.get());
  ASSERT_NE(classicControlled, nullptr);
  const auto& operation = classicControlled->getOperation();
  EXPECT_EQ(operation->getNqubits(), 2);
  EXPECT_TRUE(operation->getType() == qc::X);
  EXPECT_EQ(classicControlled->getNtargets(), 1);
  const auto& targets = classicControlled->getTargets();
  EXPECT_EQ(targets.at(0), 1);
  EXPECT_EQ(classicControlled->getNcontrols(), 0);
}

TEST_F(QFRFunctionality, eliminateResetsMultipleTargetReset) {
  QuantumComputation qc{};
  qc.addQubitRegister(2);
  qc.reset({0, 1});
  qc.x(0);
  qc.z(1);
  qc.cx(1, 0);

  std::cout << qc << "\n";

  EXPECT_TRUE(CircuitOptimizer::isDynamicCircuit(qc));

  EXPECT_NO_THROW(CircuitOptimizer::eliminateResets(qc););

  std::cout << qc << "\n";

  ASSERT_EQ(qc.getNqubits(), 4);
  ASSERT_EQ(qc.getNindividualOps(), 3);
  const auto& op0 = qc.at(0);
  const auto& op1 = qc.at(1);
  const auto& op2 = qc.at(2);

  EXPECT_EQ(op0->getNqubits(), 4);
  EXPECT_TRUE(op0->getType() == qc::X);
  const auto& targets0 = op0->getTargets();
  EXPECT_EQ(targets0.size(), 1);
  EXPECT_EQ(targets0.at(0), static_cast<Qubit>(2));
  EXPECT_TRUE(op0->getControls().empty());

  EXPECT_EQ(op1->getNqubits(), 4);
  EXPECT_TRUE(op1->getType() == qc::Z);
  const auto& targets1 = op1->getTargets();
  EXPECT_EQ(targets1.size(), 1);
  EXPECT_EQ(targets1.at(0), static_cast<Qubit>(3));
  EXPECT_TRUE(op1->getControls().empty());

  EXPECT_EQ(op2->getNqubits(), 4);
  EXPECT_TRUE(op2->getType() == qc::X);
  const auto& targets2 = op2->getTargets();
  EXPECT_EQ(targets2.size(), 1);
  EXPECT_EQ(targets2.at(0), static_cast<Qubit>(2));
  const auto& controls2 = op2->getControls();
  EXPECT_EQ(controls2.size(), 1);
  EXPECT_EQ(controls2.count(3), 1);
}

TEST_F(QFRFunctionality, eliminateResetsCompoundOperation) {
  QuantumComputation qc(2U, 2U);

  qc.reset(0);
  qc.reset(1);

  QuantumComputation comp(2U, 2U);
  comp.cx(1, 0);
  comp.reset(0);
  comp.measure(0, 0);
  comp.classicControlled(qc::X, 0, {0, 1U}, 1U);
  qc.emplace_back(comp.asOperation());

  std::cout << qc << "\n";

  EXPECT_TRUE(CircuitOptimizer::isDynamicCircuit(qc));

  EXPECT_NO_THROW(CircuitOptimizer::eliminateResets(qc););

  std::cout << qc << "\n";

  ASSERT_EQ(qc.getNqubits(), 5);
  ASSERT_EQ(qc.getNindividualOps(), 3);

  const auto& op = qc.at(0);
  EXPECT_TRUE(op->isCompoundOperation());
  EXPECT_EQ(op->getNqubits(), 5);
  auto* compOp0 = dynamic_cast<qc::CompoundOperation*>(op.get());
  ASSERT_NE(compOp0, nullptr);
  EXPECT_EQ(compOp0->size(), 3);

  const auto& op0 = compOp0->at(0);
  const auto& op1 = compOp0->at(1);
  const auto& op2 = compOp0->at(2);

  EXPECT_EQ(op0->getNqubits(), 5);
  EXPECT_TRUE(op0->getType() == qc::X);
  const auto& targets0 = op0->getTargets();
  EXPECT_EQ(targets0.size(), 1);
  EXPECT_EQ(targets0.at(0), static_cast<Qubit>(2));
  const auto& controls0 = op0->getControls();
  EXPECT_EQ(controls0.size(), 1);
  EXPECT_EQ(controls0.count(3), 1);

  EXPECT_EQ(op1->getNqubits(), 5);
  EXPECT_TRUE(op1->getType() == qc::Measure);
  const auto& targets1 = op1->getTargets();
  EXPECT_EQ(targets1.size(), 1);
  EXPECT_EQ(targets1.at(0), static_cast<Qubit>(4));
  auto* measure0 = dynamic_cast<qc::NonUnitaryOperation*>(op1.get());
  ASSERT_NE(measure0, nullptr);
  const auto& classics0 = measure0->getClassics();
  EXPECT_EQ(classics0.size(), 1);
  EXPECT_EQ(classics0.at(0), 0);

  EXPECT_EQ(op2->getNqubits(), 5);
  EXPECT_TRUE(op2->isClassicControlledOperation());
  auto* classicControlled =
      dynamic_cast<qc::ClassicControlledOperation*>(op2.get());
  ASSERT_NE(classicControlled, nullptr);
  const auto& operation = classicControlled->getOperation();
  EXPECT_EQ(operation->getNqubits(), 5);
  EXPECT_TRUE(operation->getType() == qc::X);
  EXPECT_EQ(classicControlled->getNtargets(), 1);
  const auto& targets = classicControlled->getTargets();
  EXPECT_EQ(targets.at(0), 4);
  EXPECT_EQ(classicControlled->getNcontrols(), 0);
}

TEST_F(QFRFunctionality, deferMeasurementsBasicTest) {
  // Input:
  // i: 		0	1
  // 1: 	H   	H 	|
  // 2: 	Meas	0	|
  // 3: 	c_X   	|	X 		c[0] == 1
  // o: 		0	1

  // Expected Output:
  // i: 		0	1
  // 1: 	H   	H 	|
  // 3: 	X   	c	X
  // 2: 	Meas	0	|
  // o: 		0	1

  QuantumComputation qc{};
  qc.addQubitRegister(2);
  qc.addClassicalRegister(1);
  qc.h(0);
  qc.measure(0, 0U);
  qc.classicControlled(qc::X, 1, {0, 1U}, 1U);
  std::cout << qc << "\n";

  EXPECT_TRUE(CircuitOptimizer::isDynamicCircuit(qc));

  EXPECT_NO_THROW(CircuitOptimizer::deferMeasurements(qc););

  std::cout << qc << "\n";

  EXPECT_FALSE(CircuitOptimizer::isDynamicCircuit(qc));

  ASSERT_EQ(qc.getNqubits(), 2);
  ASSERT_EQ(qc.getNindividualOps(), 3);
  const auto& op0 = qc.at(0);
  const auto& op1 = qc.at(1);
  const auto& op2 = qc.at(2);

  EXPECT_EQ(op0->getNqubits(), 2);
  EXPECT_TRUE(op0->getType() == qc::H);
  const auto& targets0 = op0->getTargets();
  EXPECT_EQ(targets0.size(), 1);
  EXPECT_EQ(targets0.at(0), static_cast<Qubit>(0));
  EXPECT_TRUE(op0->getControls().empty());

  EXPECT_EQ(op1->getNqubits(), 2);
  EXPECT_TRUE(op1->getType() == qc::X);
  const auto& targets1 = op1->getTargets();
  EXPECT_EQ(targets1.size(), 1);
  EXPECT_EQ(targets1.at(0), static_cast<Qubit>(1));
  const auto& controls1 = op1->getControls();
  EXPECT_EQ(controls1.size(), 1);
  EXPECT_EQ(controls1.count(0), 1);

  EXPECT_EQ(op2->getNqubits(), 2);
  ASSERT_TRUE(op2->getType() == qc::Measure);
  const auto& targets2 = op2->getTargets();
  EXPECT_EQ(targets2.size(), 1);
  EXPECT_EQ(targets2.at(0), static_cast<Qubit>(0));
  auto* measure0 = dynamic_cast<qc::NonUnitaryOperation*>(op2.get());
  ASSERT_NE(measure0, nullptr);
  const auto& classics0 = measure0->getClassics();
  EXPECT_EQ(classics0.size(), 1);
  EXPECT_EQ(classics0.at(0), 0);
}

TEST_F(QFRFunctionality,
       deferMeasurementsOperationBetweenMeasurementAndClassic) {
  // Input:
  // i: 			0	1
  // 1: 	H   	H 	|
  // 2: 	Meas	0	|
  // 3: 	H   	H 	|
  // 4: 	c_X   	|	X 		c[0] == 1
  // o: 			0	1

  // Expected Output:
  // i: 			0	1
  // 1: 	H   	H 	|
  // 2: 	X   	c	X
  // 3: 	H   	H 	|
  // 4: 	Meas	0	|
  // o: 			0	1

  QuantumComputation qc{};
  qc.addQubitRegister(2);
  qc.addClassicalRegister(1);
  qc.h(0);
  qc.measure(0, 0U);
  qc.h(0);
  qc.classicControlled(qc::X, 1, {0, 1U}, 1U);
  std::cout << qc << "\n";

  EXPECT_TRUE(CircuitOptimizer::isDynamicCircuit(qc));

  EXPECT_NO_THROW(CircuitOptimizer::deferMeasurements(qc););

  std::cout << qc << "\n";

  EXPECT_FALSE(CircuitOptimizer::isDynamicCircuit(qc));

  ASSERT_EQ(qc.getNqubits(), 2);
  ASSERT_EQ(qc.getNindividualOps(), 4);
  const auto& op0 = qc.at(0);
  const auto& op1 = qc.at(1);
  const auto& op2 = qc.at(2);
  const auto& op3 = qc.at(3);

  EXPECT_EQ(op0->getNqubits(), 2);
  EXPECT_TRUE(op0->getType() == qc::H);
  const auto& targets0 = op0->getTargets();
  EXPECT_EQ(targets0.size(), 1);
  EXPECT_EQ(targets0.at(0), static_cast<Qubit>(0));
  EXPECT_TRUE(op0->getControls().empty());

  EXPECT_EQ(op1->getNqubits(), 2);
  EXPECT_TRUE(op1->getType() == qc::X);
  const auto& targets1 = op1->getTargets();
  EXPECT_EQ(targets1.size(), 1);
  EXPECT_EQ(targets1.at(0), static_cast<Qubit>(1));
  const auto& controls1 = op1->getControls();
  EXPECT_EQ(controls1.size(), 1);
  EXPECT_EQ(controls1.count(0), 1);

  EXPECT_EQ(op2->getNqubits(), 2);
  EXPECT_TRUE(op2->getType() == qc::H);
  const auto& targets2 = op2->getTargets();
  EXPECT_EQ(targets2.size(), 1);
  EXPECT_EQ(targets2.at(0), static_cast<Qubit>(0));
  EXPECT_TRUE(op2->getControls().empty());

  EXPECT_EQ(op3->getNqubits(), 2);
  ASSERT_TRUE(op3->getType() == qc::Measure);
  const auto& targets3 = op3->getTargets();
  EXPECT_EQ(targets3.size(), 1);
  EXPECT_EQ(targets3.at(0), static_cast<Qubit>(0));
  auto* measure0 = dynamic_cast<qc::NonUnitaryOperation*>(op3.get());
  ASSERT_NE(measure0, nullptr);
  const auto& classics0 = measure0->getClassics();
  EXPECT_EQ(classics0.size(), 1);
  EXPECT_EQ(classics0.at(0), 0);
}

TEST_F(QFRFunctionality, deferMeasurementsTwoClassic) {
  // Input:
  // i: 			0	1
  // 1: 	H   	H 	|
  // 2: 	Meas	0	|
  // 3: 	H   	H 	|
  // 4: 	c_X   	|	X 		c[0] == 1
  // 5:    c_Z     |   Z       c[0] == 1
  // o: 			0	1

  // Expected Output:
  // i: 			0	1
  // 1: 	H   	H 	|
  // 2: 	X   	c	X
  // 3: 	Z   	c	Z
  // 4: 	H   	H 	|
  // 5: 	Meas	0	|
  // o: 			0	1

  QuantumComputation qc{};
  qc.addQubitRegister(2);
  qc.addClassicalRegister(1);
  qc.h(0);
  qc.measure(0, 0U);
  qc.h(0);
  qc.classicControlled(qc::X, 1, {0, 1U}, 1U);
  qc.classicControlled(qc::Z, 1, {0, 1U}, 1U);

  std::cout << qc << "\n";

  EXPECT_TRUE(CircuitOptimizer::isDynamicCircuit(qc));

  EXPECT_NO_THROW(CircuitOptimizer::deferMeasurements(qc););

  std::cout << qc << "\n";

  EXPECT_FALSE(CircuitOptimizer::isDynamicCircuit(qc));

  ASSERT_EQ(qc.getNqubits(), 2);
  ASSERT_EQ(qc.getNindividualOps(), 5);
  const auto& op0 = qc.at(0);
  const auto& op1 = qc.at(1);
  const auto& op2 = qc.at(2);
  const auto& op3 = qc.at(3);
  const auto& op4 = qc.at(4);

  EXPECT_EQ(op0->getNqubits(), 2);
  EXPECT_TRUE(op0->getType() == qc::H);
  const auto& targets0 = op0->getTargets();
  EXPECT_EQ(targets0.size(), 1);
  EXPECT_EQ(targets0.at(0), static_cast<Qubit>(0));
  EXPECT_TRUE(op0->getControls().empty());

  EXPECT_EQ(op1->getNqubits(), 2);
  EXPECT_TRUE(op1->getType() == qc::X);
  const auto& targets1 = op1->getTargets();
  EXPECT_EQ(targets1.size(), 1);
  EXPECT_EQ(targets1.at(0), static_cast<Qubit>(1));
  const auto& controls1 = op1->getControls();
  EXPECT_EQ(controls1.size(), 1);
  EXPECT_EQ(controls1.count(0), 1);

  EXPECT_EQ(op2->getNqubits(), 2);
  EXPECT_TRUE(op2->getType() == qc::Z);
  const auto& targets2 = op2->getTargets();
  EXPECT_EQ(targets2.size(), 1);
  EXPECT_EQ(targets2.at(0), static_cast<Qubit>(1));
  const auto& controls2 = op2->getControls();
  EXPECT_EQ(controls2.size(), 1);
  EXPECT_EQ(controls2.count(0), 1);

  EXPECT_EQ(op3->getNqubits(), 2);
  EXPECT_TRUE(op3->getType() == qc::H);
  const auto& targets3 = op3->getTargets();
  EXPECT_EQ(targets3.size(), 1);
  EXPECT_EQ(targets3.at(0), static_cast<Qubit>(0));
  EXPECT_TRUE(op3->getControls().empty());

  EXPECT_EQ(op4->getNqubits(), 2);
  ASSERT_TRUE(op4->getType() == qc::Measure);
  const auto& targets4 = op4->getTargets();
  EXPECT_EQ(targets4.size(), 1);
  EXPECT_EQ(targets4.at(0), static_cast<Qubit>(0));
  auto* measure0 = dynamic_cast<qc::NonUnitaryOperation*>(op4.get());
  ASSERT_NE(measure0, nullptr);
  const auto& classics0 = measure0->getClassics();
  EXPECT_EQ(classics0.size(), 1);
  EXPECT_EQ(classics0.at(0), 0);
}

TEST_F(QFRFunctionality, deferMeasurementsCorrectOrder) {
  // Input:
  // i: 			0	1
  // 1: 	H   	H 	|
  // 2: 	Meas	0	|
  // 3: 	H   	| 	H
  // 4: 	c_X   	|	X 		c[0] == 1
  // o: 			0	1

  // Expected Output:
  // i: 			0	1
  // 1: 	H   	H 	|
  // 2: 	H   	| 	H
  // 3: 	X   	c	X
  // 4: 	Meas	0	|
  // o: 			0	1

  QuantumComputation qc{};
  qc.addQubitRegister(2);
  qc.addClassicalRegister(1);
  qc.h(0);
  qc.measure(0, 0U);
  qc.h(1);
  qc.classicControlled(qc::X, 1, {0, 1U}, 1U);
  std::cout << qc << "\n";

  EXPECT_TRUE(CircuitOptimizer::isDynamicCircuit(qc));

  EXPECT_NO_THROW(CircuitOptimizer::deferMeasurements(qc););

  std::cout << qc << "\n";

  EXPECT_FALSE(CircuitOptimizer::isDynamicCircuit(qc));

  ASSERT_EQ(qc.getNqubits(), 2);
  ASSERT_EQ(qc.getNindividualOps(), 4);
  const auto& op0 = qc.at(0);
  const auto& op1 = qc.at(1);
  const auto& op2 = qc.at(2);
  const auto& op3 = qc.at(3);

  EXPECT_EQ(op0->getNqubits(), 2);
  EXPECT_TRUE(op0->getType() == qc::H);
  const auto& targets0 = op0->getTargets();
  EXPECT_EQ(targets0.size(), 1);
  EXPECT_EQ(targets0.at(0), static_cast<Qubit>(0));
  EXPECT_TRUE(op0->getControls().empty());

  EXPECT_EQ(op1->getNqubits(), 2);
  EXPECT_TRUE(op1->getType() == qc::H);
  const auto& targets1 = op2->getTargets();
  EXPECT_EQ(targets1.size(), 1);
  EXPECT_EQ(targets1.at(0), static_cast<Qubit>(1));
  EXPECT_TRUE(op1->getControls().empty());

  EXPECT_EQ(op2->getNqubits(), 2);
  EXPECT_TRUE(op2->getType() == qc::X);
  const auto& targets2 = op1->getTargets();
  EXPECT_EQ(targets2.size(), 1);
  EXPECT_EQ(targets2.at(0), static_cast<Qubit>(1));
  const auto& controls2 = op2->getControls();
  EXPECT_EQ(controls2.size(), 1);
  EXPECT_EQ(controls2.count(0), 1);

  EXPECT_EQ(op3->getNqubits(), 2);
  ASSERT_TRUE(op3->getType() == qc::Measure);
  const auto& targets3 = op3->getTargets();
  EXPECT_EQ(targets3.size(), 1);
  EXPECT_EQ(targets3.at(0), static_cast<Qubit>(0));
  auto* measure0 = dynamic_cast<qc::NonUnitaryOperation*>(op3.get());
  ASSERT_NE(measure0, nullptr);
  const auto& classics0 = measure0->getClassics();
  EXPECT_EQ(classics0.size(), 1);
  EXPECT_EQ(classics0.at(0), 0);
}

TEST_F(QFRFunctionality, deferMeasurementsTwoClassicCorrectOrder) {
  // Input:
  // i: 			0	1
  // 1: 	H   	H 	|
  // 2: 	Meas	0	|
  // 3: 	H   	| 	H
  // 4: 	c_X   	|	X 		c[0] == 1
  // 5:    c_Z     |   Z       c[0] == 1
  // o: 			0	1

  // Expected Output:
  // i: 			0	1
  // 1: 	H   	H 	|
  // 2: 	H   	| 	H
  // 3: 	X   	c	X
  // 4: 	Z   	c	Z
  // 5: 	Meas	0	|
  // o: 			0	1

  QuantumComputation qc{};
  qc.addQubitRegister(2);
  qc.addClassicalRegister(1);
  qc.h(0);
  qc.measure(0, 0U);
  qc.h(1);
  qc.classicControlled(qc::X, 1, {0, 1U}, 1U);
  qc.classicControlled(qc::Z, 1, {0, 1U}, 1U);
  std::cout << qc << "\n";

  EXPECT_TRUE(CircuitOptimizer::isDynamicCircuit(qc));

  EXPECT_NO_THROW(CircuitOptimizer::deferMeasurements(qc););

  std::cout << qc << "\n";

  EXPECT_FALSE(CircuitOptimizer::isDynamicCircuit(qc));

  ASSERT_EQ(qc.getNqubits(), 2);
  ASSERT_EQ(qc.getNindividualOps(), 5);
  const auto& op0 = qc.at(0);
  const auto& op1 = qc.at(1);
  const auto& op2 = qc.at(2);
  const auto& op3 = qc.at(3);
  const auto& op4 = qc.at(4);

  EXPECT_EQ(op0->getNqubits(), 2);
  EXPECT_TRUE(op0->getType() == qc::H);
  const auto& targets0 = op0->getTargets();
  EXPECT_EQ(targets0.size(), 1);
  EXPECT_EQ(targets0.at(0), static_cast<Qubit>(0));
  EXPECT_TRUE(op0->getControls().empty());

  EXPECT_EQ(op1->getNqubits(), 2);
  EXPECT_TRUE(op1->getType() == qc::H);
  const auto& targets1 = op1->getTargets();
  EXPECT_EQ(targets1.size(), 1);
  EXPECT_EQ(targets1.at(0), static_cast<Qubit>(1));
  EXPECT_TRUE(op1->getControls().empty());

  EXPECT_EQ(op2->getNqubits(), 2);
  EXPECT_TRUE(op2->getType() == qc::X);
  const auto& targets2 = op2->getTargets();
  EXPECT_EQ(targets2.size(), 1);
  EXPECT_EQ(targets2.at(0), static_cast<Qubit>(1));
  const auto& controls2 = op2->getControls();
  EXPECT_EQ(controls2.size(), 1);
  EXPECT_EQ(controls2.count(0), 1);

  EXPECT_EQ(op3->getNqubits(), 2);
  EXPECT_TRUE(op3->getType() == qc::Z);
  const auto& targets3 = op3->getTargets();
  EXPECT_EQ(targets3.size(), 1);
  EXPECT_EQ(targets3.at(0), static_cast<Qubit>(1));
  const auto& controls3 = op3->getControls();
  EXPECT_EQ(controls3.size(), 1);
  EXPECT_EQ(controls3.count(0), 1);

  EXPECT_EQ(op4->getNqubits(), 2);
  ASSERT_TRUE(op4->getType() == qc::Measure);
  const auto& targets4 = op4->getTargets();
  EXPECT_EQ(targets4.size(), 1);
  EXPECT_EQ(targets4.at(0), static_cast<Qubit>(0));
  auto* measure0 = dynamic_cast<qc::NonUnitaryOperation*>(op4.get());
  ASSERT_NE(measure0, nullptr);
  const auto& classics0 = measure0->getClassics();
  EXPECT_EQ(classics0.size(), 1);
  EXPECT_EQ(classics0.at(0), 0);
}

TEST_F(QFRFunctionality, deferMeasurementsErrorOnImplicitReset) {
  // Input:
  // i: 			0
  // 1: 	H   	H
  // 2: 	Meas	0
  // 3: 	c_X   	X	c[0] == 1
  // o: 			0

  // Expected Output:
  // Error, since the classic-controlled operation targets the qubit being
  // measured (this implicitly realizes a reset operation)

  QuantumComputation qc(1U, 1U);
  qc.h(0);
  qc.measure(0, 0U);
  qc.classicControlled(qc::X, 0, {0, 1U}, 1U);
  std::cout << qc << "\n";

  EXPECT_TRUE(CircuitOptimizer::isDynamicCircuit(qc));

  EXPECT_THROW(CircuitOptimizer::deferMeasurements(qc), qc::QFRException);
}

TEST_F(QFRFunctionality, trivialOperationReordering) {
  QuantumComputation qc(2);
  qc.h(0);
  qc.h(1);
  std::cout << qc << "\n";
  qc::CircuitOptimizer::reorderOperations(qc);
  std::cout << qc << "\n";
  auto it = qc.begin();
  const auto target = (*it)->getTargets().at(0);
  EXPECT_EQ(target, 1);
  ++it;
  const auto target2 = (*it)->getTargets().at(0);
  EXPECT_EQ(target2, 0);
}

TEST_F(QFRFunctionality, OperationReorderingBarrier) {
  QuantumComputation qc(2);
  qc.h(0);
  qc.barrier({0, 1});
  qc.h(1);
  std::cout << qc << "\n";
  qc::CircuitOptimizer::reorderOperations(qc);
  std::cout << qc << "\n";
  auto it = qc.begin();
  const auto target = (*it)->getTargets().at(0);
  EXPECT_EQ(target, 0);
  ++it;
  ++it;
  const auto target2 = (*it)->getTargets().at(0);
  EXPECT_EQ(target2, 1);
}

TEST_F(QFRFunctionality, FlattenRandomClifford) {
  qc::RandomCliffordCircuit rcs(2U, 3U, 0U);
  std::cout << rcs << "\n";
  const auto nops = rcs.getNindividualOps();

  qc::CircuitOptimizer::flattenOperations(rcs);
  std::cout << rcs << "\n";

  for (const auto& op : rcs) {
    EXPECT_FALSE(op->isCompoundOperation());
  }
  EXPECT_EQ(nops, rcs.getNindividualOps());
}

TEST_F(QFRFunctionality, FlattenRecursive) {
  const std::size_t nqubits = 1U;

  // create a nested compound operation
  QuantumComputation op(nqubits);
  op.x(0);
  QuantumComputation op2(nqubits);
  op2.emplace_back(op.asCompoundOperation());
  QuantumComputation qc(nqubits);
  qc.emplace_back(op2.asCompoundOperation());
  std::cout << qc << "\n";

  qc::CircuitOptimizer::flattenOperations(qc);
  std::cout << qc << "\n";

  for (const auto& g : qc) {
    EXPECT_FALSE(g->isCompoundOperation());
  }

  auto& gate = **qc.begin();
  EXPECT_EQ(gate.getType(), qc::X);
  EXPECT_EQ(gate.getTargets().at(0), 0U);
  EXPECT_TRUE(gate.getControls().empty());
}

TEST_F(QFRFunctionality, OperationEquality) {
  const auto x = StandardOperation(1U, 0, qc::X);
  const auto z = StandardOperation(1U, 0, qc::Z);
  EXPECT_TRUE(x.equals(x));
  EXPECT_FALSE(x.equals(z));

  const auto x0 = StandardOperation(2U, 0, qc::X);
  const auto x1 = StandardOperation(2U, 1, qc::X);
  EXPECT_FALSE(x0.equals(x1));
  Permutation perm0{};
  perm0[0] = 1;
  perm0[1] = 0;
  EXPECT_TRUE(x0.equals(x1, perm0, {}));
  EXPECT_TRUE(x0.equals(x1, {}, perm0));

  const auto cx01 = StandardOperation(2U, 0, 1, qc::X);
  const auto cx10 = StandardOperation(2U, 1, 0, qc::X);
  EXPECT_FALSE(cx01.equals(cx10));
  EXPECT_FALSE(x0.equals(cx01));

  const auto p = StandardOperation(1U, 0, qc::P, {2.0});
  const auto pm = StandardOperation(1U, 0, qc::P, {-2.0});
  EXPECT_FALSE(p.equals(pm));

  const auto measure0 = NonUnitaryOperation(2U, 0, 0U);
  const auto measure1 = NonUnitaryOperation(2U, 0, 1U);
  const auto measure2 = NonUnitaryOperation(2U, 1, 0U);
  EXPECT_FALSE(measure0.equals(x0));
  EXPECT_TRUE(measure0.equals(measure0));
  EXPECT_FALSE(measure0.equals(measure1));
  EXPECT_FALSE(measure0.equals(measure2));
  EXPECT_TRUE(measure0.equals(measure2, perm0, {}));
  EXPECT_TRUE(measure0.equals(measure2, {}, perm0));

  const auto controlRegister0 = qc::QuantumRegister{0, 1U};
  const auto controlRegister1 = qc::QuantumRegister{1, 1U};
  const auto expectedValue0 = 0U;
  const auto expectedValue1 = 1U;

  std::unique_ptr<Operation> xp0 =
      std::make_unique<StandardOperation>(1U, 0, qc::X);
  std::unique_ptr<Operation> xp1 =
      std::make_unique<StandardOperation>(1U, 0, qc::X);
  std::unique_ptr<Operation> xp2 =
      std::make_unique<StandardOperation>(1U, 0, qc::X);
  const auto classic0 =
      ClassicControlledOperation(xp0, controlRegister0, expectedValue0);
  const auto classic1 =
      ClassicControlledOperation(xp1, controlRegister0, expectedValue1);
  const auto classic2 =
      ClassicControlledOperation(xp2, controlRegister1, expectedValue0);
  std::unique_ptr<Operation> zp =
      std::make_unique<StandardOperation>(1U, 0, qc::Z);
  const auto classic3 =
      ClassicControlledOperation(zp, controlRegister0, expectedValue0);
  EXPECT_FALSE(classic0.equals(x));
  EXPECT_TRUE(classic0.equals(classic0));
  EXPECT_FALSE(classic0.equals(classic1));
  EXPECT_FALSE(classic0.equals(classic2));
  EXPECT_FALSE(classic0.equals(classic3));

  auto compound0 = CompoundOperation(1U);
  compound0.emplace_back<StandardOperation>(1U, 0, qc::X);

  auto compound1 = CompoundOperation(1U);
  compound1.emplace_back<StandardOperation>(1U, 0, qc::X);
  compound1.emplace_back<StandardOperation>(1U, 0, qc::Z);

  auto compound2 = CompoundOperation(1U);
  compound2.emplace_back<StandardOperation>(1U, 0, qc::Z);

  EXPECT_FALSE(compound0.equals(x));
  EXPECT_TRUE(compound0.equals(compound0));
  EXPECT_FALSE(compound0.equals(compound1));
  EXPECT_FALSE(compound0.equals(compound2));
}

TEST_F(QFRFunctionality, CNOTCancellation1) {
  QuantumComputation qc(2);
  qc.cx(1, 0);
  qc.cx(1, 0);

  CircuitOptimizer::cancelCNOTs(qc);
  EXPECT_TRUE(qc.empty());
}

TEST_F(QFRFunctionality, CNOTCancellation2) {
  QuantumComputation qc(2);
  qc.swap(0, 1);
  qc.swap(1, 0);

  CircuitOptimizer::cancelCNOTs(qc);
  EXPECT_TRUE(qc.empty());
}

TEST_F(QFRFunctionality, CNOTCancellation3) {
  QuantumComputation qc(2);
  qc.swap(0, 1);
  qc.cx(1, 0);

  CircuitOptimizer::cancelCNOTs(qc);
  EXPECT_TRUE(qc.size() == 2U);
  const auto& firstOperation = qc.front();
  EXPECT_EQ(firstOperation->getType(), qc::X);
  EXPECT_EQ(firstOperation->getTargets().front(), 0U);
  EXPECT_EQ(firstOperation->getControls().begin()->qubit, 1U);

  const auto& secondOperation = qc.back();
  EXPECT_EQ(secondOperation->getType(), qc::X);
  EXPECT_EQ(secondOperation->getTargets().front(), 1U);
  EXPECT_EQ(secondOperation->getControls().begin()->qubit, 0U);
}

TEST_F(QFRFunctionality, CNOTCancellation4) {
  QuantumComputation qc(2);
  qc.cx(1, 0);
  qc.swap(0, 1);

  CircuitOptimizer::cancelCNOTs(qc);
  EXPECT_TRUE(qc.size() == 2U);
  const auto& firstOperation = qc.front();
  EXPECT_EQ(firstOperation->getType(), qc::X);
  EXPECT_EQ(firstOperation->getTargets().front(), 1U);
  EXPECT_EQ(firstOperation->getControls().begin()->qubit, 0U);

  const auto& secondOperation = qc.back();
  EXPECT_EQ(secondOperation->getType(), qc::X);
  EXPECT_EQ(secondOperation->getTargets().front(), 0U);
  EXPECT_EQ(secondOperation->getControls().begin()->qubit, 1U);
}

TEST_F(QFRFunctionality, CNOTCancellation5) {
  QuantumComputation qc(2);
  qc.cx(1, 0);
  qc.cx(0, 1);
  qc.cx(1, 0);

  CircuitOptimizer::cancelCNOTs(qc);
  EXPECT_TRUE(qc.size() == 1U);
  const auto& firstOperation = qc.front();
  EXPECT_EQ(firstOperation->getType(), qc::SWAP);
  EXPECT_EQ(firstOperation->getTargets().front(), 0U);
  EXPECT_EQ(firstOperation->getTargets().back(), 1U);
}

TEST_F(QFRFunctionality, IndexOutOfRange) {
  QuantumComputation qc(2);
  qc::Permutation layout{};
  layout[0] = 0;
  layout[2] = 1;
  qc.initialLayout = layout;
  qc.x(0);

  EXPECT_THROW(qc.x(1), QFRException);
  EXPECT_THROW(qc.cx(1_nc, 0), QFRException);
  EXPECT_THROW(qc.mcx({2_nc, 1_nc}, 0), QFRException);
  EXPECT_THROW(qc.swap(0, 1), QFRException);
  EXPECT_THROW(qc.cswap(1_nc, 0, 2), QFRException);
  EXPECT_THROW(qc.reset({0, 1, 2}), QFRException);
}

TEST_F(QFRFunctionality, ContainsLogicalQubit) {
  const QuantumComputation qc(2);
  const auto [contains0, index0] = qc.containsLogicalQubit(0);
  EXPECT_TRUE(contains0);
  const auto hasValue0 = index0.has_value();
  ASSERT_TRUE(hasValue0);
  if (hasValue0) {
    EXPECT_EQ(*index0, 0);
  }
  const auto [contains1, index1] = qc.containsLogicalQubit(1);
  EXPECT_TRUE(contains1);
  const auto hasValue1 = index1.has_value();
  ASSERT_TRUE(hasValue1);
  if (hasValue1) {
    EXPECT_EQ(*index1, 1);
  }
  const auto [contains2, index2] = qc.containsLogicalQubit(2);
  EXPECT_FALSE(contains2);
  EXPECT_FALSE(index2.has_value());
}

TEST_F(QFRFunctionality, AddAncillaryQubits) {
  QuantumComputation qc(1);
  qc.addAncillaryQubit(1, std::nullopt);
  EXPECT_EQ(qc.getNqubits(), 2);
  EXPECT_EQ(qc.getNancillae(), 1);
  ASSERT_EQ(qc.ancillary.size(), 2U);
  ASSERT_EQ(qc.garbage.size(), 2U);
  EXPECT_FALSE(qc.ancillary[0]);
  EXPECT_TRUE(qc.ancillary[1]);
  EXPECT_FALSE(qc.garbage[0]);
  EXPECT_TRUE(qc.garbage[1]);
}

TEST_F(QFRFunctionality, SingleQubitGateCount) {
  QuantumComputation qc(2U, 2U);
  qc.x(0);
  qc.h(0);
  qc.cx(1, 0);
  qc.z(0);
  qc.measure(0, 0);

  EXPECT_EQ(qc.getNops(), 5U);
  EXPECT_EQ(qc.getNindividualOps(), 5U);
  EXPECT_EQ(qc.getNsingleQubitOps(), 3U);

  CircuitOptimizer::singleQubitGateFusion(qc);

  EXPECT_EQ(qc.getNops(), 4U);
  EXPECT_EQ(qc.getNindividualOps(), 5U);
  EXPECT_EQ(qc.getNsingleQubitOps(), 3U);
}

TEST_F(QFRFunctionality, CircuitDepthEmptyCircuit) {
  const QuantumComputation qc(2);
  EXPECT_EQ(qc.getDepth(), 0U);
}

TEST_F(QFRFunctionality, CircuitDepthStandardOperations) {
  QuantumComputation qc(2);
  qc.h(0);
  qc.h(1);
  qc.cx(1, 0);

  EXPECT_EQ(qc.getDepth(), 2U);
}

TEST_F(QFRFunctionality, CircuitDepthNonUnitaryOperations) {
  QuantumComputation qc(2U, 2U);
  qc.h(0);
  qc.h(1);
  qc.cx(1, 0);
  qc.barrier({0, 1});
  qc.measure(0, 0);
  qc.measure(1, 1);

  EXPECT_EQ(qc.getDepth(), 3U);
}

// Test with compound operation
TEST_F(QFRFunctionality, CircuitDepthCompoundOperation) {
  QuantumComputation comp(2);
  comp.h(0);
  comp.h(1);
  comp.cx(1, 0);

  QuantumComputation qc(2);
  qc.emplace_back(comp.asOperation());

  EXPECT_EQ(qc.getDepth(), 2U);
}

TEST_F(QFRFunctionality, CircuitToOperation) {
  QuantumComputation qc(2U, 2U);
  EXPECT_EQ(qc.asOperation(), nullptr);
  qc.x(0);
  const auto& op = qc.asOperation();
  EXPECT_EQ(op->getType(), qc::X);
  EXPECT_EQ(op->getNqubits(), 2U);
  EXPECT_EQ(op->getNcontrols(), 0U);
  EXPECT_EQ(op->getTargets().front(), 0U);
  EXPECT_TRUE(qc.empty());
  qc.x(0);
  qc.h(0);
  qc.classicControlled(qc::X, 0, 1, {0, 1U}, 1U);
  const auto& op2 = qc.asOperation();
  EXPECT_EQ(op2->getType(), qc::Compound);
  EXPECT_EQ(op2->getNqubits(), 2U);
  EXPECT_TRUE(qc.empty());
}

TEST_F(QFRFunctionality, AvoidStrippingIdleQubitWhenInOutputPermutation) {
  // a qubit being present in the output permutation should not be stripped
  QuantumComputation qc(2U, 2U);
  qc.measure(1, 0);
  qc.initializeIOMapping();
  qc.stripIdleQubits();
  EXPECT_EQ(qc.getNqubits(), 2U);
  EXPECT_EQ(qc.outputPermutation[1], 0U);
}

TEST_F(QFRFunctionality, RzAndPhaseDifference) {
  QuantumComputation qc(2);
  const std::string qasm = "// i 0 1\n"
                           "// o 0 1\n"
                           "OPENQASM 2.0;\n"
                           "include \"qelib1.inc\";\n"
                           "qreg q[2];\n"
                           "rz(1/8) q[0];\n"
                           "p(1/8) q[1];\n"
                           "crz(1/8) q[0],q[1];\n"
                           "cp(1/8) q[0],q[1];\n";
  std::stringstream ss;
  ss << qasm;
  qc.import(ss, qc::Format::OpenQASM);
  std::cout << qc << "\n";
  std::stringstream oss;
  qc.dumpOpenQASM(oss);
}

TEST_F(QFRFunctionality, U3toU2Gate) {
  QuantumComputation qc(1);
  qc.u(PI_2, 0., PI, 0);      // H
  qc.u(PI_2, 0., 0., 0);      // RY(pi/2)
  qc.u(PI_2, -PI_2, PI_2, 0); // V = RX(pi/2)
  qc.u(PI_2, PI_2, -PI_2, 0); // Vdag = RX(-pi/2)
  qc.u(PI_2, 0.25, 0.5, 0);   // U2(0.25, 0.5)
  std::cout << qc << "\n";
  EXPECT_EQ(qc.at(0)->getType(), qc::H);
  EXPECT_EQ(qc.at(1)->getType(), qc::RY);
  EXPECT_EQ(qc.at(1)->getParameter().at(0), PI_2);
  EXPECT_EQ(qc.at(2)->getType(), qc::V);
  EXPECT_EQ(qc.at(3)->getType(), qc::Vdg);
  EXPECT_EQ(qc.at(4)->getType(), qc::U2);
  EXPECT_EQ(qc.at(4)->getParameter().at(0), 0.25);
  EXPECT_EQ(qc.at(4)->getParameter().at(1), 0.5);
}

TEST_F(QFRFunctionality, U3toU1Gate) {
  QuantumComputation qc(1);
  qc.u(0., 0., 0., 0);    // I
  qc.u(0., 0., PI, 0);    // Z
  qc.u(0., 0., PI_2, 0);  // S
  qc.u(0., 0., -PI_2, 0); // Sdg
  qc.u(0., 0., PI_4, 0);  // T
  qc.u(0., 0., -PI_4, 0); // Tdg
  qc.u(0., 0., 0.5, 0);   // p(0.5)

  std::cout << qc << "\n";
  EXPECT_EQ(qc.at(0)->getType(), qc::I);
  EXPECT_EQ(qc.at(1)->getType(), qc::Z);
  EXPECT_EQ(qc.at(2)->getType(), qc::S);
  EXPECT_EQ(qc.at(3)->getType(), qc::Sdg);
  EXPECT_EQ(qc.at(4)->getType(), qc::T);
  EXPECT_EQ(qc.at(5)->getType(), qc::Tdg);
  EXPECT_EQ(qc.at(6)->getType(), qc::P);
  EXPECT_EQ(qc.at(6)->getParameter().at(0), 0.5);
}

TEST_F(QFRFunctionality, U3SpecialCases) {
  QuantumComputation qc(1);
  qc.u(0.5, 0., 0., 0);      // RY(0.5)
  qc.u(0.5, -PI_2, PI_2, 0); // RX(0.5)
  qc.u(0.5, PI_2, -PI_2, 0); // RX(-0.5)
  qc.u(PI, PI_2, PI_2, 0);   // Y
  qc.u(PI, 0., PI, 0);       // X
  qc.u(0.5, 0.25, 0.125, 0); // U3(0.5, 0.25, 0.125)

  std::cout << qc << "\n";
  EXPECT_EQ(qc.at(0)->getType(), qc::RY);
  EXPECT_EQ(qc.at(0)->getParameter().at(0), 0.5);
  EXPECT_EQ(qc.at(1)->getType(), qc::RX);
  EXPECT_EQ(qc.at(1)->getParameter().at(0), 0.5);
  EXPECT_EQ(qc.at(2)->getType(), qc::RX);
  EXPECT_EQ(qc.at(2)->getParameter().at(0), -0.5);
  EXPECT_EQ(qc.at(3)->getType(), qc::Y);
  EXPECT_EQ(qc.at(4)->getType(), qc::X);
  EXPECT_EQ(qc.at(5)->getType(), qc::U);
  EXPECT_EQ(qc.at(5)->getParameter().at(0), 0.5);
  EXPECT_EQ(qc.at(5)->getParameter().at(1), 0.25);
  EXPECT_EQ(qc.at(5)->getParameter().at(2), 0.125);
}

TEST_F(QFRFunctionality, GlobalPhaseNormalization) {
  QuantumComputation qc(1);
  EXPECT_EQ(qc.getGlobalPhase(), 0.);
  qc.gphase(-PI);
  EXPECT_EQ(qc.getGlobalPhase(), PI);
  qc.gphase(PI);
  EXPECT_EQ(qc.getGlobalPhase(), 0.);
}

TEST_F(QFRFunctionality, OpNameToTypeSimple) {
  EXPECT_EQ(qc::OpType::X, qc::opTypeFromString("x"));
  EXPECT_EQ(qc::OpType::Y, qc::opTypeFromString("y"));
  EXPECT_EQ(qc::OpType::Z, qc::opTypeFromString("z"));

  EXPECT_EQ(qc::OpType::H, qc::opTypeFromString("h"));
  EXPECT_EQ(qc::OpType::S, qc::opTypeFromString("s"));
  EXPECT_EQ(qc::OpType::Sdg, qc::opTypeFromString("sdg"));
  EXPECT_EQ(qc::OpType::T, qc::opTypeFromString("t"));
  EXPECT_EQ(qc::OpType::Tdg, qc::opTypeFromString("tdg"));

  EXPECT_EQ(qc::OpType::X, qc::opTypeFromString("cnot"));

  EXPECT_THROW([[maybe_unused]] const auto type = qc::opTypeFromString("foo"),
               std::invalid_argument);
}

TEST_F(QFRFunctionality, dumpAndImportTeleportation) {
  QuantumComputation qc(3);
  qc.emplace_back<StandardOperation>(3, Targets{0, 1, 2},
                                     OpType::Teleportation);
  std::stringstream ss;
  qc.dumpOpenQASM(ss);
  EXPECT_TRUE(ss.str().find("teleport") != std::string::npos);

  QuantumComputation qcImported(3);
  qcImported.import(ss, qc::Format::OpenQASM);
  ASSERT_EQ(qcImported.size(), 1);
  EXPECT_EQ(qcImported.at(0)->getType(), OpType::Teleportation);
}

TEST_F(QFRFunctionality, addControlStandardOperation) {
  auto op = StandardOperation(3, 0, OpType::X);
  op.addControl(1);
  op.addControl(2);
  ASSERT_EQ(op.getNcontrols(), 2);
  const auto expectedControls = Controls{1, 2};
  EXPECT_EQ(op.getControls(), expectedControls);
  op.removeControl(1);
  const auto expectedControlsAfterRemove = Controls{2};
  EXPECT_EQ(op.getControls(), expectedControlsAfterRemove);
  op.clearControls();
  EXPECT_EQ(op.getNcontrols(), 0);
  ASSERT_THROW(op.removeControl(1), QFRException);

  op.addControl(1);
  const auto& controls = op.getControls();
  EXPECT_EQ(op.removeControl(controls.begin()), controls.end());
}

TEST_F(QFRFunctionality, addControlSymbolicOperation) {
  auto op = SymbolicOperation(3, 0, OpType::X);

  op.addControl(1);
  op.addControl(2);

  ASSERT_EQ(op.getNcontrols(), 2);
  auto expectedControls = Controls{1, 2};
  EXPECT_EQ(op.getControls(), expectedControls);
  op.removeControl(1);
  auto expectedControlsAfterRemove = Controls{2};
  EXPECT_EQ(op.getControls(), expectedControlsAfterRemove);
  op.clearControls();
  EXPECT_EQ(op.getNcontrols(), 0);

  op.addControl(1);
  const auto& controls = op.getControls();
  EXPECT_EQ(op.removeControl(controls.begin()), controls.end());
}

TEST_F(QFRFunctionality, addControlClassicControlledOperation) {
  std::unique_ptr<Operation> xp =
      std::make_unique<StandardOperation>(1U, 0, qc::X);
  const auto controlRegister = qc::QuantumRegister{0, 1U};
  const auto expectedValue = 0U;
  auto op = ClassicControlledOperation(xp, controlRegister, expectedValue);

  op.addControl(1);
  op.addControl(2);

  ASSERT_EQ(op.getNcontrols(), 2);
  auto expectedControls = Controls{1, 2};
  EXPECT_EQ(op.getControls(), expectedControls);
  op.removeControl(1);
  auto expectedControlsAfterRemove = Controls{2};
  EXPECT_EQ(op.getControls(), expectedControlsAfterRemove);
  op.clearControls();
  EXPECT_EQ(op.getNcontrols(), 0);
  op.addControl(1);
  const auto& controls = op.getControls();
  EXPECT_EQ(op.removeControl(controls.begin()), controls.end());
}

TEST_F(QFRFunctionality, addControlNonUnitaryOperation) {
  auto op = NonUnitaryOperation(1U, 0U, Measure);

  EXPECT_THROW(static_cast<void>(op.getControls()), QFRException);
  EXPECT_THROW(op.addControl(1), QFRException);
  EXPECT_THROW(op.removeControl(1), QFRException);
  EXPECT_THROW(op.clearControls(), QFRException);
  // we pass an invalid iterator to removeControl, which is fine, since the
  // function call should unconditionally trap
  EXPECT_THROW(op.removeControl(Controls::const_iterator{}), QFRException);
}

TEST_F(QFRFunctionality, addControlCompundOperation) {
  auto op = CompoundOperation(4);

  auto control0 = 0U;
  auto control1 = 1U;

  auto xOp = std::make_unique<StandardOperation>(4, Targets{1}, OpType::X);
  auto cxOp = std::make_unique<StandardOperation>(4, Targets{3}, OpType::X);
  cxOp->addControl(control1);

  op.emplace_back(xOp);
  op.emplace_back(cxOp);

  op.addControl(control0);

  ASSERT_EQ(op.getOps()[0]->getNcontrols(), 1);
  ASSERT_EQ(op.getOps()[1]->getNcontrols(), 2);

  op.clearControls();

  ASSERT_EQ(op.getOps()[0]->getNcontrols(), 0);
  ASSERT_EQ(op.getOps()[1]->getNcontrols(), 1);
  ASSERT_EQ(*op.getOps()[1]->getControls().begin(), control1);
  EXPECT_THROW(op.removeControl(control0), QFRException);
}

TEST_F(QFRFunctionality, addControlTwice) {
  auto control = 0U;

  std::unique_ptr<Operation> op =
      std::make_unique<StandardOperation>(2, Targets{1}, OpType::X);
  op->addControl(control);
  EXPECT_THROW(op->addControl(control), QFRException);

  auto classicControlledOp =
      ClassicControlledOperation(op, qc::QuantumRegister{0, 1U}, 0U);
  EXPECT_THROW(classicControlledOp.addControl(control), QFRException);

  auto symbolicOp = SymbolicOperation(2, Targets{1}, OpType::X);
  symbolicOp.addControl(control);
  EXPECT_THROW(symbolicOp.addControl(control), QFRException);
}

TEST_F(QFRFunctionality, addTargetAsControl) {
  // Adding a control that is already a target
  auto control = 1U;

  std::unique_ptr<Operation> op =
      std::make_unique<StandardOperation>(2, Targets{1}, OpType::X);
  EXPECT_THROW(op->addControl(control), QFRException);

  auto classicControlledOp =
      ClassicControlledOperation(op, qc::QuantumRegister{0, 1U}, 0U);
  EXPECT_THROW(classicControlledOp.addControl(control), QFRException);

  auto symbolicOp = SymbolicOperation(2, Targets{1}, OpType::X);
  EXPECT_THROW(symbolicOp.addControl(control), QFRException);
}

TEST_F(QFRFunctionality, addControlCompundOperationInvalid) {
  auto op = CompoundOperation(4);

  auto control1 = 1U;

  auto xOp = std::make_unique<StandardOperation>(4, Targets{1}, OpType::X);
  auto cxOp = std::make_unique<StandardOperation>(4, Targets{3}, OpType::X);
  cxOp->addControl(control1);

  op.emplace_back(xOp);
  op.emplace_back(cxOp);

  ASSERT_THROW(op.addControl(control1), QFRException);
  ASSERT_THROW(op.addControl(Control{1}), QFRException);
}

TEST_F(QFRFunctionality, invertUnsupportedOperation) {
  auto op = NonUnitaryOperation(1U, 0U, OpType::Measure);

  ASSERT_THROW(op.invert(), QFRException);
}

TEST_F(QFRFunctionality, invertStandardOpSelfInverting) {
  const auto opTypes = {
      OpType::I, OpType::X, OpType::Y, OpType::Z, OpType::H, OpType::SWAP,
  };

  for (auto opType : opTypes) {
    auto op = StandardOperation(1U, 0U, opType);
    op.invert();
    ASSERT_EQ(op.getType(), opType);
  }
}

TEST_F(QFRFunctionality, invertStandardOpInvertClone) {
  auto op1 = StandardOperation(1U, 0U, S);
  auto op2 = op1.getInverted();
  ASSERT_EQ(op1.getType(), S);
  ASSERT_EQ(op2->getType(), Sdg);
}

TEST_F(QFRFunctionality, invertStandardOpSpecial) {
  const auto opTypes = {
      std::pair{S, Sdg},   std::pair{T, Tdg},         std::pair{V, Vdg},
      std::pair{SX, SXdg}, std::pair{Peres, Peresdg},
  };

  for (const auto& [opType, opTypeInv] : opTypes) {
    auto op = StandardOperation(1U, 0U, opType);
    op.invert();
    ASSERT_EQ(op.getType(), opTypeInv);

    auto op2 = StandardOperation(1U, 0U, opTypeInv);
    op2.invert();
    ASSERT_EQ(op2.getType(), opType);
  }
}

TEST_F(QFRFunctionality, invertStandardOpParamChange) {
  const auto cases = {
      std::tuple{OpType::GPhase, std::vector<fp>{1}, std::vector<fp>{-1}},
      std::tuple{OpType::P, std::vector<fp>{1}, std::vector<fp>{-1}},
      std::tuple{OpType::RX, std::vector<fp>{1}, std::vector<fp>{-1}},
      std::tuple{OpType::RY, std::vector<fp>{1}, std::vector<fp>{-1}},
      std::tuple{OpType::RZ, std::vector<fp>{1}, std::vector<fp>{-1}},
      std::tuple{OpType::RXX, std::vector<fp>{1}, std::vector<fp>{-1}},
      std::tuple{OpType::RYY, std::vector<fp>{1}, std::vector<fp>{-1}},
      std::tuple{OpType::RZZ, std::vector<fp>{1}, std::vector<fp>{-1}},
      std::tuple{OpType::RZX, std::vector<fp>{1}, std::vector<fp>{-1}},
      std::tuple{OpType::U2, std::vector<fp>{1, 1},
                 std::vector<fp>{-1 + PI, -1 - PI}},
      std::tuple{OpType::U, std::vector<fp>{1, 2, 3},
                 std::vector<fp>{-1, -3, -2}},
      std::tuple{OpType::XXminusYY, std::vector<fp>{1}, std::vector<fp>{-1}},
      std::tuple{OpType::XXplusYY, std::vector<fp>{1}, std::vector<fp>{-1}},
  };

  for (const auto& testcase : cases) {
    auto op =
        StandardOperation(1U, 0U, std::get<0>(testcase), std::get<1>(testcase));
    op.invert();
    ASSERT_EQ(op.getParameter(), std::get<2>(testcase));
  }

  auto op = StandardOperation(2U, Targets{0U, 1U}, OpType::DCX);
  op.invert();
  const auto expectedTargets = Targets{1U, 0U};
  ASSERT_EQ(op.getTargets(), expectedTargets);
}

TEST_F(QFRFunctionality, invertCompoundOperation) {
  auto op = CompoundOperation(4);

  op.emplace_back<StandardOperation>(4U, 0U, OpType::X);
  op.emplace_back<StandardOperation>(4U, 1U, OpType::RZ, std::vector<fp>{1});
  op.emplace_back<StandardOperation>(4U, 1U, OpType::S);

  op.invert();

  ASSERT_EQ(op.getOps()[0]->getType(), OpType::Sdg);
  ASSERT_EQ(op.getOps()[1]->getType(), OpType::RZ);
  ASSERT_EQ(op.getOps()[1]->getParameter(), std::vector<fp>{-1});
  ASSERT_EQ(op.getOps()[2]->getType(), OpType::X);
}

TEST_F(QFRFunctionality, invertSymbolicOpParamChange) {
  auto x = sym::Variable("x");
  auto y = sym::Variable("y");
  const auto cases = {
      std::tuple{OpType::GPhase, std::vector<SymbolOrNumber>{Symbolic({x})},
                 std::vector<SymbolOrNumber>{-Symbolic({x})}},
      std::tuple{OpType::GPhase, std::vector<SymbolOrNumber>{1.0},
                 std::vector<SymbolOrNumber>{-1.0}},
      std::tuple{OpType::U2, std::vector<SymbolOrNumber>{Symbolic({x}), 1.0},
                 std::vector<SymbolOrNumber>{-1.0 + PI, -Symbolic({x}) - PI}},
      std::tuple{
          OpType::U,
          std::vector<SymbolOrNumber>{Symbolic({x}), 2.0, Symbolic({y})},
          std::vector<SymbolOrNumber>{-Symbolic({x}), -Symbolic({y}), -2.0}},
      std::tuple{OpType::XXminusYY, std::vector<SymbolOrNumber>{Symbolic({x})},
                 std::vector<SymbolOrNumber>{-Symbolic({x})}},
      std::tuple{OpType::XXplusYY, std::vector<SymbolOrNumber>{1.0},
                 std::vector<SymbolOrNumber>{-1.0}},
  };

  for (const auto& testcase : cases) {
    auto op =
        SymbolicOperation(1U, 0U, std::get<0>(testcase), std::get<1>(testcase));
    op.invert();

    for (size_t i = 0; i < std::get<1>(testcase).size(); ++i) {
      ASSERT_EQ(op.getParameter(i), std::get<2>(testcase)[i]);
    }
  }

  // The following gate should be handled by the StandardOperation function
  auto op = SymbolicOperation(2U, Targets{0U, 1U}, OpType::DCX);
  op.invert();
  const auto expectedTargets = Targets{1U, 0U};
  ASSERT_EQ(op.getTargets(), expectedTargets);
}

TEST_F(QFRFunctionality, measureAll) {
  qc::QuantumComputation qc(2U);
  qc.measureAll();
  std::cout << qc << "\n";
  EXPECT_EQ(qc.getNops(), 3U);
  EXPECT_EQ(qc.getNcbits(), 2U);
  EXPECT_EQ(qc.getCregs().size(), 1U);
  EXPECT_EQ(qc.getClassicalRegister(0U), "meas");
  EXPECT_EQ(qc.getClassicalRegister(1U), "meas");
}

TEST_F(QFRFunctionality, measureAllExistingRegister) {
  qc::QuantumComputation qc(2U, 2U);
  qc.measureAll(false);
  std::cout << qc << "\n";
  EXPECT_EQ(qc.getNops(), 3U);
  EXPECT_EQ(qc.getNcbits(), 2U);
  EXPECT_EQ(qc.getCregs().size(), 1U);
  EXPECT_EQ(qc.getClassicalRegister(0U), "c");
  EXPECT_EQ(qc.getClassicalRegister(1U), "c");
}

TEST_F(QFRFunctionality, measureAllInsufficientRegisterSize) {
  qc::QuantumComputation qc(2U, 1U);
  EXPECT_THROW(qc.measureAll(false), QFRException);
}

TEST_F(QFRFunctionality, checkClassicalRegisters) {
  qc::QuantumComputation qc(1U, 1U);
  EXPECT_THROW(qc.classicControlled(qc::X, 0U, {0U, 2U}), QFRException);
}

TEST_F(QFRFunctionality, MeasurementSanityCheck) {
  qc::QuantumComputation qc(1U);
  qc.addClassicalRegister(1U, "c");

  EXPECT_THROW(qc.measure(0, {"c", 1U}), QFRException);
  EXPECT_THROW(qc.measure(0, {"d", 0U}), QFRException);
}
