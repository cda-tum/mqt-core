#include "CircuitOptimizer.hpp"
#include "Definitions.hpp"
#include "QuantumComputation.hpp"
#include "operations/NonUnitaryOperation.hpp"
#include "operations/OpType.hpp"

#include <gtest/gtest.h>
#include <iostream>

namespace qc {
TEST(DeferMeasurements, basicTest) {
  // Input:
  // i:   0   1
  // 1:   h   |
  // 2:   0   |
  // 3:   |   x  c[0] == 1
  // o:   0   1

  // Expected Output:
  // i:   0   1
  // 1:   h   |
  // 2:   c   x
  // 3:   0   |
  // o:   0   |

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

  EXPECT_TRUE(op0->getType() == qc::H);
  const auto& targets0 = op0->getTargets();
  EXPECT_EQ(targets0.size(), 1);
  EXPECT_EQ(targets0.at(0), static_cast<Qubit>(0));
  EXPECT_TRUE(op0->getControls().empty());

  EXPECT_TRUE(op1->getType() == qc::X);
  const auto& targets1 = op1->getTargets();
  EXPECT_EQ(targets1.size(), 1);
  EXPECT_EQ(targets1.at(0), static_cast<Qubit>(1));
  const auto& controls1 = op1->getControls();
  EXPECT_EQ(controls1.size(), 1);
  EXPECT_EQ(controls1.count(0), 1);

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

TEST(DeferMeasurements, measurementBetweenMeasurementAndClassic) {
  // Input:
  // i:   0   1
  // 1:   h   |
  // 2:   0   |
  // 3:   h   |
  // 4:   |   x  c[0] == 1
  // o:   0   1

  // Expected Output:
  // i:   0   1
  // 1:   h   |
  // 2:   c   x
  // 3:   h   |
  // 4:   0   |
  // o:   0   |

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

  EXPECT_TRUE(op0->getType() == qc::H);
  const auto& targets0 = op0->getTargets();
  EXPECT_EQ(targets0.size(), 1);
  EXPECT_EQ(targets0.at(0), static_cast<Qubit>(0));
  EXPECT_TRUE(op0->getControls().empty());

  EXPECT_TRUE(op1->getType() == qc::X);
  const auto& targets1 = op1->getTargets();
  EXPECT_EQ(targets1.size(), 1);
  EXPECT_EQ(targets1.at(0), static_cast<Qubit>(1));
  const auto& controls1 = op1->getControls();
  EXPECT_EQ(controls1.size(), 1);
  EXPECT_EQ(controls1.count(0), 1);

  EXPECT_TRUE(op2->getType() == qc::H);
  const auto& targets2 = op2->getTargets();
  EXPECT_EQ(targets2.size(), 1);
  EXPECT_EQ(targets2.at(0), static_cast<Qubit>(0));
  EXPECT_TRUE(op2->getControls().empty());

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

TEST(DeferMeasurements, twoClassic) {
  // Input:
  // i:   0   1
  // 1:   h   |
  // 2:   0   |
  // 3:   h   |
  // 4:   |   x  c[0] == 1
  // 5:   |   z  c[0] == 1
  // o:   0   1

  // Expected Output:
  // i:   0   1
  // 1:   h   |
  // 2:   c   x
  // 3:   c   z
  // 4:   h   |
  // 5:   0   |
  // o:   0   |

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

  EXPECT_TRUE(op0->getType() == qc::H);
  const auto& targets0 = op0->getTargets();
  EXPECT_EQ(targets0.size(), 1);
  EXPECT_EQ(targets0.at(0), static_cast<Qubit>(0));
  EXPECT_TRUE(op0->getControls().empty());

  EXPECT_TRUE(op1->getType() == qc::X);
  const auto& targets1 = op1->getTargets();
  EXPECT_EQ(targets1.size(), 1);
  EXPECT_EQ(targets1.at(0), static_cast<Qubit>(1));
  const auto& controls1 = op1->getControls();
  EXPECT_EQ(controls1.size(), 1);
  EXPECT_EQ(controls1.count(0), 1);

  EXPECT_TRUE(op2->getType() == qc::Z);
  const auto& targets2 = op2->getTargets();
  EXPECT_EQ(targets2.size(), 1);
  EXPECT_EQ(targets2.at(0), static_cast<Qubit>(1));
  const auto& controls2 = op2->getControls();
  EXPECT_EQ(controls2.size(), 1);
  EXPECT_EQ(controls2.count(0), 1);

  EXPECT_TRUE(op3->getType() == qc::H);
  const auto& targets3 = op3->getTargets();
  EXPECT_EQ(targets3.size(), 1);
  EXPECT_EQ(targets3.at(0), static_cast<Qubit>(0));
  EXPECT_TRUE(op3->getControls().empty());

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

TEST(DeferMeasurements, correctOrder) {
  // Input:
  // i:   0   1
  // 1:   h   |
  // 2:   0   |
  // 3:   |   h
  // 4:   |   x  c[0] == 1
  // o:   0   1

  // Expected Output:
  // i:   0   1
  // 1:   h   |
  // 2:   |   h
  // 3:   c   x
  // 4:   0   |
  // o:   0   |

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

  EXPECT_TRUE(op0->getType() == qc::H);
  const auto& targets0 = op0->getTargets();
  EXPECT_EQ(targets0.size(), 1);
  EXPECT_EQ(targets0.at(0), static_cast<Qubit>(0));
  EXPECT_TRUE(op0->getControls().empty());

  EXPECT_TRUE(op1->getType() == qc::H);
  const auto& targets1 = op2->getTargets();
  EXPECT_EQ(targets1.size(), 1);
  EXPECT_EQ(targets1.at(0), static_cast<Qubit>(1));
  EXPECT_TRUE(op1->getControls().empty());

  EXPECT_TRUE(op2->getType() == qc::X);
  const auto& targets2 = op1->getTargets();
  EXPECT_EQ(targets2.size(), 1);
  EXPECT_EQ(targets2.at(0), static_cast<Qubit>(1));
  const auto& controls2 = op2->getControls();
  EXPECT_EQ(controls2.size(), 1);
  EXPECT_EQ(controls2.count(0), 1);

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

TEST(DeferMeasurements, twoClassicCorrectOrder) {
  // Input:
  // i:   0   1
  // 1:   h   |
  // 2:   0   |
  // 3:   |   h
  // 4:   |   x  c[0] == 1
  // 5:   |   z  c[0] == 1
  // o:   0   1

  // Expected Output:
  // i:   0   1
  // 1:   h   |
  // 2:   |   h
  // 3:   c   x
  // 4:   c   z
  // 5:   0   |
  // o:   0   |

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

  EXPECT_TRUE(op0->getType() == qc::H);
  const auto& targets0 = op0->getTargets();
  EXPECT_EQ(targets0.size(), 1);
  EXPECT_EQ(targets0.at(0), static_cast<Qubit>(0));
  EXPECT_TRUE(op0->getControls().empty());

  EXPECT_TRUE(op1->getType() == qc::H);
  const auto& targets1 = op1->getTargets();
  EXPECT_EQ(targets1.size(), 1);
  EXPECT_EQ(targets1.at(0), static_cast<Qubit>(1));
  EXPECT_TRUE(op1->getControls().empty());

  EXPECT_TRUE(op2->getType() == qc::X);
  const auto& targets2 = op2->getTargets();
  EXPECT_EQ(targets2.size(), 1);
  EXPECT_EQ(targets2.at(0), static_cast<Qubit>(1));
  const auto& controls2 = op2->getControls();
  EXPECT_EQ(controls2.size(), 1);
  EXPECT_EQ(controls2.count(0), 1);

  EXPECT_TRUE(op3->getType() == qc::Z);
  const auto& targets3 = op3->getTargets();
  EXPECT_EQ(targets3.size(), 1);
  EXPECT_EQ(targets3.at(0), static_cast<Qubit>(1));
  const auto& controls3 = op3->getControls();
  EXPECT_EQ(controls3.size(), 1);
  EXPECT_EQ(controls3.count(0), 1);

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

TEST(DeferMeasurements, errorOnImplicitReset) {
  // Input:
  // i:   0
  // 1:   h
  // 2:   0
  // 3:   x  c[0] == 1
  // o:   0

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

TEST(DeferMeasurements, preserveOutputPermutationWithoutMeasurements) {
  QuantumComputation qc(2);
  qc.h(0);
  qc.cx(0, 1);
  qc.setLogicalQubitGarbage(1);
  EXPECT_EQ(qc.outputPermutation.size(), 1);
  EXPECT_EQ(qc.outputPermutation.at(0), 0);

  std::cout << qc << "\n";
  CircuitOptimizer::deferMeasurements(qc);
  std::cout << qc << "\n";

  EXPECT_EQ(qc.outputPermutation.size(), 1);
  EXPECT_EQ(qc.outputPermutation.at(0), 0);
}
} // namespace qc
