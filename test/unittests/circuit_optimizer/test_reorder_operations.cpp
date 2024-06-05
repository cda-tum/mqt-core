#include "CircuitOptimizer.hpp"
#include "QuantumComputation.hpp"

#include <gtest/gtest.h>
#include <iostream>

namespace qc {
TEST(ReorderOperations, trivialOperationReordering) {
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

TEST(ReorderOperations, OperationReorderingBarrier) {
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

} // namespace qc
