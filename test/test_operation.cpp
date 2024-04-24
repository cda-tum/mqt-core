//
// This file is part of the MQT QMAP library released under the MIT license.
// See README.md or go to https://github.com/cda-tum/mqt-core for more
// information.
//

#include "operations/ClassicControlledOperation.hpp"
#include "operations/CompoundOperation.hpp"
#include "operations/NonUnitaryOperation.hpp"
#include "operations/Operation.hpp"
#include "operations/StandardOperation.hpp"
#include "operations/SymbolicOperation.hpp"

#include <gtest/gtest.h>

TEST(StandardOperation, CommutesAtQubit) {
  const qc::StandardOperation op1(0, 1, qc::OpType::RY,
                                  std::vector<qc::fp>{qc::PI_2});
  const qc::StandardOperation op2(0, 1, qc::OpType::RY,
                                  std::vector<qc::fp>{-qc::PI_4});
  const qc::StandardOperation op3(0, qc::OpType::RY,
                                  std::vector<qc::fp>{-qc::PI_4});
  EXPECT_TRUE(op1.commutesAtQubit(op2, 0));
  EXPECT_TRUE(op1.commutesAtQubit(op2, 0));
  EXPECT_FALSE(op1.commutesAtQubit(op3, 0));
  EXPECT_ANY_THROW(std::ignore = op1.commutesAtQubit(op2, 2));
}

TEST(CompoundOperation, CommutesAtQubit) {
  qc::CompoundOperation op1;
  op1.emplace_back<qc::StandardOperation>(0, qc::OpType::RY,
                                          std::vector<qc::fp>{qc::PI_2});
  op1.emplace_back<qc::StandardOperation>(1, qc::OpType::RX,
                                          std::vector<qc::fp>{qc::PI_2});
  qc::CompoundOperation op2;
  op2.emplace_back<qc::StandardOperation>(0, qc::OpType::RY,
                                          std::vector<qc::fp>{qc::PI_2});
  op2.emplace_back<qc::StandardOperation>(1, qc::OpType::RY,
                                          std::vector<qc::fp>{qc::PI_2});
  op2.emplace_back<qc::StandardOperation>(1, qc::OpType::RY,
                                          std::vector<qc::fp>{qc::PI_2});
  const qc::StandardOperation op3(0, qc::OpType::RY,
                                  std::vector<qc::fp>{-qc::PI_4});
  const qc::StandardOperation op4(1, qc::OpType::RY,
                                  std::vector<qc::fp>{-qc::PI_4});
  EXPECT_TRUE(op1.commutesAtQubit(op3, 0));
  EXPECT_TRUE(op3.commutesAtQubit(op1, 0));
  EXPECT_FALSE(op4.commutesAtQubit(op1, 1));
  EXPECT_FALSE(op4.commutesAtQubit(op2, 1));
  EXPECT_TRUE(op1.commutesAtQubit(op2, 0));
  EXPECT_FALSE(op1.commutesAtQubit(op2, 1));
  EXPECT_TRUE(op1.commutesAtQubit(op2, 2));
}

TEST(StandardOperation, IsInverseOf) {
  const qc::StandardOperation op1(0, qc::OpType::RY,
                                  std::vector<qc::fp>{qc::PI_2});
  qc::StandardOperation op1Inv = op1;
  op1Inv.invert();
  EXPECT_TRUE(op1.isInverseOf(op1Inv));
  EXPECT_FALSE(op1.isInverseOf(op1));
  const qc::StandardOperation op2(0, qc::OpType::Sdg);
  qc::StandardOperation op2Inv = op2;
  op2Inv.invert();
  EXPECT_TRUE(op2.isInverseOf(op2Inv));
  EXPECT_FALSE(op2.isInverseOf(op2));
  const qc::StandardOperation op3(0, qc::OpType::X);
  EXPECT_FALSE(op3.isInverseOf(op1));
}

TEST(CompoundOperation, GlobalIsInverseOf) {
  qc::CompoundOperation op1;
  op1.emplace_back<qc::StandardOperation>(0, qc::RY, std::vector{qc::PI_2});
  op1.emplace_back<qc::StandardOperation>(1, qc::RY, std::vector{qc::PI_2});
  // the actual inverse of op1
  qc::CompoundOperation op2;
  op2.emplace_back<qc::StandardOperation>(0, qc::RY, std::vector{-qc::PI_2});
  op2.emplace_back<qc::StandardOperation>(1, qc::RY, std::vector{-qc::PI_2});
  // the compound operations with different number of operations
  qc::CompoundOperation op3 = op2;
  op3.emplace_back<qc::StandardOperation>(2, qc::RY, std::vector{qc::PI_2});
  // the operations come in wrong order (this is a false negative)
  qc::CompoundOperation op4;
  op4.emplace_back<qc::StandardOperation>(1, qc::RY, std::vector{-qc::PI_2});
  op4.emplace_back<qc::StandardOperation>(0, qc::RY, std::vector{-qc::PI_2});
  EXPECT_TRUE(op1.isInverseOf(op2));
  EXPECT_TRUE(op2.isInverseOf(op1));
  EXPECT_FALSE(op1.isInverseOf(op3));
  EXPECT_FALSE(op1.isInverseOf(qc::StandardOperation(0, qc::RY)));
  EXPECT_FALSE(op1.isInverseOf(op4));
}

TEST(CompoundOperation, IsInverseOf) {
  // This functionality is not implemented yet, the function isInversOf leads to
  // false negatives
  GTEST_SKIP();
  qc::CompoundOperation op1;
  op1.emplace_back<qc::StandardOperation>(0, qc::OpType::RY,
                                          std::vector<qc::fp>{-qc::PI_2});
  op1.emplace_back<qc::StandardOperation>(1, qc::OpType::RY,
                                          std::vector<qc::fp>{-qc::PI_2});
  qc::CompoundOperation op2 = op1;
  op2.invert();
  EXPECT_TRUE(op1.isInverseOf(op2));
}

TEST(OpType, General) {
  EXPECT_EQ(qc::toString(qc::RZ), "rz");
  std::stringstream ss;
  ss << qc::OpType::RZ;
  EXPECT_EQ(ss.str(), "rz");
}

TEST(OpType, SingleQubitGate) {
  EXPECT_TRUE(qc::isSingleQubitGate(qc::P));
  EXPECT_FALSE(qc::isSingleQubitGate(qc::ECR));
}

TEST(NonStandardOperation, IsInverseOf) {
  const qc::StandardOperation op(0, qc::I);
  EXPECT_FALSE(qc::NonUnitaryOperation(qc::Targets{0}).isInverseOf(op));
  EXPECT_FALSE(qc::SymbolicOperation().isInverseOf(op));
}

TEST(NonStandardOperation, CommutesAtQubit) {
  const qc::StandardOperation op(0, qc::X);
  EXPECT_FALSE(qc::NonUnitaryOperation(qc::Targets{0}).commutesAtQubit(op, 0));
  EXPECT_FALSE(
      qc::SymbolicOperation(0, qc::P, {sym::Expression<qc::fp, qc::fp>()})
          .commutesAtQubit(op, 0));
}

TEST(Operation, IsIndividualGate) {
  const qc::StandardOperation op1(0, qc::X);
  EXPECT_TRUE(op1.isIndividual());
  const qc::StandardOperation op2(0, 1, qc::X);
  EXPECT_FALSE(op2.isIndividual());
  const qc::StandardOperation op3(1, qc::RXX);
  EXPECT_FALSE(op3.isIndividual());
}

TEST(Operation, IsDiagonalGate) {
  const qc::StandardOperation op1(0, qc::X);
  EXPECT_FALSE(op1.isDiagonal());
  const qc::StandardOperation op2(0, qc::Z);
  EXPECT_TRUE(op2.isDiagonal());
}
