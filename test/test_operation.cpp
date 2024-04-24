//
// This file is part of the MQT QMAP library released under the MIT license.
// See README.md or go to https://github.com/cda-tum/mqt-core for more
// information.
//

#include "operations/CompoundOperation.hpp"
#include "operations/Operation.hpp"
#include "operations/StandardOperation.hpp"

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
}

TEST(CompoundOperation, GlobalIsInverseOf) {
  qc::CompoundOperation op1;
  op1.emplace_back<qc::StandardOperation>(0, qc::OpType::RY,
                                          std::vector<qc::fp>{qc::PI_2});
  op1.emplace_back<qc::StandardOperation>(1, qc::OpType::RY,
                                          std::vector<qc::fp>{qc::PI_2});
  qc::CompoundOperation op2;
  op2.emplace_back<qc::StandardOperation>(0, qc::OpType::RY,
                                          std::vector<qc::fp>{-qc::PI_2});
  op2.emplace_back<qc::StandardOperation>(1, qc::OpType::RY,
                                          std::vector<qc::fp>{-qc::PI_2});
  EXPECT_TRUE(op1.isInverseOf(op2));
  EXPECT_TRUE(op2.isInverseOf(op1));
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
