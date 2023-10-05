#include "QuantumComputation.hpp"
#include "operations/Expression.hpp"
#include "operations/SymbolicOperation.hpp"

#include "gtest/gtest.h"
#include <memory>

using namespace qc;
using namespace sym;
class SymbolicTest : public ::testing::Test {
public:
  Variable x = Variable("x");
  Variable y = Variable("y");
  Variable z = Variable("z");

  Symbolic xMonom = Symbolic{Term<fp>{x}};
  Symbolic yMonom = Symbolic{Term<fp>{y}};
  Symbolic zMonom = Symbolic{Term<fp>{z}};

  QuantumComputation symQc = QuantumComputation(4);
  QuantumComputation qc = QuantumComputation(4);
};

TEST_F(SymbolicTest, Gates) {
  auto xVal = PI_4 / 2;
  auto yVal = PI_4 / 4;
  auto zVal = PI / 3;

  // test all kinds of symbolic operations supported
  symQc.u(xMonom, yMonom, zMonom, {1_pc, 2_nc}, 0);
  symQc.u(xMonom, yMonom, zMonom, 1_pc, 0);
  symQc.u(xMonom, yMonom, zMonom, 0);

  symQc.u2(xMonom, yMonom, {1_pc, 2_nc}, 0);
  symQc.u2(xMonom, yMonom, 1_pc, 0);
  symQc.u2(xMonom, yMonom, 0);

  symQc.phase(xMonom, {1_pc, 2_nc}, 0);
  symQc.phase(xMonom, 1_pc, 0);
  symQc.phase(xMonom, 0);

  symQc.rx(xMonom, {1_pc, 2_nc}, 0);
  symQc.rx(xMonom, 1_pc, 0);
  symQc.rx(xMonom, 0);

  symQc.ry(xMonom, {1_pc, 2_nc}, 0);
  symQc.ry(xMonom, 1_pc, 0);
  symQc.ry(xMonom, 0);

  symQc.rz(xMonom, {1_pc, 2_nc}, 0);
  symQc.rz(xMonom, 1_pc, 0);
  symQc.rz(xMonom, 0);

  symQc.rxx(xMonom, {3_pc, 2_nc}, 0, 1);
  symQc.rxx(xMonom, 2_pc, 0, 1);
  symQc.rxx(xMonom, 0, 1);

  symQc.ryy(xMonom, {3_pc, 2_nc}, 0, 1);
  symQc.ryy(xMonom, 2_pc, 0, 1);
  symQc.ryy(xMonom, 0, 1);

  symQc.rzz(xMonom, {3_pc, 2_nc}, 0, 1);
  symQc.rzz(xMonom, 2_pc, 0, 1);
  symQc.rzz(xMonom, 0, 1);

  symQc.rzx(xMonom, {3_pc, 2_nc}, 0, 1);
  symQc.rzx(xMonom, 2_pc, 0, 1);
  symQc.rzx(xMonom, 0, 1);

  symQc.xx_minus_yy(xMonom, yMonom, {3_pc, 2_nc}, 0, 1);
  symQc.xx_minus_yy(xMonom, yMonom, 2_pc, 0, 1);
  symQc.xx_minus_yy(xMonom, yMonom, 0, 1);

  symQc.xx_plus_yy(xMonom, yMonom, {3_pc, 2_nc}, 0, 1);
  symQc.xx_plus_yy(xMonom, yMonom, 2_pc, 0, 1);
  symQc.xx_plus_yy(xMonom, yMonom, 0, 1);

  EXPECT_FALSE(symQc.isVariableFree());
  for (const auto& symOp : symQc) {
    EXPECT_TRUE(symOp->isSymbolicOperation());
  }

  // normal circuit
  qc.u(xVal, yVal, zVal, {1_pc, 2_nc}, 0);
  qc.u(xVal, yVal, zVal, 1_pc, 0);
  qc.u(xVal, yVal, zVal, 0);

  qc.u2(xVal, yVal, {1_pc, 2_nc}, 0);
  qc.u2(xVal, yVal, 1_pc, 0);
  qc.u2(xVal, yVal, 0);

  qc.phase(xVal, {1_pc, 2_nc}, 0);
  qc.phase(xVal, 1_pc, 0);
  qc.phase(xVal, 0);

  qc.rx(xVal, {1_pc, 2_nc}, 0);
  qc.rx(xVal, 1_pc, 0);
  qc.rx(xVal, 0);

  qc.ry(xVal, {1_pc, 2_nc}, 0);
  qc.ry(xVal, 1_pc, 0);
  qc.ry(xVal, 0);

  qc.rz(xVal, {1_pc, 2_nc}, 0);
  qc.rz(xVal, 1_pc, 0);
  qc.rz(xVal, 0);

  qc.rxx(xVal, {3_pc, 2_nc}, 0, 1);
  qc.rxx(xVal, 2_pc, 0, 1);
  qc.rxx(xVal, 0, 1);

  qc.ryy(xVal, {3_pc, 2_nc}, 0, 1);
  qc.ryy(xVal, 2_pc, 0, 1);
  qc.ryy(xVal, 0, 1);

  qc.rzz(xVal, {3_pc, 2_nc}, 0, 1);
  qc.rzz(xVal, 2_pc, 0, 1);
  qc.rzz(xVal, 0, 1);

  qc.rzx(xVal, {3_pc, 2_nc}, 0, 1);
  qc.rzx(xVal, 2_pc, 0, 1);
  qc.rzx(xVal, 0, 1);

  qc.xx_minus_yy(xVal, yVal, {3_pc, 2_nc}, 0, 1);
  qc.xx_minus_yy(xVal, yVal, 2_pc, 0, 1);
  qc.xx_minus_yy(xVal, yVal, 0, 1);

  qc.xx_plus_yy(xVal, yVal, {3_pc, 2_nc}, 0, 1);
  qc.xx_plus_yy(xVal, yVal, 2_pc, 0, 1);
  qc.xx_plus_yy(xVal, yVal, 0, 1);

  EXPECT_TRUE(qc.isVariableFree());

  // no operation in the uninstantiated circuit should be equal to the standard
  // circuit
  for (auto it1 = symQc.begin(), it2 = qc.begin();
       it1 != symQc.end() && it2 != qc.end(); ++it1, ++it2) {
    EXPECT_FALSE((*it1)->equals(*(*it2)));
  }

  const VariableAssignment assignment{{x, xVal}, {y, yVal}, {z, zVal}};
  symQc.instantiate(assignment);

  // after the instantiation, the symbolic circuit should be equal to the
  // standard circuit
  for (auto it1 = symQc.begin(), it2 = qc.begin();
       it1 != symQc.end() && it2 != qc.end(); ++it1, ++it2) {
    EXPECT_TRUE((*it1)->equals(*(*it2)));
  }
}

TEST_F(SymbolicTest, TestClone) {
  symQc.u(xMonom, yMonom, zMonom, {1_pc, 2_nc}, 0);
  const auto clonedQc = symQc;

  symQc.u(xMonom, yMonom, zMonom, 0);
  EXPECT_NE(symQc.getNops(), clonedQc.getNops());
}

TEST_F(SymbolicTest, TestU3SymLambdaPhase) {
  symQc.u(0.0, 0.0, xMonom, 0);
  EXPECT_EQ((*symQc.begin())->getType(), OpType::Phase);
}

TEST_F(SymbolicTest, TestU3SymLambdaU2) {
  symQc.u(PI_2, 1.234, xMonom, 0);
  EXPECT_EQ((*symQc.begin())->getType(), OpType::U2);
}

TEST_F(SymbolicTest, TestU3SymLambdaU3) {
  symQc.u(4.567, 1.234, xMonom, 0);
  EXPECT_EQ((*symQc.begin())->getType(), OpType::U3);
}

TEST_F(SymbolicTest, TestU3SymPhiU2) {
  symQc.u(PI_2, xMonom, 1.234, 0);
  EXPECT_EQ((*symQc.begin())->getType(), OpType::U2);
}

TEST_F(SymbolicTest, TestU3SymPhiU3) {
  symQc.u(4.567, xMonom, 0.0, 0);
  EXPECT_EQ((*symQc.begin())->getType(), OpType::U3);

  symQc.u(1.234, xMonom, PI_2, 0);
  EXPECT_EQ((*(symQc.begin() + 1))->getType(), OpType::U3);

  symQc.u(3.465, xMonom, PI, 0);
  EXPECT_EQ((*(symQc.begin() + 2))->getType(), OpType::U3);

  symQc.u(1.2345, xMonom, 3.465, 0);
  EXPECT_EQ((*(symQc.begin() + 3))->getType(), OpType::U3);
}

TEST_F(SymbolicTest, TestU3SymThetaU3) {
  symQc.u(xMonom, 0.0, 0.0, 0);
  EXPECT_EQ((*symQc.begin())->getType(), OpType::U3);

  symQc.u(xMonom, PI_2, PI_2, 0);
  EXPECT_EQ((*(symQc.begin() + 1))->getType(), OpType::U3);

  symQc.u(xMonom, 0.0, PI, 0);
  EXPECT_EQ((*(symQc.begin() + 2))->getType(), OpType::U3);

  symQc.u(xMonom, 4.567, 1.234, 0);
  EXPECT_EQ((*(symQc.begin() + 3))->getType(), OpType::U3);
}

TEST_F(SymbolicTest, TestU3SymLambdaSymPhiU2) {
  symQc.u(PI_2, xMonom, yMonom, 0);
  EXPECT_EQ((*symQc.begin())->getType(), OpType::U2);
}

TEST_F(SymbolicTest, TestU3SymLambdaSymPhiU3) {
  symQc.u(PI_2 - 0.2, xMonom, yMonom, 0);
  EXPECT_EQ((*symQc.begin())->getType(), OpType::U3);
}

TEST_F(SymbolicTest, TestU3SymLambdaSymThetaU3) {
  symQc.u(xMonom, 0.0, yMonom, 0);
  EXPECT_EQ((*symQc.begin())->getType(), OpType::U3);
}

TEST_F(SymbolicTest, TestU3SymPhiSymThetaU3) {
  symQc.u(xMonom, yMonom, 1.2345, 0);
  EXPECT_EQ((*symQc.begin())->getType(), OpType::U3);
}

TEST_F(SymbolicTest, TestU2SymLambda) {
  symQc.u2(0.0, xMonom, 0);
  EXPECT_EQ((*symQc.begin())->getType(), OpType::U2);
}

TEST_F(SymbolicTest, TestU2SymPhi) {
  symQc.u2(xMonom, 1.2345, 0);
  EXPECT_EQ((*symQc.begin())->getType(), OpType::U2);
}
