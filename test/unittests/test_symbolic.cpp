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

  QuantumComputation symQc = QuantumComputation(3);
  QuantumComputation qc = QuantumComputation(3);
};

TEST_F(SymbolicTest, Gates) {
  auto xVal = PI_4 / 2;
  auto yVal = PI_4 / 4;
  auto zVal = PI / 3;

  auto noRealSymQc = QuantumComputation(3);

  // test all kinds of symbolic operations supported
  symQc.u3(0, {1_pc, 2_nc}, xMonom, yMonom, zMonom);

  symQc.u3(0, {1_pc}, xMonom, yMonom, zMonom);
  symQc.u3(0, xMonom, yMonom, zMonom);

  symQc.u2(0, {1_pc, 2_nc}, xMonom, yMonom);
  symQc.u2(0, {1_pc}, xMonom, yMonom);
  symQc.u2(0, xMonom, yMonom);

  symQc.phase(0, {1_pc, 2_nc}, xMonom);
  symQc.phase(0, {1_pc}, xMonom);
  symQc.phase(0, xMonom);

  symQc.rx(0, {1_pc, 2_nc}, xMonom);
  symQc.rx(0, {1_pc}, xMonom);
  symQc.rx(0, xMonom);

  symQc.ry(0, {1_pc, 2_nc}, xMonom);
  symQc.ry(0, {1_pc}, xMonom);
  symQc.ry(0, xMonom);

  symQc.rz(0, {1_pc, 2_nc}, xMonom);
  symQc.rz(0, {1_pc}, xMonom);
  symQc.rz(0, xMonom);

  EXPECT_FALSE(symQc.isVariableFree());
  for (const auto& symOp : symQc) {
    EXPECT_TRUE(symOp->isSymbolicOperation());
  }

  // normal circuit
  qc.u3(0, {1_pc, 2_nc}, xVal, yVal, zVal);
  qc.u3(0, {1_pc}, xVal, yVal, zVal);
  qc.u3(0, xVal, yVal, zVal);

  qc.u2(0, {1_pc, 2_nc}, xVal, yVal);
  qc.u2(0, {1_pc}, xVal, yVal);
  qc.u2(0, xVal, yVal);

  qc.phase(0, {1_pc, 2_nc}, xVal);
  qc.phase(0, {1_pc}, xVal);
  qc.phase(0, xVal);

  qc.rx(0, {1_pc, 2_nc}, xVal);
  qc.rx(0, {1_pc}, xVal);
  qc.rx(0, xVal);

  qc.ry(0, {1_pc, 2_nc}, xVal);
  qc.ry(0, {1_pc}, xVal);
  qc.ry(0, xVal);

  qc.rz(0, {1_pc, 2_nc}, xVal);
  qc.rz(0, {1_pc}, xVal);
  qc.rz(0, xVal);

  // symbolic but variable free circuit
  noRealSymQc.u3(0, {1_pc, 2_nc}, xVal, yVal, zVal);
  noRealSymQc.u3(0, {1_pc}, xVal, yVal, zVal);
  noRealSymQc.u3(0, xVal, yVal, zVal);

  noRealSymQc.u2(0, {1_pc, 2_nc}, xVal, yVal);
  noRealSymQc.u2(0, {1_pc}, xVal, yVal);
  noRealSymQc.u2(0, xVal, yVal);

  noRealSymQc.phase(0, {1_pc, 2_nc}, xVal);
  noRealSymQc.phase(0, {1_pc}, xVal);
  noRealSymQc.phase(0, xVal);

  noRealSymQc.rx(0, {1_pc, 2_nc}, xVal);
  noRealSymQc.rx(0, {1_pc}, xVal);
  noRealSymQc.rx(0, xVal);

  noRealSymQc.ry(0, {1_pc, 2_nc}, xVal);
  noRealSymQc.ry(0, {1_pc}, xVal);
  noRealSymQc.ry(0, xVal);

  noRealSymQc.rz(0, {1_pc, 2_nc}, xVal);
  noRealSymQc.rz(0, {1_pc}, xVal);
  noRealSymQc.rz(0, xVal);

  EXPECT_TRUE(noRealSymQc.isVariableFree());

  // no operation in the uninstantiated circuit should be equal to the standard
  // circuit
  for (auto it1 = symQc.begin(), it2 = noRealSymQc.begin();
       it1 != symQc.end() && it2 != noRealSymQc.end(); ++it1, ++it2) {
    EXPECT_FALSE((*it1)->equals(*(*it2)));
  }

  const VariableAssignment assignment{{x, xVal}, {y, yVal}, {z, zVal}};
  symQc.instantiate(assignment);

  // after the instantiation, the symbolic circuit should be equal to the
  // standard circuit
  for (auto it1 = symQc.begin(), it2 = noRealSymQc.begin();
       it1 != symQc.end() && it2 != noRealSymQc.end(); ++it1, ++it2) {
    EXPECT_TRUE((*it1)->equals(*(*it2)));
  }
}

TEST_F(SymbolicTest, TestClone) {
  symQc.u3(0, {1_pc, 2_nc}, xMonom, yMonom, zMonom);
  const auto& clonedQc = symQc.clone();

  symQc.u3(0, xMonom, yMonom, zMonom);
  EXPECT_NE(symQc.getNops(), clonedQc.getNops());
}

TEST_F(SymbolicTest, TestU3SymLambdaPhase) {
  symQc.u3(0, 0.0, 0.0, xMonom);
  EXPECT_EQ((*symQc.begin())->getType(), OpType::Phase);
}

TEST_F(SymbolicTest, TestU3SymLambdaU2) {
  symQc.u3(0, PI_2, 1.234, xMonom);
  EXPECT_EQ((*symQc.begin())->getType(), OpType::U2);
}

TEST_F(SymbolicTest, TestU3SymLambdaU3) {
  symQc.u3(0, 4.567, 1.234, xMonom);
  EXPECT_EQ((*symQc.begin())->getType(), OpType::U3);
}

TEST_F(SymbolicTest, TestU3SymPhiU2) {
  symQc.u3(0, PI_2, xMonom, 1.234);
  EXPECT_EQ((*symQc.begin())->getType(), OpType::U2);
}

TEST_F(SymbolicTest, TestU3SymPhiU3) {
  symQc.u3(0, 4.567, xMonom, 0.0);
  EXPECT_EQ((*symQc.begin())->getType(), OpType::U3);

  symQc.u3(0, 1.234, xMonom, PI_2);
  EXPECT_EQ((*(symQc.begin() + 1))->getType(), OpType::U3);

  symQc.u3(0, 3.465, xMonom, PI);
  EXPECT_EQ((*(symQc.begin() + 2))->getType(), OpType::U3);

  symQc.u3(0, 1.2345, xMonom, 3.465);
  EXPECT_EQ((*(symQc.begin() + 3))->getType(), OpType::U3);
}

TEST_F(SymbolicTest, TestU3SymThetaU3) {
  symQc.u3(0, xMonom, 0.0, 0.0);
  EXPECT_EQ((*symQc.begin())->getType(), OpType::U3);

  symQc.u3(0, xMonom, PI_2, PI_2);
  EXPECT_EQ((*(symQc.begin() + 1))->getType(), OpType::U3);

  symQc.u3(0, xMonom, 0.0, PI);
  EXPECT_EQ((*(symQc.begin() + 2))->getType(), OpType::U3);

  symQc.u3(0, xMonom, 4.567, 1.234);
  EXPECT_EQ((*(symQc.begin() + 3))->getType(), OpType::U3);
}

TEST_F(SymbolicTest, TestU3SymLambdaSymPhiU2) {
  symQc.u3(0, PI_2, xMonom, yMonom);
  EXPECT_EQ((*symQc.begin())->getType(), OpType::U2);
}

TEST_F(SymbolicTest, TestU3SymLambdaSymPhiU3) {
  symQc.u3(0, PI_2 - 0.2, xMonom, yMonom);
  EXPECT_EQ((*symQc.begin())->getType(), OpType::U3);
}

TEST_F(SymbolicTest, TestU3SymLambdaSymThetaU3) {
  symQc.u3(0, xMonom, 0.0, yMonom);
  EXPECT_EQ((*symQc.begin())->getType(), OpType::U3);
}

TEST_F(SymbolicTest, TestU3SymPhiSymThetaU3) {
  symQc.u3(0, xMonom, yMonom, 1.2345);
  EXPECT_EQ((*symQc.begin())->getType(), OpType::U3);
}

TEST_F(SymbolicTest, TestU2SymLambda) {
  symQc.u2(0, 0.0, xMonom);
  EXPECT_EQ((*symQc.begin())->getType(), OpType::U2);
}

TEST_F(SymbolicTest, TestU2SymPhi) {
  symQc.u2(0, xMonom, 1.2345);
  EXPECT_EQ((*symQc.begin())->getType(), OpType::U2);
}
