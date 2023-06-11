#include "Rational.hpp"
#include "Utils.hpp"
#include "operations/Expression.hpp"

#include <gtest/gtest.h>
#include <stdexcept>
#include <variant>

class ExpressionTest : public ::testing::Test {
public:
  sym::Term<double> x{1.0, sym::Variable("x")};
  sym::Term<double> y{sym::Variable("y")};
  sym::Term<double> z{sym::Variable("z")};
};

TEST_F(ExpressionTest, basicOps1) {
  sym::Expression<double, zx::PiRational> e(x);

  EXPECT_EQ(1, e.numTerms());
  EXPECT_EQ(zx::PiRational(0, 1), e.getConst());

  e += x;

  EXPECT_EQ(1, e.numTerms());
  EXPECT_EQ(zx::PiRational(0, 1), e.getConst());
  EXPECT_PRED_FORMAT2(testing::DoubleLE, e[0].getCoeff(), 2.0);

  e += y;
  EXPECT_EQ(2, e.numTerms());
  EXPECT_PRED_FORMAT2(testing::DoubleLE, e[0].getCoeff(), 2.0);
  EXPECT_PRED_FORMAT2(testing::DoubleLE, e[1].getCoeff(), 1.0);
  EXPECT_EQ(e[0].getVar().getName(), "x");
  EXPECT_EQ(e[1].getVar().getName(), "y");
}

TEST_F(ExpressionTest, basicOps2) {
  sym::Expression<double, zx::PiRational> e1;
  e1 += x;
  e1 += 10.0 * y;
  e1 += 5.0 * z;
  e1 += zx::PiRational(1, 2);

  sym::Expression<double, zx::PiRational> e2;
  e2 += -5.0 * x;
  e2 += -10.0 * y;
  e2 += -4.9 * z;
  e2 += zx::PiRational(3, 2);

  auto sum = e1 + e2;

  EXPECT_EQ(2, sum.numTerms());
  EXPECT_PRED_FORMAT2(testing::DoubleLE, sum[0].getCoeff(), -4.0);
  EXPECT_PRED_FORMAT2(testing::DoubleLE, sum[1].getCoeff(), 0.1);
  EXPECT_EQ(sum[0].getVar().getName(), "x");
  EXPECT_EQ(sum[1].getVar().getName(), "z");
  EXPECT_EQ(sum.getConst(), zx::PiRational(0, 1));

  sum += zx::PiRational(1, 1);
  EXPECT_EQ(sum.getConst(), zx::PiRational(1, 1));
  sum -= zx::PiRational(1, 1);
  EXPECT_EQ(sum.getConst(), zx::PiRational(0, 1));
}

TEST_F(ExpressionTest, mult) {
  sym::Expression<double, zx::PiRational> e(x);

  e = e * 2.0;

  EXPECT_PRED_FORMAT2(testing::DoubleLE, e[0].getCoeff(), 2);

  e = e * zx::PiRational(0.5);

  EXPECT_PRED_FORMAT2(testing::DoubleLE, e[0].getCoeff(), 1);

  e += sym::Expression<double, zx::PiRational>{};

  EXPECT_PRED_FORMAT2(testing::DoubleLE, e[0].getCoeff(), 1);

  e = e * 0.0;

  EXPECT_TRUE(e.isZero());
}

TEST_F(ExpressionTest, div) {
  sym::Expression<double, Rational> e(x);

  e = e / 2.0;

  EXPECT_PRED_FORMAT2(testing::DoubleLE, e[0].getCoeff(), 0.5);

  e = e / Rational(0.5);

  EXPECT_PRED_FORMAT2(testing::DoubleLE, e[0].getCoeff(), 1);

  EXPECT_THROW(e = e / 0.0, std::runtime_error);
}

TEST_F(ExpressionTest, Commutativity) {
  const sym::Expression<double, double> e1(x, y);
  sym::Expression<double, double> e2(z);
  e2.setConst(1.0);

  EXPECT_EQ(e1 + e2, e2 + e1);
  EXPECT_EQ(e1 * 2.0, 2.0 * e1);
}

TEST_F(ExpressionTest, Associativity) {
  const sym::Expression<double, double> e1(x, y);
  const sym::Expression<double, double> e2(z);
  const sym::Expression<double, double> e3(1.0);

  EXPECT_EQ(e1 + (e2 + e3), (e1 + e2) + e3);
  EXPECT_EQ(e1 * (2.0 * 4.0), (e1 * 2.0) * 4.0);
}

TEST_F(ExpressionTest, Distributive) {
  const sym::Expression<double, double> e1(x, y);
  const sym::Expression<double, double> e2(z);

  EXPECT_EQ((e1 + e2) * 2.0, (e1 * 2.0) + (e2 * 2.0));
  EXPECT_EQ((e1 - e2) * 2.0, (e1 * 2.0) - (e2 * 2.0));
  std::cout << ((e1 + e2) / 2.0) << "\n";
  std::cout << ((e1 / 2.0) + (e2 / 2.0)) << "\n";
  EXPECT_EQ((e1 + e2) / 2.0, (e1 / 2.0) + (e2 / 2.0));
  EXPECT_EQ((e1 - e2) / 2.0, (e1 / 2.0) - (e2 / 2.0));
}

TEST_F(ExpressionTest, Variable) {
  EXPECT_TRUE(sym::Variable{"x"} != sym::Variable{"y"});
  EXPECT_TRUE(sym::Variable{"x"} == sym::Variable{"x"});
  EXPECT_TRUE(sym::Variable{"x"} < sym::Variable{"y"});
  EXPECT_TRUE(sym::Variable{"z"} > sym::Variable{"y"});
}

TEST_F(ExpressionTest, SumNegation) {
  const sym::Expression<double, double> e1(x, y);
  const sym::Expression<double, double> e2(z, y);

  EXPECT_EQ(e1 - e2, e1 + (-e2));
  const auto& zero = sym::Expression<double, double>{};
  EXPECT_EQ(e1 + (-e1), zero);
}

TEST_F(ExpressionTest, SumMult) {
  const sym::Expression<double, double> e1(x, y);
  EXPECT_EQ(e1 + e1, e1 * 2.0);
}

TEST_F(ExpressionTest, CliffordRounding) {
  const double eps = 1e-14;
  zx::PiExpression e{zx::PiRational(zx::PI - eps)};
  zx::roundToClifford(e, 1e-9);
  EXPECT_EQ(e, zx::PiExpression(zx::PiRational(1, 1)));
  e = zx::PiExpression{zx::PiRational((zx::PI / 2) - eps)};
  zx::roundToClifford(e, 1e-9);
  EXPECT_EQ(e, zx::PiExpression(zx::PiRational(1, 2)));
  e = zx::PiExpression{zx::PiRational((-zx::PI / 2) - eps)};
  zx::roundToClifford(e, 1e-9);
  EXPECT_EQ(e, zx::PiExpression(zx::PiRational(-1, 2)));
}

TEST_F(ExpressionTest, Clifford) {
  zx::PiExpression e{zx::PiRational(zx::PI)};

  EXPECT_TRUE(zx::isPauli(e));
  EXPECT_TRUE(zx::isClifford(e));
  EXPECT_FALSE(zx::isProperClifford(e));

  e = zx::PiExpression{zx::PiRational(zx::PI / 2)};
  EXPECT_FALSE(zx::isPauli(e));
  EXPECT_TRUE(zx::isClifford(e));
  EXPECT_TRUE(zx::isProperClifford(e));

  e = zx::PiExpression{zx::PiRational(zx::PI / 4)};
  EXPECT_FALSE(zx::isPauli(e));
  EXPECT_FALSE(zx::isClifford(e));
  EXPECT_FALSE(zx::isProperClifford(e));
}

TEST_F(ExpressionTest, Convertability) {
  const sym::Expression<double, double> e({x, y}, zx::PI);
  const auto piE = e.convert<zx::PiRational>();

  EXPECT_EQ(piE, zx::PiExpression({x, y}, zx::PiRational(1, 1)));
}

TEST_F(ExpressionTest, Instantiation) {
  sym::Expression<double, double> e(2 * x, y);

  const sym::VariableAssignment assignment{{sym::Variable{"x"}, 2.0},
                                           {sym::Variable{"y"}, 1.0}};

  EXPECT_PRED_FORMAT2(testing::DoubleLE, e.evaluate(assignment), 5.0);

  e += z;
  EXPECT_THROW([[maybe_unused]] const auto h = e.evaluate(assignment),
               sym::SymbolicException);
}
