#include "Expression.hpp"
#include "ZXDiagram.hpp"

#include <gtest/gtest.h>

class ExpressionTest: public ::testing::Test {
public:
    // const zx::Variable x_var{0, "x"};
    // const zx::Variable y_var{1, "y"};
    // const zx::Variable z_var{2, "z"};

    zx::Term x{zx::Variable(0, "x")};
    zx::Term y{zx::Variable(1, "y")};
    zx::Term z{zx::Variable(2, "z")};

protected:
    virtual void SetUp() {}
};

TEST_F(ExpressionTest, basic_ops_1) {
    zx::Expression e(x);

    EXPECT_EQ(1, e.numTerms());
    EXPECT_EQ(zx::PiRational(0, 1), e.getConst());

    e += x; // zx::Term(x);

    EXPECT_EQ(1, e.numTerms());
    EXPECT_EQ(zx::PiRational(0, 1), e.getConst());
    EXPECT_PRED_FORMAT2(testing::FloatLE, e[0].getCoeff(), 2.0);

    e += y;
    EXPECT_EQ(2, e.numTerms());
    EXPECT_PRED_FORMAT2(testing::FloatLE, e[0].getCoeff(), 2.0);
    EXPECT_PRED_FORMAT2(testing::FloatLE, e[1].getCoeff(), 1.0);
    EXPECT_EQ(e[0].getVar().name, "x");
    EXPECT_EQ(e[1].getVar().name, "y");
}

TEST_F(ExpressionTest, basic_ops_2) {
    zx::Expression e1;
    e1 += x;
    e1 += 10.0 * y;
    e1 += 5.0 * z;
    e1 += zx::PiRational(1, 2);

    zx::Expression e2;
    e2 += -5.0 * x;
    e2 += -10.0 * y;
    e2 += -4.9 * z;
    e2 += zx::PiRational(3, 2);

    auto sum = e1 + e2;

    EXPECT_EQ(2, sum.numTerms());
    EXPECT_PRED_FORMAT2(testing::FloatLE, sum[0].getCoeff(), -4.0);
    EXPECT_PRED_FORMAT2(testing::FloatLE, sum[1].getCoeff(), 0.1);
    EXPECT_EQ(sum[0].getVar().name, "x");
    EXPECT_EQ(sum[1].getVar().name, "z");
    EXPECT_EQ(sum.getConst(), zx::PiRational(0, 1));
}
