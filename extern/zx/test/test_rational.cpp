#include "Definitions.hpp"
#include "Rational.hpp"

#include <gtest/gtest.h>
#include <iostream>

class RationalTest: public ::testing::Test {};

TEST_F(RationalTest, normalize) {
    zx::PiRational r(-33, 16);
    EXPECT_EQ(r, zx::PiRational(-1, 16));
}

TEST_F(RationalTest, from_double) {
    zx::PiRational r(-zx::PI / 8);
    EXPECT_EQ(r, zx::PiRational(-1, 8));
}

TEST_F(RationalTest, from_double_2) {
    zx::PiRational r(-3 * zx::PI / 4);
    EXPECT_EQ(r, zx::PiRational(-3, 4));
}

TEST_F(RationalTest, from_double_3) {
    zx::PiRational r(-7 * zx::PI / 8);
    EXPECT_EQ(r, zx::PiRational(-7, 8));
}

TEST_F(RationalTest, from_double_4) {
    zx::PiRational r(-1 * zx::PI / 32);
    EXPECT_EQ(r, zx::PiRational(-1, 32));
}

TEST_F(RationalTest, from_double_5) {
    zx::PiRational r(5000 * zx::PI + zx::PI / 4);
    EXPECT_EQ(r, zx::PiRational(1, 4));
}

TEST_F(RationalTest, from_double_6) {
    zx::PiRational r(-5000 * zx::PI + 5 * zx::PI / 4);
    EXPECT_EQ(r, zx::PiRational(-3, 4));
}

// TEST_F(RationalTest, from_double_7) {
//     zx::PiRational r(0.1);
//     EXPECT_EQ(r, zx::PiRational(-3, 4));
// }

TEST_F(RationalTest, add) {
    zx::PiRational r0(1, 8);
    zx::PiRational r1(7, 8);
    auto           r = r0 + r1;

    EXPECT_EQ(r, 1);
}

TEST_F(RationalTest, add_2) {
    zx::PiRational r0(9, 8);
    zx::PiRational r1(7, 8);
    auto           r = r0 + r1;

    EXPECT_EQ(r, 0);
}

TEST_F(RationalTest, sub) {
    zx::PiRational r0(9, 8);
    zx::PiRational r1(-7, 8);
    auto           r = r0 - r1;

    EXPECT_EQ(r, 0);
}

TEST_F(RationalTest, sub_2) {
    zx::PiRational r0(-1, 2);
    zx::PiRational r1(1, 2);
    auto           r = r0 - r1;

    EXPECT_EQ(r, 1);
}

TEST_F(RationalTest, mul) {
    zx::PiRational r0(1, 8);
    zx::PiRational r1(1, 2);
    auto           r = r0 * r1;

    EXPECT_EQ(r, zx::PiRational(1, 16));
}

TEST_F(RationalTest, mul_2) {
    zx::PiRational r0(1, 8);
    zx::PiRational r1(0, 1);
    auto           r = r0 * r1;

    EXPECT_EQ(r, 0);
}

TEST_F(RationalTest, div) {
    zx::PiRational r0(1, 2);
    zx::PiRational r1(1, 2);
    auto           r = r0 / r1;

    EXPECT_EQ(r, 1);
}

TEST_F(RationalTest, approximateDivPi) {
    zx::PiRational r(1000000000000000 - 1, 1000000000000000);
    EXPECT_TRUE(r.isCloseDivPi(1, 1e-9));
}

TEST_F(RationalTest, approximate) {
    zx::PiRational r(1, 1);
    EXPECT_TRUE(r.isClose(3.14159, 1e-5));
}
