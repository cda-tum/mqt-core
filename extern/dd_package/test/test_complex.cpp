/*
 * This file is part of the JKQ DD Package which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum_dd/ for more information.
 */

#include "DDcomplex.h"

#include "gtest/gtest.h"
#include <memory>

using CN = dd::ComplexNumbers;

TEST(DDComplexTest, TrivialTest) {
    auto         cn           = std::make_unique<dd::ComplexNumbers>();
    unsigned int before_count = cn->count;

    auto a = cn->getCachedComplex(2, -3);
    auto b = cn->getCachedComplex(3, 2);

    auto r0 = cn->getCachedComplex(12, -5);
    auto r1 = cn->mulCached(a, b);

    ASSERT_PRED2(static_cast<bool (*)(const dd::Complex&, const dd::Complex&)>(dd::ComplexNumbers::equals), r0, r1);

    cn->lookup(a);
    cn->lookup(b);
    unsigned int between_count = cn->count;
    // the lookup increases the count in the complex table
    ASSERT_LT(before_count, between_count);
    cn->releaseCached(a);
    cn->releaseCached(b);
    cn->releaseCached(r0);
    cn->releaseCached(r1);
    cn->garbageCollect();
    // since lookup does not increase the ref count, garbage collection removes the new values
    unsigned int end_count = cn->count;
    ASSERT_EQ(before_count, end_count);
}

TEST(DDComplexTest, ComplexNumberCreation) {
    auto cn = dd::ComplexNumbers();
    EXPECT_EQ(cn.lookup(cn.ZERO), cn.ZERO);
    EXPECT_EQ(cn.lookup(cn.ONE), cn.ONE);
    EXPECT_EQ(cn.lookup(1e-14, 0.), cn.ZERO);
    EXPECT_EQ(cn.val(cn.lookup(1e-14, 1.).r), 0.);
    EXPECT_EQ(cn.val(cn.lookup(1e-14, 1.).i), 1.);
    EXPECT_EQ(cn.val(cn.lookup(1e-14, -1.).r), 0.);
    EXPECT_EQ(cn.val(cn.lookup(1e-14, -1.).i), -1.);
    EXPECT_EQ(cn.val(cn.lookup(-1., -1.).r), -1.);
    EXPECT_EQ(cn.val(cn.lookup(-1., -1.).i), -1.);
    auto c = cn.lookup(0., -1.);
    std::cout << c << std::endl;
    EXPECT_EQ(cn.val(cn.lookup(c).r), 0.);
    EXPECT_EQ(cn.val(cn.lookup(c).i), -1.);
    c = cn.lookup(0., 1.);
    EXPECT_EQ(cn.val(cn.lookup(c).r), 0.);
    EXPECT_EQ(cn.val(cn.lookup(c).i), 1.);
    c = cn.lookup(0., -0.5);
    std::cout << c << std::endl;
    EXPECT_EQ(cn.val(cn.lookup(c).r), 0.);
    EXPECT_EQ(cn.val(cn.lookup(c).i), -0.5);
    c = cn.lookup(-1., -1.);
    EXPECT_EQ(cn.val(cn.lookup(c).r), -1.);
    EXPECT_EQ(cn.val(cn.lookup(c).i), -1.);
    std::cout << c << std::endl;

    auto e = cn.lookup(1., -1.);
    std::cout << e << std::endl;
    std::cout << dd::ComplexValue{1., 1.} << std::endl;
    std::cout << dd::ComplexValue{1., -1.} << std::endl;
    std::cout << dd::ComplexValue{1., -0.5} << std::endl;
    cn.printComplexTable();
    cn.statistics();
    std::cout << "Cache size: " << cn.cacheSize() << std::endl;
}

TEST(DDComplexTest, ComplexNumberArithmetic) {
    auto cn = dd::ComplexNumbers();
    auto c  = cn.lookup(0., 1.);
    auto d  = dd::ComplexNumbers::conj(c);
    EXPECT_EQ(cn.val(d.r), 0.);
    EXPECT_EQ(cn.val(d.i), -1.);
    c = cn.lookup(-1., -1.);
    d = dd::ComplexNumbers::neg(c);
    EXPECT_EQ(cn.val(d.r), 1.);
    EXPECT_EQ(cn.val(d.i), 1.);
    c = cn.lookup(0.5, 0.5);
    dd::ComplexNumbers::incRef(c);
    d = cn.lookup(-0.5, 0.5);
    dd::ComplexNumbers::incRef(d);
    auto e = cn.getTempCachedComplex();
    dd::ComplexNumbers::sub(e, c, d);
    dd::ComplexNumbers::decRef(c);
    dd::ComplexNumbers::decRef(d);
    e = cn.lookup(e);
    EXPECT_EQ(e, cn.ONE);
    auto f = cn.getTempCachedComplex();
    dd::ComplexNumbers::div(f, dd::ComplexNumbers::ZERO, dd::ComplexNumbers::ONE);
}

TEST(DDComplexTest, NearZeroLookup) {
    auto cn = dd::ComplexNumbers();
    auto c  = cn.getTempCachedComplex(dd::ComplexNumbers::TOLERANCE / 10., dd::ComplexNumbers::TOLERANCE / 10.);
    auto d  = cn.lookup(c);
    EXPECT_EQ(d.r, cn.ZERO.r);
    EXPECT_EQ(d.i, cn.ZERO.i);
}

TEST(DDComplexTest, GarbageCollectSomeInBucket) {
    auto cn = dd::ComplexNumbers();

    fp num = 0.25;
    cn.lookup(num, 0.0);

    fp num2 = num + 2. * dd::ComplexNumbers::TOLERANCE;
    dd::ComplexNumbers::incRef(cn.lookup(num2, 0.0)); // num2 should be placed in same bucket as num

    auto key = dd::ComplexNumbers::getKey(num);
    EXPECT_NEAR(cn.ComplexTable[key]->val, num2, dd::ComplexNumbers::TOLERANCE);
    EXPECT_NEAR(cn.ComplexTable[key]->next->val, num, dd::ComplexNumbers::TOLERANCE);

    cn.garbageCollect(); // num should be collected
    EXPECT_NEAR(cn.ComplexTable[key]->val, num2, dd::ComplexNumbers::TOLERANCE);
    EXPECT_EQ(cn.ComplexTable[key]->next, nullptr);
}

TEST(DDComplexTest, LookupInNeighbouringBuckets) {
    auto cn = dd::ComplexNumbers();

    // lower border of a bucket
    fp bucketBorder = 0.25 * dd::ComplexNumbers::NBUCKET / (dd::ComplexNumbers::NBUCKET - 1);

    // insert a number slightly away from the border
    fp num = bucketBorder + 2 * dd::ComplexNumbers::TOLERANCE;
    cn.lookup(num, 0.0);
    auto key = dd::ComplexNumbers::getKey(num);
    EXPECT_EQ(key, dd::ComplexNumbers::NBUCKET / 4);

    // insert a number barely in the bucket below
    fp num2 = bucketBorder - dd::ComplexNumbers::TOLERANCE / 10;
    cn.lookup(num2, 0.0);
    auto key2 = dd::ComplexNumbers::getKey(num2);
    EXPECT_EQ(key2, dd::ComplexNumbers::NBUCKET / 4 - 1);

    // insert another number in the bucket below a bit farther away from the border
    fp num3 = bucketBorder - 2 * dd::ComplexNumbers::TOLERANCE;
    cn.lookup(num3, 0.0);
    auto key3 = dd::ComplexNumbers::getKey(num3);
    EXPECT_EQ(key3, dd::ComplexNumbers::NBUCKET / 4 - 1);

    // insert border number that is too far away from the number in the bucket, but is close enough to a number in the bucket below
    fp   num4 = bucketBorder;
    auto c    = cn.lookup(num4, 0.0);
    auto key4 = dd::ComplexNumbers::getKey(num4 - dd::ComplexNumbers::TOLERANCE);
    EXPECT_EQ(key2, key4);
    EXPECT_NEAR(c.r->val, num2, dd::ComplexNumbers::TOLERANCE);

    // insert a number in the higher bucket
    fp nextBorder = bucketBorder + 1.0 / (dd::ComplexNumbers::NBUCKET - 1);
    fp num5       = nextBorder;
    cn.lookup(num5, 0.0);
    auto key5 = dd::ComplexNumbers::getKey(num5);
    EXPECT_EQ(key5, dd::ComplexNumbers::NBUCKET / 4 + 1);

    // search for a number in the lower bucket that is ultimately close enough to a number in the upper bucket
    fp   num6 = nextBorder - dd::ComplexNumbers::TOLERANCE / 10;
    auto d    = cn.lookup(num6, 0.0);
    auto key6 = dd::ComplexNumbers::getKey(num6 + dd::ComplexNumbers::TOLERANCE);
    EXPECT_EQ(key5, key6);
    EXPECT_NEAR(d.r->val, num5, dd::ComplexNumbers::TOLERANCE);
}

TEST(DDComplexTest, ComplexValueEquals) {
    dd::ComplexValue a{1.0, 0.0};
    dd::ComplexValue a_tol{1.0 + dd::ComplexNumbers::TOLERANCE / 10, 0.0};
    dd::ComplexValue b{0.0, 1.0};
    EXPECT_TRUE(dd::ComplexNumbers::equals(a, a_tol));
    EXPECT_FALSE(dd::ComplexNumbers::equals(a, b));
}

TEST(DDComplexTest, NumberPrinting) {
    auto cn       = dd::ComplexNumbers();
    auto imag     = cn.lookup(0., 1.);
    auto imag_str = dd::ComplexNumbers::toString(imag, false);
    EXPECT_STREQ(imag_str.c_str(), "1i");
    auto imag_str_formatted = dd::ComplexNumbers::toString(imag, true);
    EXPECT_STREQ(imag_str_formatted.c_str(), "+i");

    auto superposition     = cn.lookup(CN::SQRT_2, CN::SQRT_2);
    auto superposition_str = dd::ComplexNumbers::toString(superposition, false, 3);
    EXPECT_STREQ(superposition_str.c_str(), "0.707+0.707i");
    auto superposition_str_formatted = dd::ComplexNumbers::toString(superposition, true, 3);
    EXPECT_STREQ(superposition_str_formatted.c_str(), "√½(1+i)");
    auto neg_superposition               = cn.lookup(CN::SQRT_2, -CN::SQRT_2);
    auto neg_superposition_str_formatted = dd::ComplexNumbers::toString(neg_superposition, true, 3);
    EXPECT_STREQ(neg_superposition_str_formatted.c_str(), "√½(1-i)");

    std::stringstream ss{};
    dd::ComplexNumbers::printFormattedReal(ss, CN::SQRT_2, false);
    EXPECT_STREQ(ss.str().c_str(), "√½");
    ss.str("");
    dd::ComplexNumbers::printFormattedReal(ss, CN::SQRT_2, true);
    EXPECT_STREQ(ss.str().c_str(), "+√½i");
    ss.str("");

    dd::ComplexNumbers::printFormattedReal(ss, 0.5, false);
    EXPECT_STREQ(ss.str().c_str(), "½");
    ss.str("");
    dd::ComplexNumbers::printFormattedReal(ss, 0.5, true);
    EXPECT_STREQ(ss.str().c_str(), "+½i");
    ss.str("");

    dd::ComplexNumbers::printFormattedReal(ss, 0.5 * CN::SQRT_2, false);
    EXPECT_STREQ(ss.str().c_str(), "√½ ½");
    ss.str("");
    dd::ComplexNumbers::printFormattedReal(ss, 0.5 * CN::SQRT_2, true);
    EXPECT_STREQ(ss.str().c_str(), "+√½ ½i");
    ss.str("");

    dd::ComplexNumbers::printFormattedReal(ss, 0.25, false);
    EXPECT_STREQ(ss.str().c_str(), "½**2");
    ss.str("");
    dd::ComplexNumbers::printFormattedReal(ss, 0.25, true);
    EXPECT_STREQ(ss.str().c_str(), "+½**2i");
    ss.str("");

    dd::ComplexNumbers::printFormattedReal(ss, 0.25 * CN::SQRT_2, false);
    EXPECT_STREQ(ss.str().c_str(), "√½ ½**2");
    ss.str("");
    dd::ComplexNumbers::printFormattedReal(ss, 0.25 * CN::SQRT_2, true);
    EXPECT_STREQ(ss.str().c_str(), "+√½ ½**2i");
    ss.str("");

    dd::ComplexNumbers::printFormattedReal(ss, CN::PI, false);
    EXPECT_STREQ(ss.str().c_str(), "π");
    ss.str("");
    dd::ComplexNumbers::printFormattedReal(ss, CN::PI, true);
    EXPECT_STREQ(ss.str().c_str(), "+πi");
    ss.str("");

    dd::ComplexNumbers::printFormattedReal(ss, 0.5 * CN::PI, false);
    EXPECT_STREQ(ss.str().c_str(), "½ π");
    ss.str("");
    dd::ComplexNumbers::printFormattedReal(ss, 0.5 * CN::PI, true);
    EXPECT_STREQ(ss.str().c_str(), "+½ πi");
    ss.str("");

    dd::ComplexNumbers::printFormattedReal(ss, 0.25 * CN::PI, false);
    EXPECT_STREQ(ss.str().c_str(), "½**2 π");
    ss.str("");
    dd::ComplexNumbers::printFormattedReal(ss, 0.25 * CN::PI, true);
    EXPECT_STREQ(ss.str().c_str(), "+½**2 πi");
    ss.str("");

    dd::ComplexNumbers::printFormattedReal(ss, 0.1234, false);
    EXPECT_STREQ(ss.str().c_str(), "0.1234");
    ss.str("");
    dd::ComplexNumbers::printFormattedReal(ss, 0.1234, true);
    EXPECT_STREQ(ss.str().c_str(), "+0.1234i");
    ss.str("");
}
