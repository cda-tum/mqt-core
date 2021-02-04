#include <memory>

#include "DDcomplex.h"
#include "gtest/gtest.h"

TEST(DDComplexTest, TrivialTest) {
    auto cn = std::make_unique<dd::ComplexNumbers>();
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

	auto e = cn.lookup(1., -1);
	std::cout << e << std::endl;
	std::cout << dd::ComplexValue{1., 1.} << std::endl;
	std::cout << dd::ComplexValue{1., -1.} << std::endl;
	std::cout << dd::ComplexValue{1., -0.5} << std::endl;
	cn.printComplexTable();
	cn.statistics();
	cn.cacheSize();
}

TEST(DDComplexTest, ComplexNumberArithmetic) {
	auto cn = dd::ComplexNumbers();
	auto c = cn.lookup(0., 1.);
	auto d = dd::ComplexNumbers::conj(c);
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
