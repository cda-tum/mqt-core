/*
 * This file is part of the JKQ DD Package which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum_dd/ for more information.
 */

#include "dd/ComplexNumbers.hpp"

#include "gtest/gtest.h"
#include <memory>

using namespace dd;
using CN = ComplexNumbers;

TEST(DDComplexTest, TrivialTest) {
    auto         cn           = std::make_unique<ComplexNumbers>();
    unsigned int before_count = cn->cacheCount();

    auto a = cn->getCached(2, -3);
    auto b = cn->getCached(3, 2);

    auto r0 = cn->getCached(12, -5);
    auto r1 = cn->mulCached(a, b);

    cn->lookup(a);
    cn->lookup(b);
    unsigned int between_count = cn->cacheCount();
    // the lookup increases the count in the complex table
    ASSERT_TRUE(before_count < between_count);
    cn->returnToCache(a);
    cn->returnToCache(b);
    cn->returnToCache(r0);
    cn->returnToCache(r1);
    cn->garbageCollect(true);
    // since lookup does not increase the ref count, garbage collection removes the new values
    unsigned int end_count = cn->cacheCount();
    ASSERT_EQ(before_count, end_count);

    EXPECT_NO_THROW(cn->incRef({nullptr, nullptr}));
    EXPECT_NO_THROW(cn->decRef({nullptr, nullptr}));
}

TEST(DDComplexTest, ComplexNumberCreation) {
    auto cn = ComplexNumbers();
    EXPECT_EQ(cn.lookup(Complex::zero), Complex::zero);
    EXPECT_EQ(cn.lookup(Complex::one), Complex::one);
    EXPECT_EQ(cn.lookup(1e-14, 0.), Complex::zero);
    EXPECT_EQ(CTEntry::val(cn.lookup(1e-14, 1.).r), 0.);
    EXPECT_EQ(CTEntry::val(cn.lookup(1e-14, 1.).i), 1.);
    EXPECT_EQ(CTEntry::val(cn.lookup(1e-14, -1.).r), 0.);
    EXPECT_EQ(CTEntry::val(cn.lookup(1e-14, -1.).i), -1.);
    EXPECT_EQ(CTEntry::val(cn.lookup(-1., -1.).r), -1.);
    EXPECT_EQ(CTEntry::val(cn.lookup(-1., -1.).i), -1.);
    auto c = cn.lookup(0., -1.);
    std::cout << c << std::endl;
    EXPECT_EQ(CTEntry::val(cn.lookup(c).r), 0.);
    EXPECT_EQ(CTEntry::val(cn.lookup(c).i), -1.);
    c = cn.lookup(0., 1.);
    EXPECT_EQ(CTEntry::val(cn.lookup(c).r), 0.);
    EXPECT_EQ(CTEntry::val(cn.lookup(c).i), 1.);
    c = cn.lookup(0., -0.5);
    std::cout << c << std::endl;
    EXPECT_EQ(CTEntry::val(cn.lookup(c).r), 0.);
    EXPECT_EQ(CTEntry::val(cn.lookup(c).i), -0.5);
    c = cn.lookup(-1., -1.);
    EXPECT_EQ(CTEntry::val(cn.lookup(c).r), -1.);
    EXPECT_EQ(CTEntry::val(cn.lookup(c).i), -1.);
    std::cout << c << std::endl;

    auto e = cn.lookup(1., -1.);
    std::cout << e << std::endl;
    std::cout << ComplexValue{1., 1.} << std::endl;
    std::cout << ComplexValue{1., -1.} << std::endl;
    std::cout << ComplexValue{1., -0.5} << std::endl;
    cn.complexTable.print();
    cn.complexTable.printStatistics();
}

TEST(DDComplexTest, ComplexNumberArithmetic) {
    auto cn = ComplexNumbers();
    auto c  = cn.lookup(0., 1.);
    auto d  = ComplexNumbers::conj(c);
    EXPECT_EQ(CTEntry::val(d.r), 0.);
    EXPECT_EQ(CTEntry::val(d.i), -1.);
    c = cn.lookup(-1., -1.);
    d = ComplexNumbers::neg(c);
    EXPECT_EQ(CTEntry::val(d.r), 1.);
    EXPECT_EQ(CTEntry::val(d.i), 1.);
    c = cn.lookup(0.5, 0.5);
    ComplexNumbers::incRef(c);
    d = cn.lookup(-0.5, 0.5);
    ComplexNumbers::incRef(d);
    auto e = cn.getTemporary();
    ComplexNumbers::sub(e, c, d);
    ComplexNumbers::decRef(c);
    ComplexNumbers::decRef(d);
    e = cn.lookup(e);
    EXPECT_EQ(e, Complex::one);
    auto f = cn.getTemporary();
    ComplexNumbers::div(f, Complex::zero, Complex::one);

    dd::ComplexValue zero{0., 0.};
    dd::ComplexValue one{1., 0.};
    EXPECT_EQ(one + zero, one);
}

TEST(DDComplexTest, NearZeroLookup) {
    auto cn = ComplexNumbers();
    auto c  = cn.getTemporary(ComplexTable<>::tolerance() / 10., ComplexTable<>::tolerance() / 10.);
    auto d  = cn.lookup(c);
    EXPECT_EQ(d.r, Complex::zero.r);
    EXPECT_EQ(d.i, Complex::zero.i);
}

TEST(DDComplexTest, SortedBuckets) {
    auto     ct  = ComplexTable<>{};
    const fp num = 0.25;

    const std::array<dd::fp, 7> numbers = {
            num + 2. * ComplexTable<>::tolerance(),
            num - 2. * ComplexTable<>::tolerance(),
            num + 4. * ComplexTable<>::tolerance(),
            num,
            num - 4. * ComplexTable<>::tolerance(),
            num + 6. * ComplexTable<>::tolerance(),
            num + 8. * ComplexTable<>::tolerance()};

    const std::size_t the_bucket = ct.hash(num);

    for (auto const& number: numbers) {
        ASSERT_EQ(the_bucket, ct.hash(number));
        ct.lookup(number);
    }

    CTEntry* p = ct.getTable().at(the_bucket);
    ASSERT_NE(p, nullptr);

    dd::fp      last    = std::numeric_limits<dd::fp>::min();
    std::size_t counter = 0;
    while (p != nullptr) {
        ASSERT_LT(last, p->value);
        p = p->next;
        ++counter;
    }
    ct.printStatistics(std::cout);
    EXPECT_EQ(ct.getStatistics().at("lowerNeighbors"), 1); // default insertion of 0.5 is close to lower bucket
    EXPECT_EQ(counter, numbers.size());
}

TEST(DDComplexTest, GarbageCollectSomeInBucket) {
    auto cn = ComplexNumbers();
    EXPECT_EQ(cn.garbageCollect(), 0);

    const fp num = 0.25;
    cn.lookup(num, 0.0);

    const fp num2 = num + 2. * ComplexTable<>::tolerance();
    ComplexNumbers::incRef(cn.lookup(num2, 0.0)); // num2 should be placed in same bucket as num

    auto key  = ComplexTable<>::hash(num);
    auto key2 = ComplexTable<>::hash(num2);
    ASSERT_EQ(key, key2);

    auto* p = cn.complexTable.getTable()[key];
    EXPECT_NEAR(p->value, num, ComplexTable<>::tolerance());

    ASSERT_NE(p->next, nullptr);
    EXPECT_NEAR((p->next)->value, num2, ComplexTable<>::tolerance());

    cn.garbageCollect(true); // num should be collected
    EXPECT_NEAR(cn.complexTable.getTable()[key]->value, num2, ComplexTable<>::tolerance());
    EXPECT_EQ(cn.complexTable.getTable()[key]->next, nullptr);
}

TEST(DDComplexTest, LookupInNeighbouringBuckets) {
    auto                  cn      = ComplexNumbers();
    constexpr std::size_t NBUCKET = ComplexTable<>::MASK + 1;

    // lower border of a bucket
    fp bucketBorder = (0.25 * NBUCKET - 0.5) / (NBUCKET - 1);

    // insert a number slightly away from the border
    fp num = bucketBorder + 2 * ComplexTable<>::tolerance();
    cn.lookup(num, 0.0);
    auto key = ComplexTable<>::hash(num);
    EXPECT_EQ(key, NBUCKET / 4);

    // insert a number barely in the bucket below
    fp num2 = bucketBorder - ComplexTable<>::tolerance() / 10;
    cn.lookup(num2, 0.0);
    auto key2 = ComplexTable<>::hash(num2);
    EXPECT_EQ(key2, NBUCKET / 4 - 1);

    // insert another number in the bucket below a bit farther away from the border
    fp num3 = bucketBorder - 2 * ComplexTable<>::tolerance();
    cn.lookup(num3, 0.0);
    auto key3 = ComplexTable<>::hash(num3);
    EXPECT_EQ(key3, NBUCKET / 4 - 1);

    // insert border number that is too far away from the number in the bucket, but is close enough to a number in the bucket below
    fp   num4 = bucketBorder;
    auto c    = cn.lookup(num4, 0.0);
    auto key4 = ComplexTable<>::hash(num4 - ComplexTable<>::tolerance());
    EXPECT_EQ(key2, key4);
    EXPECT_NEAR(c.r->value, num2, ComplexTable<>::tolerance());

    // insert a number in the higher bucket
    fp nextBorder = bucketBorder + 1.0 / (NBUCKET - 1);
    fp num5       = nextBorder;
    cn.lookup(num5, 0.0);
    auto key5 = ComplexTable<>::hash(num5);
    EXPECT_EQ(key5, NBUCKET / 4 + 1);

    // search for a number in the lower bucket that is ultimately close enough to a number in the upper bucket
    fp   num6 = nextBorder - ComplexTable<>::tolerance() / 10;
    auto d    = cn.lookup(num6, 0.0);
    auto key6 = ComplexTable<>::hash(num6 + ComplexTable<>::tolerance());
    EXPECT_EQ(key5, key6);
    EXPECT_NEAR(d.r->value, num5, ComplexTable<>::tolerance());
}

TEST(DDComplexTest, ComplexValueEquals) {
    ComplexValue a{1.0, 0.0};
    ComplexValue a_tol{1.0 + ComplexTable<>::tolerance() / 10, 0.0};
    ComplexValue b{0.0, 1.0};
    EXPECT_TRUE(a.approximatelyEquals(a_tol));
    EXPECT_FALSE(a.approximatelyEquals(b));
}

TEST(DDComplexTest, NumberPrinting) {
    auto cn       = ComplexNumbers();
    auto imag     = cn.lookup(0., 1.);
    auto imag_str = imag.toString(false);
    EXPECT_STREQ(imag_str.c_str(), "1i");
    auto imag_str_formatted = imag.toString(true);
    EXPECT_STREQ(imag_str_formatted.c_str(), "+i");

    auto superposition     = cn.lookup(dd::SQRT2_2, dd::SQRT2_2);
    auto superposition_str = superposition.toString(false, 3);
    EXPECT_STREQ(superposition_str.c_str(), "0.707+0.707i");
    auto superposition_str_formatted = superposition.toString(true, 3);
    EXPECT_STREQ(superposition_str_formatted.c_str(), "√½(1+i)");
    auto neg_superposition               = cn.lookup(dd::SQRT2_2, -dd::SQRT2_2);
    auto neg_superposition_str_formatted = neg_superposition.toString(true, 3);
    EXPECT_STREQ(neg_superposition_str_formatted.c_str(), "√½(1-i)");

    std::stringstream ss{};
    ComplexValue::printFormatted(ss, dd::SQRT2_2, false);
    EXPECT_STREQ(ss.str().c_str(), "√½");
    ss.str("");
    ComplexValue::printFormatted(ss, dd::SQRT2_2, true);
    EXPECT_STREQ(ss.str().c_str(), "+√½i");
    ss.str("");

    ComplexValue::printFormatted(ss, 0.5, false);
    EXPECT_STREQ(ss.str().c_str(), "½");
    ss.str("");
    ComplexValue::printFormatted(ss, 0.5, true);
    EXPECT_STREQ(ss.str().c_str(), "+½i");
    ss.str("");

    ComplexValue::printFormatted(ss, 0.5 * dd::SQRT2_2, false);
    EXPECT_STREQ(ss.str().c_str(), "√½ ½");
    ss.str("");
    ComplexValue::printFormatted(ss, 0.5 * dd::SQRT2_2, true);
    EXPECT_STREQ(ss.str().c_str(), "+√½ ½i");
    ss.str("");

    ComplexValue::printFormatted(ss, 0.25, false);
    EXPECT_STREQ(ss.str().c_str(), "½**2");
    ss.str("");
    ComplexValue::printFormatted(ss, 0.25, true);
    EXPECT_STREQ(ss.str().c_str(), "+½**2i");
    ss.str("");

    ComplexValue::printFormatted(ss, 0.25 * dd::SQRT2_2, false);
    EXPECT_STREQ(ss.str().c_str(), "√½ ½**2");
    ss.str("");
    ComplexValue::printFormatted(ss, 0.25 * dd::SQRT2_2, true);
    EXPECT_STREQ(ss.str().c_str(), "+√½ ½**2i");
    ss.str("");

    ComplexValue::printFormatted(ss, dd::PI, false);
    EXPECT_STREQ(ss.str().c_str(), "π");
    ss.str("");
    ComplexValue::printFormatted(ss, dd::PI, true);
    EXPECT_STREQ(ss.str().c_str(), "+πi");
    ss.str("");

    ComplexValue::printFormatted(ss, 0.5 * dd::PI, false);
    EXPECT_STREQ(ss.str().c_str(), "½ π");
    ss.str("");
    ComplexValue::printFormatted(ss, 0.5 * dd::PI, true);
    EXPECT_STREQ(ss.str().c_str(), "+½ πi");
    ss.str("");

    ComplexValue::printFormatted(ss, 0.25 * dd::PI, false);
    EXPECT_STREQ(ss.str().c_str(), "½**2 π");
    ss.str("");
    ComplexValue::printFormatted(ss, 0.25 * dd::PI, true);
    EXPECT_STREQ(ss.str().c_str(), "+½**2 πi");
    ss.str("");

    ComplexValue::printFormatted(ss, 0.1234, false);
    EXPECT_STREQ(ss.str().c_str(), "0.1234");
    ss.str("");
    ComplexValue::printFormatted(ss, 0.1234, true);
    EXPECT_STREQ(ss.str().c_str(), "+0.1234i");
    ss.str("");
}

TEST(DDComplexTest, MaxRefCountReached) {
    auto cn       = ComplexNumbers();
    auto c        = cn.lookup(SQRT2_2, SQRT2_2);
    auto max      = std::numeric_limits<decltype(c.r->refCount)>::max();
    c.r->refCount = max;
    CN::incRef(c);
    EXPECT_EQ(c.r->refCount, max);
    EXPECT_EQ(c.i->refCount, max);
}

TEST(DDComplexTest, ComplexTableAllocation) {
    auto cn     = ComplexNumbers();
    auto allocs = cn.complexTable.getAllocations();
    std::cout << allocs << std::endl;
    std::vector<ComplexTable<>::Entry*> nums{allocs};
    // get all the numbers that are pre-allocated
    for (auto i = 0U; i < allocs; ++i) {
        nums[i] = cn.complexTable.getEntry();
    }

    // trigger new allocation
    [[maybe_unused]] auto num = cn.complexTable.getEntry();
    EXPECT_EQ(cn.complexTable.getAllocations(), (1. + cn.complexTable.getGrowthFactor()) * allocs);

    // clearing the complex table should reduce the allocated size to the original size
    cn.complexTable.clear();
    EXPECT_EQ(cn.complexTable.getAllocations(), allocs);

    EXPECT_TRUE(cn.complexTable.availableEmpty());
    // obtain entry
    auto entry = cn.complexTable.getEntry();
    // immediately return entry
    cn.complexTable.returnEntry(entry);
    EXPECT_FALSE(cn.complexTable.availableEmpty());
    // obtain the same entry again, but this time from the available stack
    auto entry2 = cn.complexTable.getEntry();
    EXPECT_EQ(entry, entry2);
}

TEST(DDComplexTest, ComplexCacheAllocation) {
    auto cn     = ComplexNumbers();
    auto allocs = cn.complexCache.getAllocations();
    std::cout << allocs << std::endl;
    std::vector<Complex> cnums{allocs};
    // get all the cached complex numbers that are pre-allocated
    for (auto i = 0U; i < allocs; i += 2) {
        cnums[i % 2] = cn.getCached();
    }

    // trigger new allocation for obtaining a complex from cache
    [[maybe_unused]] auto cnum = cn.getCached();
    EXPECT_EQ(cn.complexCache.getAllocations(), (1. + cn.complexCache.getGrowthFactor()) * allocs);

    // clearing the cache should reduce the allocated size to the original size
    cn.complexCache.clear();
    EXPECT_EQ(cn.complexCache.getAllocations(), allocs);

    // get all the cached complex numbers again
    for (auto i = 0U; i < allocs; i += 2) {
        cnums[i % 2] = cn.getCached();
    }

    // trigger new allocation for obtaining a temporary from cache
    [[maybe_unused]] auto cnumtmp = cn.getTemporary();
    EXPECT_EQ(cn.complexCache.getAllocations(), (1. + cn.complexCache.getGrowthFactor()) * allocs);

    // clearing the unique table should reduce the allocated size to the original size
    cn.complexCache.clear();
    EXPECT_EQ(cn.complexCache.getAllocations(), allocs);
}
