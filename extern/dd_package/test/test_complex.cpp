/*
 * This file is part of the JKQ DD Package which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum_dd/ for more information.
 */

#include "dd/ComplexNumbers.hpp"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <limits>
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
    auto cn = std::make_unique<ComplexNumbers>();
    EXPECT_EQ(cn->lookup(Complex::zero), Complex::zero);
    EXPECT_EQ(cn->lookup(Complex::one), Complex::one);
    EXPECT_EQ(cn->lookup(1e-14, 0.), Complex::zero);
    EXPECT_EQ(CTEntry::val(cn->lookup(1e-14, 1.).r), 0.);
    EXPECT_EQ(CTEntry::val(cn->lookup(1e-14, 1.).i), 1.);
    EXPECT_EQ(CTEntry::val(cn->lookup(1e-14, -1.).r), 0.);
    EXPECT_EQ(CTEntry::val(cn->lookup(1e-14, -1.).i), -1.);
    EXPECT_EQ(CTEntry::val(cn->lookup(-1., -1.).r), -1.);
    EXPECT_EQ(CTEntry::val(cn->lookup(-1., -1.).i), -1.);
    auto c = cn->lookup(0., -1.);
    std::cout << c << std::endl;
    EXPECT_EQ(CTEntry::val(cn->lookup(c).r), 0.);
    EXPECT_EQ(CTEntry::val(cn->lookup(c).i), -1.);
    c = cn->lookup(0., 1.);
    EXPECT_EQ(CTEntry::val(cn->lookup(c).r), 0.);
    EXPECT_EQ(CTEntry::val(cn->lookup(c).i), 1.);
    c = cn->lookup(0., -0.5);
    std::cout << c << std::endl;
    EXPECT_EQ(CTEntry::val(cn->lookup(c).r), 0.);
    EXPECT_EQ(CTEntry::val(cn->lookup(c).i), -0.5);
    c = cn->lookup(-1., -1.);
    EXPECT_EQ(CTEntry::val(cn->lookup(c).r), -1.);
    EXPECT_EQ(CTEntry::val(cn->lookup(c).i), -1.);
    std::cout << c << std::endl;

    auto e = cn->lookup(1., -1.);
    std::cout << e << std::endl;
    std::cout << ComplexValue{1., 1.} << std::endl;
    std::cout << ComplexValue{1., -1.} << std::endl;
    std::cout << ComplexValue{1., -0.5} << std::endl;
    cn->complexTable.print();
    cn->complexTable.printStatistics();
}

TEST(DDComplexTest, ComplexNumberArithmetic) {
    auto cn = std::make_unique<ComplexNumbers>();
    auto c  = cn->lookup(0., 1.);
    auto d  = ComplexNumbers::conj(c);
    EXPECT_EQ(CTEntry::val(d.r), 0.);
    EXPECT_EQ(CTEntry::val(d.i), -1.);
    c = cn->lookup(-1., -1.);
    d = ComplexNumbers::neg(c);
    EXPECT_EQ(CTEntry::val(d.r), 1.);
    EXPECT_EQ(CTEntry::val(d.i), 1.);
    c = cn->lookup(0.5, 0.5);
    ComplexNumbers::incRef(c);
    d = cn->lookup(-0.5, 0.5);
    ComplexNumbers::incRef(d);
    auto e = cn->getTemporary();
    ComplexNumbers::sub(e, c, d);
    ComplexNumbers::decRef(c);
    ComplexNumbers::decRef(d);
    e = cn->lookup(e);
    EXPECT_EQ(e, Complex::one);
    auto f = cn->getTemporary();
    ComplexNumbers::div(f, Complex::zero, Complex::one);

    dd::ComplexValue zero{0., 0.};
    dd::ComplexValue one{1., 0.};
    EXPECT_EQ(one + zero, one);
}

TEST(DDComplexTest, NearZeroLookup) {
    auto cn = std::make_unique<ComplexNumbers>();
    auto c  = cn->getTemporary(ComplexTable<>::tolerance() / 10., ComplexTable<>::tolerance() / 10.);
    auto d  = cn->lookup(c);
    EXPECT_EQ(d.r, Complex::zero.r);
    EXPECT_EQ(d.i, Complex::zero.i);
}

TEST(DDComplexTest, SortedBuckets) {
    auto     ct  = std::make_unique<ComplexTable<>>();
    const fp num = 0.25;

    const std::array<dd::fp, 7> numbers = {
            num + 2. * ComplexTable<>::tolerance(),
            num - 2. * ComplexTable<>::tolerance(),
            num + 4. * ComplexTable<>::tolerance(),
            num,
            num - 4. * ComplexTable<>::tolerance(),
            num + 6. * ComplexTable<>::tolerance(),
            num + 8. * ComplexTable<>::tolerance()};

    const std::size_t the_bucket = ct->hash(num);

    for (auto const& number: numbers) {
        ASSERT_EQ(the_bucket, ct->hash(number));
        ct->lookup(number);
    }

    CTEntry* p = ct->getTable().at(the_bucket);
    ASSERT_NE(p, nullptr);

    dd::fp      last    = std::numeric_limits<dd::fp>::min();
    std::size_t counter = 0;
    while (p != nullptr) {
        ASSERT_LT(last, p->value);
        p = p->next;
        ++counter;
    }
    ct->printStatistics(std::cout);
    EXPECT_EQ(ct->getStatistics().at("lowerNeighbors"), 0);
    EXPECT_EQ(counter, numbers.size());
}

TEST(DDComplexTest, GarbageCollectSomeInBucket) {
    auto cn = std::make_unique<ComplexNumbers>();
    EXPECT_EQ(cn->garbageCollect(), 0);

    const fp num = 0.25;
    cn->lookup(num, 0.0);

    const fp num2 = num + 2. * ComplexTable<>::tolerance();
    ComplexNumbers::incRef(cn->lookup(num2, 0.0)); // num2 should be placed in same bucket as num

    auto key  = ComplexTable<>::hash(num);
    auto key2 = ComplexTable<>::hash(num2);
    ASSERT_EQ(key, key2);

    auto* p = cn->complexTable.getTable()[key];
    EXPECT_NEAR(p->value, num, ComplexTable<>::tolerance());

    ASSERT_NE(p->next, nullptr);
    EXPECT_NEAR((p->next)->value, num2, ComplexTable<>::tolerance());

    cn->garbageCollect(true); // num should be collected
    EXPECT_NEAR(cn->complexTable.getTable()[key]->value, num2, ComplexTable<>::tolerance());
    EXPECT_EQ(cn->complexTable.getTable()[key]->next, nullptr);
}

TEST(DDComplexTest, LookupInNeighbouringBuckets) {
    std::clog << "Current rounding mode: " << std::numeric_limits<fp>::round_style << "\n";
    auto                  cn      = std::make_unique<ComplexNumbers>();
    constexpr std::size_t NBUCKET = ComplexTable<>::MASK + 1;
    auto                  preHash = [](const fp val) { return val * ComplexTable<>::MASK; };

    // lower border of a bucket
    const fp   numBucketBorder  = (0.25 * ComplexTable<>::MASK - 0.5) / (ComplexTable<>::MASK);
    const auto hashBucketBorder = ComplexTable<>::hash(numBucketBorder);
    std::cout.flush();
    std::clog << "numBucketBorder          = " << std::setprecision(std::numeric_limits<fp>::max_digits10) << numBucketBorder << "\n";
    std::clog << "preHash(numBucketBorder) = " << preHash(numBucketBorder) << "\n";
    std::clog << "hashBucketBorder         = " << hashBucketBorder << "\n"
              << std::flush;
    EXPECT_EQ(hashBucketBorder, NBUCKET / 4);

    // insert a number slightly away from the border
    const fp numAbove = numBucketBorder + 2 * ComplexTable<>::tolerance();
    cn->lookup(numAbove, 0.0);
    auto key = ComplexTable<>::hash(numAbove);
    EXPECT_EQ(key, NBUCKET / 4);

    // insert a number barely in the bucket below
    const fp numBarelyBelow = numBucketBorder - ComplexTable<>::tolerance() / 10;
    cn->lookup(numBarelyBelow, 0.0);
    auto hashBarelyBelow = ComplexTable<>::hash(numBarelyBelow);
    std::cout.flush();
    std::clog << "numBarelyBelow          = " << std::setprecision(std::numeric_limits<fp>::max_digits10) << numBarelyBelow << "\n";
    std::clog << "preHash(numBarelyBelow) = " << preHash(numBarelyBelow) << "\n";
    std::clog << "hashBarelyBelow         = " << hashBarelyBelow << "\n"
              << std::flush;
    EXPECT_EQ(hashBarelyBelow, NBUCKET / 4 - 1);

    // insert another number in the bucket below a bit farther away from the border
    const fp numBelow = numBucketBorder - 2 * ComplexTable<>::tolerance();
    cn->lookup(numBelow, 0.0);
    auto hashBelow = ComplexTable<>::hash(numBelow);
    std::cout.flush();
    std::clog << "numBelow          = " << std::setprecision(std::numeric_limits<fp>::max_digits10) << numBelow << "\n";
    std::clog << "preHash(numBelow) = " << preHash(numBelow) << "\n";
    std::clog << "hashBelow         = " << hashBelow << "\n"
              << std::flush;
    EXPECT_EQ(hashBelow, NBUCKET / 4 - 1);

    // insert border number that is too far away from the number in the bucket, but is close enough to a number in the bucket below
    fp   num4 = numBucketBorder;
    auto c    = cn->lookup(num4, 0.0);
    auto key4 = ComplexTable<>::hash(num4 - ComplexTable<>::tolerance());
    EXPECT_EQ(hashBarelyBelow, key4);
    EXPECT_NEAR(c.r->value, numBarelyBelow, ComplexTable<>::tolerance());

    // insert a number in the higher bucket
    const fp numNextBorder = numBucketBorder + 1.0 / (NBUCKET - 1) + ComplexTable<>::tolerance();
    cn->lookup(numNextBorder, 0.0);
    auto hashNextBorder = ComplexTable<>::hash(numNextBorder);
    std::cout.flush();
    std::clog << "numNextBorder          = " << std::setprecision(std::numeric_limits<fp>::max_digits10) << numNextBorder << "\n";
    std::clog << "preHash(numNextBorder) = " << preHash(numNextBorder) << "\n";
    std::clog << "hashNextBorder         = " << hashNextBorder << "\n"
              << std::flush;
    EXPECT_EQ(hashNextBorder, NBUCKET / 4 + 1);

    // search for a number in the lower bucket that is ultimately close enough to a number in the upper bucket
    fp   num6 = numNextBorder - ComplexTable<>::tolerance() / 10;
    auto d    = cn->lookup(num6, 0.0);
    auto key6 = ComplexTable<>::hash(num6 + ComplexTable<>::tolerance());
    EXPECT_EQ(hashNextBorder, key6);
    EXPECT_NEAR(d.r->value, numNextBorder, ComplexTable<>::tolerance());
}

TEST(DDComplexTest, ComplexValueEquals) {
    ComplexValue a{1.0, 0.0};
    ComplexValue a_tol{1.0 + ComplexTable<>::tolerance() / 10, 0.0};
    ComplexValue b{0.0, 1.0};
    EXPECT_TRUE(a.approximatelyEquals(a_tol));
    EXPECT_FALSE(a.approximatelyEquals(b));
}

TEST(DDComplexTest, LowestFractions) {
    EXPECT_THAT(dd::ComplexValue::getLowestFraction(0.0), ::testing::Pair(0, 1));
    EXPECT_THAT(dd::ComplexValue::getLowestFraction(0.2), ::testing::Pair(1, 5));
    EXPECT_THAT(dd::ComplexValue::getLowestFraction(0.25), ::testing::Pair(1, 4));
    EXPECT_THAT(dd::ComplexValue::getLowestFraction(0.5), ::testing::Pair(1, 2));
    EXPECT_THAT(dd::ComplexValue::getLowestFraction(0.75), ::testing::Pair(3, 4));
    EXPECT_THAT(dd::ComplexValue::getLowestFraction(1.5), ::testing::Pair(3, 2));
    EXPECT_THAT(dd::ComplexValue::getLowestFraction(2.0), ::testing::Pair(2, 1));
    EXPECT_THAT(dd::ComplexValue::getLowestFraction(2047.0 / 2048.0, 1024U), ::testing::Pair(1, 1));
}

TEST(DDComplexTest, NumberPrintingToString) {
    auto cn       = std::make_unique<ComplexNumbers>();
    auto imag     = cn->lookup(0., 1.);
    auto imag_str = imag.toString(false);
    EXPECT_STREQ(imag_str.c_str(), "1i");
    auto imag_str_formatted = imag.toString(true);
    EXPECT_STREQ(imag_str_formatted.c_str(), "+i");

    auto superposition     = cn->lookup(dd::SQRT2_2, dd::SQRT2_2);
    auto superposition_str = superposition.toString(false, 3);
    EXPECT_STREQ(superposition_str.c_str(), "0.707+0.707i");
    auto superposition_str_formatted = superposition.toString(true, 3);
    EXPECT_STREQ(superposition_str_formatted.c_str(), "1/√2(1+i)");
    auto neg_superposition               = cn->lookup(dd::SQRT2_2, -dd::SQRT2_2);
    auto neg_superposition_str_formatted = neg_superposition.toString(true, 3);
    EXPECT_STREQ(neg_superposition_str_formatted.c_str(), "1/√2(1-i)");
}

TEST(DDComplexTest, NumberPrintingFormattedFractions) {
    std::stringstream ss{};

    ComplexValue::printFormatted(ss, 0.0, false);
    EXPECT_STREQ(ss.str().c_str(), "+0");
    ss.str("");
    ComplexValue::printFormatted(ss, -0.0, false);
    EXPECT_STREQ(ss.str().c_str(), "-0");
    ss.str("");

    ComplexValue::printFormatted(ss, 0.0, true);
    EXPECT_STREQ(ss.str().c_str(), "+0i");
    ss.str("");
    ComplexValue::printFormatted(ss, -0.0, true);
    EXPECT_STREQ(ss.str().c_str(), "-0i");
    ss.str("");

    ComplexValue::printFormatted(ss, 0.25, false);
    EXPECT_STREQ(ss.str().c_str(), "1/4");
    ss.str("");
    ComplexValue::printFormatted(ss, 0.25, true);
    EXPECT_STREQ(ss.str().c_str(), "+i/4");
    ss.str("");

    ComplexValue::printFormatted(ss, 0.5, false);
    EXPECT_STREQ(ss.str().c_str(), "1/2");
    ss.str("");
    ComplexValue::printFormatted(ss, 0.5, true);
    EXPECT_STREQ(ss.str().c_str(), "+i/2");
    ss.str("");

    ComplexValue::printFormatted(ss, 0.75, false);
    EXPECT_STREQ(ss.str().c_str(), "3/4");
    ss.str("");
    ComplexValue::printFormatted(ss, 0.75, true);
    EXPECT_STREQ(ss.str().c_str(), "+3i/4");
    ss.str("");

    ComplexValue::printFormatted(ss, 1, false);
    EXPECT_STREQ(ss.str().c_str(), "1");
    ss.str("");
    ComplexValue::printFormatted(ss, 1, true);
    EXPECT_STREQ(ss.str().c_str(), "+i");
    ss.str("");

    ComplexValue::printFormatted(ss, 1.5, false);
    EXPECT_STREQ(ss.str().c_str(), "3/2");
    ss.str("");
    ComplexValue::printFormatted(ss, 1.5, true);
    EXPECT_STREQ(ss.str().c_str(), "+3i/2");
    ss.str("");

    ComplexValue::printFormatted(ss, 2, false);
    EXPECT_STREQ(ss.str().c_str(), "2");
    ss.str("");
    ComplexValue::printFormatted(ss, 2, true);
    EXPECT_STREQ(ss.str().c_str(), "+2i");
    ss.str("");
}

TEST(DDComplexTest, NumberPrintingFormattedFractionsSqrt) {
    std::stringstream ss{};

    ComplexValue::printFormatted(ss, 0.25 * dd::SQRT2_2, false);
    EXPECT_STREQ(ss.str().c_str(), "1/(4√2)");
    ss.str("");
    ComplexValue::printFormatted(ss, 0.25 * dd::SQRT2_2, true);
    EXPECT_STREQ(ss.str().c_str(), "+i/(4√2)");
    ss.str("");

    ComplexValue::printFormatted(ss, 0.5 * dd::SQRT2_2, false);
    EXPECT_STREQ(ss.str().c_str(), "1/(2√2)");
    ss.str("");
    ComplexValue::printFormatted(ss, 0.5 * dd::SQRT2_2, true);
    EXPECT_STREQ(ss.str().c_str(), "+i/(2√2)");
    ss.str("");

    ComplexValue::printFormatted(ss, 0.75 * dd::SQRT2_2, false);
    EXPECT_STREQ(ss.str().c_str(), "3/(4√2)");
    ss.str("");
    ComplexValue::printFormatted(ss, 0.75 * dd::SQRT2_2, true);
    EXPECT_STREQ(ss.str().c_str(), "+3i/(4√2)");
    ss.str("");

    ComplexValue::printFormatted(ss, dd::SQRT2_2, false);
    EXPECT_STREQ(ss.str().c_str(), "1/√2");
    ss.str("");
    ComplexValue::printFormatted(ss, dd::SQRT2_2, true);
    EXPECT_STREQ(ss.str().c_str(), "+i/√2");
    ss.str("");

    ComplexValue::printFormatted(ss, 1.5 * dd::SQRT2_2, false);
    EXPECT_STREQ(ss.str().c_str(), "3/(2√2)");
    ss.str("");
    ComplexValue::printFormatted(ss, 1.5 * dd::SQRT2_2, true);
    EXPECT_STREQ(ss.str().c_str(), "+3i/(2√2)");
    ss.str("");

    ComplexValue::printFormatted(ss, 2 * dd::SQRT2_2, false);
    EXPECT_STREQ(ss.str().c_str(), "2/√2");
    ss.str("");
    ComplexValue::printFormatted(ss, 2 * dd::SQRT2_2, true);
    EXPECT_STREQ(ss.str().c_str(), "+2i/√2");
    ss.str("");
}

TEST(DDComplexTest, NumberPrintingFormattedFractionsPi) {
    std::stringstream ss{};

    ComplexValue::printFormatted(ss, 0.25 * dd::PI, false);
    EXPECT_STREQ(ss.str().c_str(), "π/4");
    ss.str("");
    ComplexValue::printFormatted(ss, 0.25 * dd::PI, true);
    EXPECT_STREQ(ss.str().c_str(), "+πi/4");
    ss.str("");

    ComplexValue::printFormatted(ss, 0.5 * dd::PI, false);
    EXPECT_STREQ(ss.str().c_str(), "π/2");
    ss.str("");
    ComplexValue::printFormatted(ss, 0.5 * dd::PI, true);
    EXPECT_STREQ(ss.str().c_str(), "+πi/2");
    ss.str("");

    ComplexValue::printFormatted(ss, 0.75 * dd::PI, false);
    EXPECT_STREQ(ss.str().c_str(), "3π/4");
    ss.str("");
    ComplexValue::printFormatted(ss, 0.75 * dd::PI, true);
    EXPECT_STREQ(ss.str().c_str(), "+3πi/4");
    ss.str("");

    ComplexValue::printFormatted(ss, dd::PI, false);
    EXPECT_STREQ(ss.str().c_str(), "π");
    ss.str("");
    ComplexValue::printFormatted(ss, dd::PI, true);
    EXPECT_STREQ(ss.str().c_str(), "+πi");
    ss.str("");

    ComplexValue::printFormatted(ss, 1.5 * dd::PI, false);
    EXPECT_STREQ(ss.str().c_str(), "3π/2");
    ss.str("");
    ComplexValue::printFormatted(ss, 1.5 * dd::PI, true);
    EXPECT_STREQ(ss.str().c_str(), "+3πi/2");
    ss.str("");

    ComplexValue::printFormatted(ss, 2 * dd::PI, false);
    EXPECT_STREQ(ss.str().c_str(), "2π");
    ss.str("");
    ComplexValue::printFormatted(ss, 2 * dd::PI, true);
    EXPECT_STREQ(ss.str().c_str(), "+2πi");
    ss.str("");
}

TEST(DDComplexTest, NumberPrintingFormattedFloating) {
    std::stringstream ss{};
    ComplexValue::printFormatted(ss, 0.1234, false);
    EXPECT_STREQ(ss.str().c_str(), "0.1234");
    ss.str("");
    ComplexValue::printFormatted(ss, 0.1234, true);
    EXPECT_STREQ(ss.str().c_str(), "+0.1234i");
    ss.str("");
}

TEST(DDComplexTest, MaxRefCountReached) {
    auto       cn  = std::make_unique<ComplexNumbers>();
    auto       c   = cn->lookup(SQRT2_2 / 2, SQRT2_2 / 2);
    const auto max = std::numeric_limits<decltype(c.r->refCount)>::max();
    c.r->refCount  = max - 1;

    std::cout.flush();
    std::clog << "Heads up: The following three MAXREFCNT warnings are part of a passing test.\n";
    CN::incRef(c);
    CN::incRef(c);
    std::clog.flush();
    EXPECT_EQ(c.r->refCount, max);
    EXPECT_EQ(c.i->refCount, max);
}

TEST(DDComplexTest, NegativeRefCountReached) {
    auto cn = std::make_unique<ComplexNumbers>();
    auto c  = cn->lookup(SQRT2_2 / 2, SQRT2_2 / 2);

    ASSERT_THROW(CN::decRef(c), std::runtime_error);
}

TEST(DDComplexTest, ComplexTableAllocation) {
    auto cn     = std::make_unique<ComplexNumbers>();
    auto allocs = cn->complexTable.getAllocations();
    std::cout << allocs << std::endl;
    std::vector<ComplexTable<>::Entry*> nums{allocs};
    // get all the numbers that are pre-allocated
    for (auto i = 0U; i < allocs; ++i) {
        nums[i] = cn->complexTable.getEntry();
    }

    // trigger new allocation
    [[maybe_unused]] auto num = cn->complexTable.getEntry();
    EXPECT_EQ(cn->complexTable.getAllocations(), (1. + cn->complexTable.getGrowthFactor()) * allocs);

    // clearing the complex table should reduce the allocated size to the original size
    cn->complexTable.clear();
    EXPECT_EQ(cn->complexTable.getAllocations(), allocs);

    EXPECT_TRUE(cn->complexTable.availableEmpty());
    // obtain entry
    auto entry = cn->complexTable.getEntry();
    // immediately return entry
    cn->complexTable.returnEntry(entry);
    EXPECT_FALSE(cn->complexTable.availableEmpty());
    // obtain the same entry again, but this time from the available stack
    auto entry2 = cn->complexTable.getEntry();
    EXPECT_EQ(entry, entry2);
}

TEST(DDComplexTest, ComplexCacheAllocation) {
    auto cn     = std::make_unique<ComplexNumbers>();
    auto allocs = cn->complexCache.getAllocations();
    std::cout << allocs << std::endl;
    std::vector<Complex> cnums{allocs};
    // get all the cached complex numbers that are pre-allocated
    for (auto i = 0U; i < allocs; i += 2) {
        cnums[i % 2] = cn->getCached();
    }

    // trigger new allocation for obtaining a complex from cache
    [[maybe_unused]] auto cnum = cn->getCached();
    EXPECT_EQ(cn->complexCache.getAllocations(), (1. + cn->complexCache.getGrowthFactor()) * allocs);

    // clearing the cache should reduce the allocated size to the original size
    cn->complexCache.clear();
    EXPECT_EQ(cn->complexCache.getAllocations(), allocs);

    // get all the cached complex numbers again
    for (auto i = 0U; i < allocs; i += 2) {
        cnums[i % 2] = cn->getCached();
    }

    // trigger new allocation for obtaining a temporary from cache
    [[maybe_unused]] auto cnumtmp = cn->getTemporary();
    EXPECT_EQ(cn->complexCache.getAllocations(), (1. + cn->complexCache.getGrowthFactor()) * allocs);

    // clearing the unique table should reduce the allocated size to the original size
    cn->complexCache.clear();
    EXPECT_EQ(cn->complexCache.getAllocations(), allocs);
}
