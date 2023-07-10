#include "dd/ComplexNumbers.hpp"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <limits>
#include <memory>

using namespace dd;
using CN = ComplexNumbers;

TEST(DDComplexTest, TrivialTest) {
  auto cn = std::make_unique<ComplexNumbers>();
  const auto beforeCount = cn->cacheCount();

  auto a = cn->getCached(2, -3);
  auto b = cn->getCached(3, 2);

  auto r0 = cn->getCached(12, -5);
  auto r1 = cn->mulCached(a, b);
  auto r2 = cn->divCached(r0, r1);

  const auto betweenCount = cn->cacheCount();
  ASSERT_LE(beforeCount, betweenCount);
  cn->returnToCache(a);
  cn->returnToCache(b);
  cn->returnToCache(r0);
  cn->returnToCache(r1);
  cn->returnToCache(r2);
  cn->garbageCollect(true);
  const auto endCount = cn->cacheCount();
  ASSERT_EQ(beforeCount, endCount);
}

TEST(DDComplexTest, ComplexNumberCreation) {
  auto cn = std::make_unique<ComplexNumbers>();
  EXPECT_EQ(cn->lookup(Complex::zero), Complex::zero);
  EXPECT_EQ(cn->lookup(Complex::one), Complex::one);
  EXPECT_EQ(cn->lookup(1e-16, 0.), Complex::zero);
  EXPECT_EQ(CTEntry::val(cn->lookup(1e-16, 1.).r), 0.);
  EXPECT_EQ(CTEntry::val(cn->lookup(1e-16, 1.).i), 1.);
  EXPECT_EQ(CTEntry::val(cn->lookup(1e-16, -1.).r), 0.);
  EXPECT_EQ(CTEntry::val(cn->lookup(1e-16, -1.).i), -1.);
  EXPECT_EQ(CTEntry::val(cn->lookup(-1., -1.).r), -1.);
  EXPECT_EQ(CTEntry::val(cn->lookup(-1., -1.).i), -1.);
  auto c = cn->lookup(0., -1.);
  std::cout << c << "\n";
  EXPECT_EQ(CTEntry::val(cn->lookup(c).r), 0.);
  EXPECT_EQ(CTEntry::val(cn->lookup(c).i), -1.);
  c = cn->lookup(0., 1.);
  EXPECT_EQ(CTEntry::val(cn->lookup(c).r), 0.);
  EXPECT_EQ(CTEntry::val(cn->lookup(c).i), 1.);
  c = cn->lookup(0., -0.5);
  std::cout << c << "\n";
  EXPECT_EQ(CTEntry::val(cn->lookup(c).r), 0.);
  EXPECT_EQ(CTEntry::val(cn->lookup(c).i), -0.5);
  c = cn->lookup(-1., -1.);
  EXPECT_EQ(CTEntry::val(cn->lookup(c).r), -1.);
  EXPECT_EQ(CTEntry::val(cn->lookup(c).i), -1.);
  std::cout << c << "\n";

  auto e = cn->lookup(1., -1.);
  std::cout << e << "\n";
  std::cout << ComplexValue{1., 1.} << "\n";
  std::cout << ComplexValue{1., -1.} << "\n";
  std::cout << ComplexValue{1., -0.5} << "\n";
  cn->complexTable.print();
  cn->complexTable.printStatistics();
}

TEST(DDComplexTest, ComplexNumberArithmetic) {
  auto cn = std::make_unique<ComplexNumbers>();
  auto c = cn->lookup(0., 1.);
  auto d = ComplexNumbers::conj(c);
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

  const dd::ComplexValue zero{0., 0.};
  const dd::ComplexValue one{1., 0.};
  EXPECT_EQ(one + zero, one);
}

TEST(DDComplexTest, NearZeroLookup) {
  auto cn = std::make_unique<ComplexNumbers>();
  auto c = cn->getTemporary(ComplexTable::tolerance() / 10.,
                            ComplexTable::tolerance() / 10.);
  auto d = cn->lookup(c);
  EXPECT_EQ(d.r, Complex::zero.r);
  EXPECT_EQ(d.i, Complex::zero.i);
}

TEST(DDComplexTest, SortedBuckets) {
  auto ct = std::make_unique<ComplexTable>();
  const fp num = 0.25;

  const std::array<dd::fp, 7> numbers = {num + 2. * ComplexTable::tolerance(),
                                         num - 2. * ComplexTable::tolerance(),
                                         num + 4. * ComplexTable::tolerance(),
                                         num,
                                         num - 4. * ComplexTable::tolerance(),
                                         num + 6. * ComplexTable::tolerance(),
                                         num + 8. * ComplexTable::tolerance()};

  const auto theBucket = static_cast<std::size_t>(ct->hash(num));

  for (auto const& number : numbers) {
    ASSERT_EQ(theBucket, ct->hash(number));
    const auto* entry = ct->lookup(number);
    ASSERT_NE(entry, nullptr);
  }

  CTEntry* p = ct->getTable().at(theBucket);
  ASSERT_NE(p, nullptr);

  const dd::fp last = std::numeric_limits<dd::fp>::min();
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
  const auto lookup = cn->lookup(num, 0.0);
  ASSERT_NE(lookup.r, nullptr);
  ASSERT_NE(lookup.i, nullptr);

  const fp num2 = num + 2. * ComplexTable::tolerance();
  const auto lookup2 = cn->lookup(num2, 0.0);
  ASSERT_NE(lookup2.r, nullptr);
  ASSERT_NE(lookup2.i, nullptr);
  ComplexNumbers::incRef(lookup2);

  // num2 should be placed in same bucket as num
  auto key = ComplexTable::hash(num);
  auto key2 = ComplexTable::hash(num2);
  ASSERT_EQ(key, key2);

  const auto& table = cn->complexTable.getTable();
  const auto* p = table[static_cast<std::size_t>(key)];
  EXPECT_NEAR(p->value, num, ComplexTable::tolerance());

  ASSERT_NE(p->next, nullptr);
  EXPECT_NEAR((p->next)->value, num2, ComplexTable::tolerance());

  cn->garbageCollect(true); // num should be collected
  const auto* q = table[static_cast<std::size_t>(key)];
  ASSERT_NE(q, nullptr);
  EXPECT_NEAR(q->value, num2, ComplexTable::tolerance());
  EXPECT_EQ(q->next, nullptr);
}

TEST(DDComplexTest, LookupInNeighbouringBuckets) {
  std::clog << "Current rounding mode: " << std::numeric_limits<fp>::round_style
            << "\n";
  auto cn = std::make_unique<ComplexNumbers>();
  constexpr std::size_t nbucket = ComplexTable::MASK + 1;
  auto preHash = [](const fp val) { return val * ComplexTable::MASK; };

  // lower border of a bucket
  const fp numBucketBorder =
      (0.25 * ComplexTable::MASK - 0.5) / (ComplexTable::MASK);
  const auto hashBucketBorder = ComplexTable::hash(numBucketBorder);
  std::cout.flush();
  std::clog << "numBucketBorder          = "
            << std::setprecision(std::numeric_limits<fp>::max_digits10)
            << numBucketBorder << "\n";
  std::clog << "preHash(numBucketBorder) = " << preHash(numBucketBorder)
            << "\n";
  std::clog << "hashBucketBorder         = " << hashBucketBorder << "\n"
            << std::flush;
  EXPECT_EQ(hashBucketBorder, nbucket / 4);

  // insert a number slightly away from the border
  const fp numAbove = numBucketBorder + 2 * ComplexTable::tolerance();
  const auto lookupAbove = cn->lookup(numAbove, 0.0);
  ASSERT_NE(lookupAbove.r, nullptr);
  ASSERT_NE(lookupAbove.i, nullptr);
  const auto key = ComplexTable::hash(numAbove);
  EXPECT_EQ(key, nbucket / 4);

  // insert a number barely in the bucket below
  const fp numBarelyBelow = numBucketBorder - ComplexTable::tolerance() / 10;
  const auto lookupBarelyBelow = cn->lookup(numBarelyBelow, 0.0);
  ASSERT_NE(lookupBarelyBelow.r, nullptr);
  ASSERT_NE(lookupBarelyBelow.i, nullptr);
  const auto hashBarelyBelow = ComplexTable::hash(numBarelyBelow);
  std::clog << "numBarelyBelow          = "
            << std::setprecision(std::numeric_limits<fp>::max_digits10)
            << numBarelyBelow << "\n";
  std::clog << "preHash(numBarelyBelow) = " << preHash(numBarelyBelow) << "\n";
  std::clog << "hashBarelyBelow         = " << hashBarelyBelow << "\n";
  EXPECT_EQ(hashBarelyBelow, nbucket / 4 - 1);

  // insert another number in the bucket below a bit farther away from the
  // border
  const fp numBelow = numBucketBorder - 2 * ComplexTable::tolerance();
  const auto lookupBelow = cn->lookup(numBelow, 0.0);
  ASSERT_NE(lookupBelow.r, nullptr);
  ASSERT_NE(lookupBelow.i, nullptr);
  const auto hashBelow = ComplexTable::hash(numBelow);
  std::clog << "numBelow          = "
            << std::setprecision(std::numeric_limits<fp>::max_digits10)
            << numBelow << "\n";
  std::clog << "preHash(numBelow) = " << preHash(numBelow) << "\n";
  std::clog << "hashBelow         = " << hashBelow << "\n";
  EXPECT_EQ(hashBelow, nbucket / 4 - 1);

  // insert border number that is too far away from the number in the bucket,
  // but is close enough to a number in the bucket below
  const fp num4 = numBucketBorder;
  const auto c = cn->lookup(num4, 0.0);
  const auto key4 = ComplexTable::hash(num4 - ComplexTable::tolerance());
  EXPECT_EQ(hashBarelyBelow, key4);
  EXPECT_NEAR(c.r->value, numBarelyBelow, ComplexTable::tolerance());

  // insert a number in the higher bucket
  const fp numNextBorder =
      numBucketBorder + 1.0 / (nbucket - 1) + ComplexTable::tolerance();
  const auto lookupNextBorder = cn->lookup(numNextBorder, 0.0);
  ASSERT_NE(lookupNextBorder.r, nullptr);
  ASSERT_NE(lookupNextBorder.i, nullptr);
  const auto hashNextBorder = ComplexTable::hash(numNextBorder);
  std::clog << "numNextBorder          = "
            << std::setprecision(std::numeric_limits<fp>::max_digits10)
            << numNextBorder << "\n";
  std::clog << "preHash(numNextBorder) = " << preHash(numNextBorder) << "\n";
  std::clog << "hashNextBorder         = " << hashNextBorder << "\n";
  EXPECT_EQ(hashNextBorder, nbucket / 4 + 1);

  // search for a number in the lower bucket that is ultimately close enough to
  // a number in the upper bucket
  const fp num6 = numNextBorder - ComplexTable::tolerance() / 10;
  const auto d = cn->lookup(num6, 0.0);
  const auto key6 = ComplexTable::hash(num6 + ComplexTable::tolerance());
  EXPECT_EQ(hashNextBorder, key6);
  EXPECT_NEAR(d.r->value, numNextBorder, ComplexTable::tolerance());
}

TEST(DDComplexTest, ComplexValueEquals) {
  const ComplexValue a{1.0, 0.0};
  const ComplexValue aTol{1.0 + ComplexTable::tolerance() / 10, 0.0};
  const ComplexValue b{0.0, 1.0};
  EXPECT_TRUE(a.approximatelyEquals(aTol));
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
  EXPECT_THAT(dd::ComplexValue::getLowestFraction(2047.0 / 2048.0, 1024U),
              ::testing::Pair(1, 1));
}

TEST(DDComplexTest, NumberPrintingToString) {
  auto cn = std::make_unique<ComplexNumbers>();
  auto imag = cn->lookup(0., 1.);
  auto imagStr = imag.toString(false);
  EXPECT_STREQ(imagStr.c_str(), "1i");
  auto imagStrFormatted = imag.toString(true);
  EXPECT_STREQ(imagStrFormatted.c_str(), "+i");

  auto superposition = cn->lookup(dd::SQRT2_2, dd::SQRT2_2);
  auto superpositionStr = superposition.toString(false, 3);
  EXPECT_STREQ(superpositionStr.c_str(), "0.707+0.707i");
  auto superpositionStrFormatted = superposition.toString(true, 3);
  EXPECT_STREQ(superpositionStrFormatted.c_str(), "1/√2(1+i)");
  auto negSuperposition = cn->lookup(dd::SQRT2_2, -dd::SQRT2_2);
  auto negSuperpositionStrFormatted = negSuperposition.toString(true, 3);
  EXPECT_STREQ(negSuperpositionStrFormatted.c_str(), "1/√2(1-i)");
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
  auto cn = std::make_unique<ComplexNumbers>();
  auto c = cn->lookup(SQRT2_2 / 2, SQRT2_2 / 2);
  const auto max = std::numeric_limits<decltype(c.r->refCount)>::max();
  c.r->refCount = max - 1;

  std::cout.flush();
  std::clog << "Heads up: The following three MAXREFCNT warnings are part of a "
               "passing test.\n";
  CN::incRef(c);
  CN::incRef(c);
  std::clog.flush();
  EXPECT_EQ(c.r->refCount, max);
  EXPECT_EQ(c.i->refCount, max);
}

TEST(DDComplexTest, ComplexTableAllocation) {
  auto cn = std::make_unique<ComplexNumbers>();
  auto allocs = cn->complexTable.getAllocations();
  std::cout << allocs << "\n";
  std::vector<ComplexTable::Entry*> nums{allocs};
  // get all the numbers that are pre-allocated
  for (auto i = 0U; i < allocs; ++i) {
    nums[i] = cn->complexTable.getEntry();
  }

  // trigger new allocation
  const auto* num = cn->complexTable.getEntry();
  ASSERT_NE(num, nullptr);
  EXPECT_EQ(cn->complexTable.getAllocations(),
            (1. + static_cast<fp>(cn->complexTable.getGrowthFactor())) *
                static_cast<fp>(allocs));

  // clearing the complex table should reduce the allocated size to the original
  // size
  cn->complexTable.clear();
  EXPECT_EQ(cn->complexTable.getAllocations(), allocs);

  EXPECT_TRUE(cn->complexTable.availableEmpty());
  // obtain entry
  auto* entry = cn->complexTable.getEntry();
  // immediately return entry
  cn->complexTable.returnEntry(entry);
  EXPECT_FALSE(cn->complexTable.availableEmpty());
  // obtain the same entry again, but this time from the available stack
  auto* entry2 = cn->complexTable.getEntry();
  EXPECT_EQ(entry, entry2);
}

TEST(DDComplexTest, ComplexCacheAllocation) {
  auto cn = std::make_unique<ComplexNumbers>();
  auto allocs = cn->complexCache.getAllocations();
  std::cout << allocs << "\n";
  std::vector<Complex> cnums{allocs};
  // get all the cached complex numbers that are pre-allocated
  for (auto i = 0U; i < allocs; i += 2) {
    cnums[i % 2] = cn->getCached();
  }

  // trigger new allocation for obtaining a complex from cache
  const auto cnum = cn->getCached();
  ASSERT_NE(cnum.r, nullptr);
  ASSERT_NE(cnum.i, nullptr);
  EXPECT_EQ(cn->complexCache.getAllocations(),
            (1. + static_cast<fp>(cn->complexCache.getGrowthFactor())) *
                static_cast<fp>(allocs));

  // clearing the cache should reduce the allocated size to the original size
  cn->complexCache.clear();
  EXPECT_EQ(cn->complexCache.getAllocations(), allocs);

  // get all the cached complex numbers again
  for (auto i = 0U; i < allocs; i += 2) {
    cnums[i % 2] = cn->getCached();
  }

  // trigger new allocation for obtaining a temporary from cache
  const auto tmp = cn->getTemporary();
  ASSERT_NE(tmp.r, nullptr);
  ASSERT_NE(tmp.i, nullptr);
  EXPECT_EQ(cn->complexCache.getAllocations(),
            (1. + static_cast<fp>(cn->complexCache.getGrowthFactor())) *
                static_cast<fp>(allocs));

  // clearing the unique table should reduce the allocated size to the original
  // size
  cn->complexCache.clear();
  EXPECT_EQ(cn->complexCache.getAllocations(), allocs);
}

TEST(DDComplexTest, DoubleHitInFindOrInsert) {
  auto cn = std::make_unique<ComplexNumbers>();

  // insert a number somewhere in a bucket
  const fp num1 = 0.5;
  auto* tnum1 = cn->complexTable.lookup(num1);
  EXPECT_EQ(tnum1->value, num1);

  // insert a second number that is farther away than the tolerance, but closer
  // than twice the tolerance
  const fp num2 = num1 + 2.1 * dd::ComplexTable::tolerance();
  auto* tnum2 = cn->complexTable.lookup(num2);
  EXPECT_EQ(tnum2->value, num2);

  // insert a third number that is close to both previously inserted numbers,
  // but closer to the second
  const fp num3 = num1 + 2.2 * dd::ComplexTable::tolerance();
  auto* tnum3 = cn->complexTable.lookup(num3);
  EXPECT_EQ(tnum3->value, num2);
}

TEST(DDComplexTest, DoubleHitAcrossBuckets) {
  auto cn = std::make_unique<ComplexNumbers>();
  std::cout << std::setprecision(std::numeric_limits<dd::fp>::max_digits10);

  // insert a number at a lower bucket border
  const fp num1 =
      8191.5 / (static_cast<dd::fp>(cn->complexTable.getTable().size()) - 1);
  auto* tnum1 = cn->complexTable.lookup(num1);
  EXPECT_EQ(tnum1->value, num1);

  // insert a second number that is farther away than the tolerance towards the
  // lower bucket, but closer than twice the tolerance
  const fp num2 = num1 - 1.5 * dd::ComplexTable::tolerance();
  auto* tnum2 = cn->complexTable.lookup(num2);
  EXPECT_EQ(tnum2->value, num2);

  // insert a third number that is close to both previously inserted numbers,
  // but closer to the second
  const fp num3 = num1 - 0.9 * dd::ComplexTable::tolerance();
  auto* tnum3 = cn->complexTable.lookup(num3);
  EXPECT_EQ(tnum3->value, num2);

  // insert a third number that is close to both previously inserted numbers,
  // but closer to the first
  const fp num4 = num1 - 0.6 * dd::ComplexTable::tolerance();
  auto* tnum4 = cn->complexTable.lookup(num4);
  EXPECT_EQ(tnum4->value, num1);
}
