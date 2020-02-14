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

