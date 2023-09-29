#include "dd/StateGenerator.hpp"
#include "dd/Vector.hpp"

#include <gtest/gtest.h>
#include <memory>

namespace dd {

class DDStateGeneration : public testing::Test {
protected:
  std::size_t seed = 12345U;
  StateGenerator stateGeneration{seed};
  std::size_t nTrials = 10U;
};

TEST_F(DDStateGeneration, RandomSingleQubitState) {
  for (std::size_t i = 0U; i < nTrials; ++i) {
    const auto state = stateGeneration.generateRandomVectorDD(1U, {1U});
    ASSERT_EQ(state.w, Complex::one);
    ASSERT_NE(state.p, nullptr);
    ASSERT_EQ(state.p->v, 0U);
    ASSERT_TRUE(state.p->e[0].isTerminal());
    ASSERT_TRUE(state.p->e[1].isTerminal());

    const auto vec = getVectorFromDD(state);
    ASSERT_EQ(vec.size(), 2U);
    const auto norm = getVectorNorm(vec);
    ASSERT_NEAR(norm, 1.0, 1e-10);
  }
}

TEST_F(DDStateGeneration, RandomTwoQubitProductState) {
  for (std::size_t i = 0U; i < nTrials; ++i) {
    const auto state = stateGeneration.generateRandomVectorDD(2U, {1U, 1U});
    ASSERT_EQ(state.w, Complex::one);
    ASSERT_NE(state.p, nullptr);
    ASSERT_EQ(state.p->v, 1U);
    const auto& s0 = state.p->e[0].p;
    const auto& s1 = state.p->e[1].p;
    ASSERT_EQ(s0, s1);
    ASSERT_NE(s0, nullptr);
    ASSERT_EQ(s0->v, 0U);
    ASSERT_TRUE(s0->e[0].isTerminal());
    ASSERT_TRUE(s0->e[1].isTerminal());

    const auto vec = getVectorFromDD(state);
    ASSERT_EQ(vec.size(), 4U);
    const auto norm = getVectorNorm(vec);
    ASSERT_NEAR(norm, 1.0, 1e-10);
  }
}

TEST_F(DDStateGeneration, RandomTwoQubitEntangledState) {
  for (std::size_t i = 0U; i < nTrials; ++i) {
    const auto state = stateGeneration.generateRandomVectorDD(2U, {2U, 1U});
    ASSERT_EQ(state.w, Complex::one);
    ASSERT_NE(state.p, nullptr);
    ASSERT_EQ(state.p->v, 1U);
    const auto& s0 = state.p->e[0].p;
    const auto& s1 = state.p->e[1].p;
    ASSERT_NE(s0, s1);
    ASSERT_NE(s0, nullptr);
    ASSERT_EQ(s0->v, 0U);
    ASSERT_TRUE(s0->e[0].isTerminal());
    ASSERT_TRUE(s0->e[1].isTerminal());
    ASSERT_NE(s1, nullptr);
    ASSERT_EQ(s1->v, 0U);
    ASSERT_TRUE(s1->e[0].isTerminal());
    ASSERT_TRUE(s1->e[1].isTerminal());

    const auto vec = getVectorFromDD(state);
    ASSERT_EQ(vec.size(), 4U);
    const auto norm = getVectorNorm(vec);
    ASSERT_NEAR(norm, 1.0, 1e-10);
  }
}

TEST_F(DDStateGeneration, RandomThreeQubitStates) {
  for (std::size_t i = 0U; i < nTrials; ++i) {
    const auto state = stateGeneration.generateRandomVectorDD(3U, {3U, 2U, 1U});
    ASSERT_EQ(state.w, Complex::one);
    ASSERT_NE(state.p, nullptr);
    ASSERT_EQ(state.p->v, 2U);
    const auto vec = getVectorFromDD(state);
    ASSERT_EQ(vec.size(), 8U);
    const auto norm = getVectorNorm(vec);
    ASSERT_NEAR(norm, 1.0, 1e-10);
  }
}

} // namespace dd
