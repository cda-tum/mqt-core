#include "dd/Benchmark.hpp"

#include <gtest/gtest.h>

class BuildTaskTest : public ::testing::Test {

};

TEST_F(BuildTaskTest, TrivialTest) {
  auto qc = qc::QuantumComputation(2);
  qc.x(0);
  qc.x(1);
  benchmarkBuildFunctionality(qc);
}

class SimulateTest: public ::testing::Test {

};

TEST_F(SimulateTest, TrivialTest) {
  auto qc = qc::QuantumComputation(2);
  qc.x(0);
  qc.x(1);
  benchmarkSimulate(qc);
}
