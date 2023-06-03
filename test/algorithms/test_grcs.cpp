/*
 * This file is part of MQT QFR library which is released under the MIT license.
 * See file README.md or go to https://www.cda.cit.tum.de/research/quantum/ for
 * more information.
 */

#include "algorithms/GoogleRandomCircuitSampling.hpp"
#include "dd/FunctionalityConstruction.hpp"
#include "dd/Simulation.hpp"

#include "gtest/gtest.h"

class GRCS : public testing::Test {
protected:
  void TearDown() override {}
  void SetUp() override {}
};

TEST_F(GRCS, import) {
  auto qcBris =
      qc::GoogleRandomCircuitSampling("./circuits/grcs/bris_4_40_9_v2.txt");
  qcBris.printStatistics(std::cout);
  std::cout << qcBris << std::endl;

  auto qcInst =
      qc::GoogleRandomCircuitSampling("./circuits/grcs/inst_4x4_80_9_v2.txt");
  qcInst.printStatistics(std::cout);
  std::cout << qcInst << std::endl;
}

TEST_F(GRCS, simulate) {
  auto qcBris =
      qc::GoogleRandomCircuitSampling("./circuits/grcs/bris_4_40_9_v2.txt");

  auto dd = std::make_unique<dd::Package<>>(qcBris.getNqubits());
  auto in = dd->makeZeroState(static_cast<dd::QubitCount>(qcBris.getNqubits()));
  const std::optional<std::size_t> ncycles = 4;
  ASSERT_NO_THROW({ simulate(&qcBris, in, dd, ncycles); });
  std::cout << qcBris << std::endl;
  qcBris.printStatistics(std::cout);
}

TEST_F(GRCS, buildFunctionality) {
  auto qcBris =
      qc::GoogleRandomCircuitSampling("./circuits/grcs/bris_4_40_9_v2.txt");

  auto dd = std::make_unique<dd::Package<>>(qcBris.getNqubits());
  ASSERT_NO_THROW({ buildFunctionality(&qcBris, dd, 4); });
  std::cout << qcBris << std::endl;
  qcBris.printStatistics(std::cout);
}
