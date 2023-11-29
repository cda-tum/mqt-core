#include "algorithms/GoogleRandomCircuitSampling.hpp"
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
  std::cout << qcBris << "\n";

  auto qcInst =
      qc::GoogleRandomCircuitSampling("./circuits/grcs/inst_4x4_80_9_v2.txt");
  qcInst.printStatistics(std::cout);
  std::cout << qcInst << "\n";
}

TEST_F(GRCS, simulate) {
  auto qcBris =
      qc::GoogleRandomCircuitSampling("./circuits/grcs/bris_4_40_9_v2.txt");

  auto dd = std::make_unique<dd::Package<>>(qcBris.getNqubits());
  auto in = dd->makeZeroState(qcBris.getNqubits());
  const std::optional<std::size_t> ncycles = 4;
  ASSERT_NO_THROW({ simulate(&qcBris, in, dd, ncycles); });
  std::cout << qcBris << "\n";
  qcBris.printStatistics(std::cout);
}
