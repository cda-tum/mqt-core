#include "Definitions.hpp"
#include "dd/DDDefinitions.hpp"
#include "dd/DDpackageConfig.hpp"
#include "dd/Export.hpp"
#include "dd/GateMatrixDefinitions.hpp"
#include "dd/MemoryManager.hpp"
#include "dd/Node.hpp"
#include "dd/Package.hpp"
#include "dd/RealNumber.hpp"
#include "dd/statistics/PackageStatistics.hpp"
#include "operations/Control.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

using namespace qc::literals;

TEST(DDPackageTest, RequestInvalidPackageSize) {
  EXPECT_THROW(auto dd = std::make_unique<dd::Package<>>(
                   dd::Package<>::MAX_POSSIBLE_QUBITS + 2),
               std::invalid_argument);
}

TEST(DDPackageTest, TrivialTest) {
  auto dd = std::make_unique<dd::Package<>>(2);
  EXPECT_EQ(dd->qubits(), 2);

  auto xGate = dd->makeGateDD(dd::X_MAT, 0);
  auto hGate = dd->makeGateDD(dd::H_MAT, 0);

  ASSERT_EQ(hGate.getValueByPath(1, "0"), dd::SQRT2_2);

  auto zeroState = dd->makeZeroState(1);
  auto hState = dd->multiply(hGate, zeroState);
  auto oneState = dd->multiply(xGate, zeroState);

  ASSERT_EQ(dd->fidelity(zeroState, oneState), 0.0);
  // repeat the same calculation - triggering compute table hit
  ASSERT_EQ(dd->fidelity(zeroState, oneState), 0.0);
  ASSERT_NEAR(dd->fidelity(zeroState, hState), 0.5, dd::RealNumber::eps);
  ASSERT_NEAR(dd->fidelity(oneState, hState), 0.5, dd::RealNumber::eps);
}

TEST(DDPackageTest, BellState) {
  auto dd = std::make_unique<dd::Package<>>(2);

  auto hGate = dd->makeGateDD(dd::H_MAT, 1);
  auto cxGate = dd->makeGateDD(dd::X_MAT, 1_pc, 0);
  auto zeroState = dd->makeZeroState(2);

  auto bellState = dd->multiply(dd->multiply(cxGate, hGate), zeroState);
  bellState.printVector();

  // repeated calculation is practically for free
  auto bellState2 = dd->multiply(dd->multiply(cxGate, hGate), zeroState);
  EXPECT_EQ(bellState, bellState2);

  ASSERT_EQ(bellState.getValueByPath(dd->qubits(), "00"), dd::SQRT2_2);
  ASSERT_EQ(bellState.getValueByPath(dd->qubits(), "01"), 0.);
  ASSERT_EQ(bellState.getValueByPath(dd->qubits(), "10"), 0.);
  ASSERT_EQ(bellState.getValueByPath(dd->qubits(), "11"), dd::SQRT2_2);

  ASSERT_EQ(bellState.getValueByIndex(0), dd::SQRT2_2);
  ASSERT_EQ(bellState.getValueByIndex(1), 0.);
  ASSERT_EQ(bellState.getValueByIndex(2), 0.);
  ASSERT_EQ(bellState.getValueByIndex(3), dd::SQRT2_2);

  auto goalState =
      dd::CVec{{dd::SQRT2_2, 0.}, {0., 0.}, {0., 0.}, {dd::SQRT2_2, 0.}};
  ASSERT_EQ(bellState.getVector(), goalState);

  ASSERT_DOUBLE_EQ(dd->fidelity(zeroState, bellState), 0.5);

  export2Dot(bellState, "bell_state_colored_labels.dot", true, true, false,
             false, false);
  export2Dot(bellState, "bell_state_colored_labels_classic.dot", true, true,
             true, false, false);
  export2Dot(bellState, "bell_state_mono_labels.dot", false, true, false, false,
             false);
  export2Dot(bellState, "bell_state_mono_labels_classic.dot", false, true, true,
             false, false);
  export2Dot(bellState, "bell_state_colored.dot", true, false, false, false,
             false);
  export2Dot(bellState, "bell_state_colored_classic.dot", true, false, true,
             false, false);
  export2Dot(bellState, "bell_state_mono.dot", false, false, false, false,
             false);
  export2Dot(bellState, "bell_state_mono_classic.dot", false, false, true,
             false, false);
  export2Dot(bellState, "bell_state_memory.dot", false, true, true, true,
             false);
  dd::exportEdgeWeights(bellState, std::cout);

  const auto filenames = {
      "bell_state_colored_labels.dot", "bell_state_colored_labels_classic.dot",
      "bell_state_mono_labels.dot",    "bell_state_mono_labels_classic.dot",
      "bell_state_colored.dot",        "bell_state_colored_classic.dot",
      "bell_state_mono.dot",           "bell_state_mono_classic.dot",
      "bell_state_memory.dot"};

  for (const auto* const filename : filenames) {
    std::ifstream ifs(filename);
    ASSERT_TRUE(ifs.good());
    ASSERT_NE(ifs.peek(), std::ifstream::traits_type::eof());
    ifs.close();
    std::filesystem::remove(filename);
  }

  dd::printStatistics(dd.get());
}

TEST(DDPackageTest, QFTState) {
  auto dd = std::make_unique<dd::Package<>>(3);

  auto h0Gate = dd->makeGateDD(dd::H_MAT, 0);
  auto s0Gate = dd->makeGateDD(dd::S_MAT, 1_pc, 0);
  auto t0Gate = dd->makeGateDD(dd::T_MAT, 2_pc, 0);
  auto h1Gate = dd->makeGateDD(dd::H_MAT, 1);
  auto s1Gate = dd->makeGateDD(dd::S_MAT, 2_pc, 1);
  auto h2Gate = dd->makeGateDD(dd::H_MAT, 2);
  auto swapGate = dd->makeTwoQubitGateDD(dd::SWAP_MAT, qc::Controls{}, 0, 2);

  auto qftOp = dd->multiply(s0Gate, h0Gate);
  qftOp = dd->multiply(t0Gate, qftOp);
  qftOp = dd->multiply(h1Gate, qftOp);
  qftOp = dd->multiply(s1Gate, qftOp);
  qftOp = dd->multiply(h2Gate, qftOp);

  qftOp = dd->multiply(swapGate, qftOp);
  auto qftState = dd->multiply(qftOp, dd->makeZeroState(3));

  qftState.printVector();

  for (dd::Qubit qubit = 0; qubit < 7; ++qubit) {
    ASSERT_NEAR(
        qftState.getValueByIndex(static_cast<std::size_t>(qubit)).real(),
        0.5 * dd::SQRT2_2, dd::RealNumber::eps);
    ASSERT_EQ(qftState.getValueByIndex(static_cast<std::size_t>(qubit)).imag(),
              0);
  }

  export2Dot(qftState, "qft_state_colored_labels.dot", true, true, false, false,
             false);
  export2Dot(qftState, "qft_state_colored_labels_classic.dot", true, true, true,
             false, false);
  export2Dot(qftState, "qft_state_mono_labels.dot", false, true, false, false,
             false);
  export2Dot(qftState, "qft_state_mono_labels_classic.dot", false, true, true,
             false, false);
  export2Dot(qftState, "qft_state_colored.dot", true, false, false, false,
             false);
  export2Dot(qftState, "qft_state_colored_classic.dot", true, false, true,
             false, false);
  export2Dot(qftState, "qft_state_mono.dot", false, false, false, false, false);
  export2Dot(qftState, "qft_state_mono_classic.dot", false, false, true, false,
             false);
  export2Dot(qftState, "qft_state_memory.dot", false, true, true, true, false);
  dd::exportEdgeWeights(qftState, std::cout);

  export2Dot(qftOp, "qft_op_polar_colored_labels.dot", true, true, false, false,
             false, true);
  export2Dot(qftOp, "qft_op_polar_colored_labels_classic.dot", true, true, true,
             false, false, true);
  export2Dot(qftOp, "qft_op_polar_mono_labels.dot", false, true, false, false,
             false, true);
  export2Dot(qftOp, "qft_op_polar_mono_labels_classic.dot", false, true, true,
             false, false, true);
  export2Dot(qftOp, "qft_op_polar_colored.dot", true, false, false, false,
             false, true);
  export2Dot(qftOp, "qft_op_polar_colored_classic.dot", true, false, true,
             false, false, true);
  export2Dot(qftOp, "qft_op_polar_mono.dot", false, false, false, false, false,
             true);
  export2Dot(qftOp, "qft_op_polar_mono_classic.dot", false, false, true, false,
             false, true);
  export2Dot(qftOp, "qft_op_polar_memory.dot", false, true, true, true, false,
             true);

  export2Dot(qftOp, "qft_op_rectangular_colored_labels.dot", true, true, false,
             false, false, false);
  export2Dot(qftOp, "qft_op_rectangular_colored_labels_classic.dot", true, true,
             true, false, false, false);
  export2Dot(qftOp, "qft_op_rectangular_mono_labels.dot", false, true, false,
             false, false, false);
  export2Dot(qftOp, "qft_op_rectangular_mono_labels_classic.dot", false, true,
             true, false, false, false);
  export2Dot(qftOp, "qft_op_rectangular_colored.dot", true, false, false, false,
             false, false);
  export2Dot(qftOp, "qft_op_rectangular_colored_classic.dot", true, false, true,
             false, false, false);
  export2Dot(qftOp, "qft_op_rectangular_mono.dot", false, false, false, false,
             false, false);
  export2Dot(qftOp, "qft_op_rectangular_mono_classic.dot", false, false, true,
             false, false, false);
  export2Dot(qftOp, "qft_op_rectangular_memory.dot", false, true, true, true,
             false, false);

  const auto filenames = {"qft_state_colored_labels.dot",
                          "qft_state_colored_labels_classic.dot",
                          "qft_state_mono_labels.dot",
                          "qft_state_mono_labels_classic.dot",
                          "qft_state_colored.dot",
                          "qft_state_colored_classic.dot",
                          "qft_state_mono.dot",
                          "qft_state_mono_classic.dot",
                          "qft_state_memory.dot",
                          "qft_op_polar_colored_labels.dot",
                          "qft_op_polar_colored_labels_classic.dot",
                          "qft_op_polar_mono_labels.dot",
                          "qft_op_polar_mono_labels_classic.dot",
                          "qft_op_polar_colored.dot",
                          "qft_op_polar_colored_classic.dot",
                          "qft_op_polar_mono.dot",
                          "qft_op_polar_mono_classic.dot",
                          "qft_op_polar_memory.dot",
                          "qft_op_rectangular_colored_labels.dot",
                          "qft_op_rectangular_colored_labels_classic.dot",
                          "qft_op_rectangular_mono_labels.dot",
                          "qft_op_rectangular_mono_labels_classic.dot",
                          "qft_op_rectangular_colored.dot",
                          "qft_op_rectangular_colored_classic.dot",
                          "qft_op_rectangular_mono.dot",
                          "qft_op_rectangular_mono_classic.dot",
                          "qft_op_rectangular_memory.dot"};

  for (const auto* const filename : filenames) {
    std::ifstream ifs(filename);
    ASSERT_TRUE(ifs.good());
    ASSERT_NE(ifs.peek(), std::ifstream::traits_type::eof());
    ifs.close();
    std::filesystem::remove(filename);
  }

  dd::printStatistics(dd.get());
}

TEST(DDPackageTest, CorruptedBellState) {
  auto dd = std::make_unique<dd::Package<>>(2);

  auto hGate = dd->makeGateDD(dd::H_MAT, 1);
  auto cxGate = dd->makeGateDD(dd::X_MAT, 1_pc, 0);
  auto zeroState = dd->makeZeroState(2);

  auto bellState = dd->multiply(dd->multiply(cxGate, hGate), zeroState);

  bellState.w = dd->cn.lookup(0.5, 0);
  // prints a warning
  std::mt19937_64 mt; // NOLINT(cert-msc51-cpp)
  std::cout << dd->measureAll(bellState, false, mt) << "\n";

  bellState.w = dd::Complex::zero();

  ASSERT_THROW(dd->measureAll(bellState, false, mt), std::runtime_error);

  ASSERT_THROW(dd->measureOneCollapsing(bellState, 0, true, mt),
               std::runtime_error);
}

TEST(DDPackageTest, NegativeControl) {
  auto dd = std::make_unique<dd::Package<>>(2);

  auto xGate = dd->makeGateDD(dd::X_MAT, 1_nc, 0);
  auto zeroState = dd->makeZeroState(2);
  auto state01 = dd->multiply(xGate, zeroState);
  EXPECT_EQ(state01.getValueByIndex(0b01).real(), 1.);
}

TEST(DDPackageTest, IdentityTrace) {
  auto dd = std::make_unique<dd::Package<>>(4);
  auto fullTrace = dd->trace(dd->makeIdent(), 4);

  ASSERT_EQ(fullTrace.r, 1.);
}

TEST(DDPackageTest, CNotKronTrace) {
  auto dd = std::make_unique<dd::Package<>>(4);
  auto cxGate = dd->makeGateDD(dd::X_MAT, 1_pc, 0);
  auto cxGateKron = dd->kronecker(cxGate, cxGate, 2);
  auto fullTrace = dd->trace(cxGateKron, 4);
  ASSERT_EQ(fullTrace, 0.25);
}

TEST(DDPackageTest, PartialIdentityTrace) {
  auto dd = std::make_unique<dd::Package<>>(2);
  auto tr = dd->partialTrace(dd->makeIdent(), {false, true});
  auto mul = dd->multiply(tr, tr);
  EXPECT_EQ(dd::RealNumber::val(mul.w.r), 1.);
}

TEST(DDPackageTest, PartialSWapMatTrace) {
  auto dd = std::make_unique<dd::Package<>>(2);
  auto swapGate = dd->makeTwoQubitGateDD(dd::SWAP_MAT, 0, 1);
  auto ptr = dd->partialTrace(swapGate, {true, false});
  auto fullTrace = dd->trace(ptr, 1);
  auto fullTraceOriginal = dd->trace(swapGate, 2);
  EXPECT_EQ(dd::RealNumber::val(ptr.w.r), 0.5);
  // Check that successively tracing out subsystems is the same as computing the
  // full trace from the beginning
  EXPECT_EQ(fullTrace, fullTraceOriginal);
}

TEST(DDPackageTest, PartialTraceKeepInnerQubits) {
  // Check that the partial trace computation is correct when tracing out the
  // outer qubits only. This test shows that we should avoid storing
  // non-eliminated nodes in the compute table, as this would prevent their
  // proper elimination in subsequent trace calls.

  const std::size_t numQubits = 8;
  auto dd = std::make_unique<dd::Package<>>(numQubits);
  const auto swapGate = dd->makeTwoQubitGateDD(dd::SWAP_MAT, 0, 1);
  auto swapKron = swapGate;
  for (std::size_t i = 0; i < 3; ++i) {
    swapKron = dd->kronecker(swapKron, swapGate, 2);
  }
  auto fullTraceOriginal = dd->trace(swapKron, numQubits);
  auto ptr = dd->partialTrace(
      swapKron, {true, true, false, false, false, false, true, true});
  auto fullTrace = dd->trace(ptr, 4);
  EXPECT_EQ(dd::RealNumber::val(ptr.w.r), 0.25);
  EXPECT_EQ(fullTrace.r, 0.0625);
  // Check that successively tracing out subsystems is the same as computing the
  // full trace from the beginning
  EXPECT_EQ(fullTrace, fullTraceOriginal);
}

TEST(DDPackageTest, TraceComplexity) {
  // Check that the full trace computation scales with the number of nodes
  // instead of paths in the DD due to the usage of a compute table
  for (std::size_t numQubits = 1; numQubits <= 10; ++numQubits) {
    auto dd = std::make_unique<dd::Package<>>(numQubits);
    auto& computeTable = dd->getTraceComputeTable<dd::mNode>();
    const auto hGate = dd->makeGateDD(dd::H_MAT, 0);
    auto hKron = hGate;
    for (std::size_t i = 0; i < numQubits - 1; ++i) {
      hKron = dd->kronecker(hKron, hGate, 1);
    }
    dd->trace(hKron, numQubits);
    const auto& stats = computeTable.getStats();
    ASSERT_EQ(stats.lookups, 2 * numQubits - 1);
    ASSERT_EQ(stats.hits, numQubits - 1);
  }
}

TEST(DDPackageTest, KeepBottomQubitsPartialTraceComplexity) {
  // Check that during the trace computation, once a level is reached
  // where the remaining qubits should not be eliminated, the function does not
  // recurse further but immediately returns the current CachedEdge<Node>.
  const std::size_t numQubits = 8;
  auto dd = std::make_unique<dd::Package<>>(numQubits);
  auto& uniqueTable = dd->getUniqueTable<dd::mNode>();
  const auto hGate = dd->makeGateDD(dd::H_MAT, 0);
  auto hKron = hGate;
  for (std::size_t i = 0; i < numQubits - 1; ++i) {
    hKron = dd->kronecker(hKron, hGate, 1);
  }

  const std::size_t maxNodeVal = 6;
  std::array<std::size_t, maxNodeVal> lookupValues{};

  for (std::size_t i = 0; i < maxNodeVal; ++i) {
    // Store the number of lookups performed so far for the six bottom qubits
    lookupValues[i] = uniqueTable.getStats(i).lookups;
  }
  dd->partialTrace(hKron,
                   {false, false, false, false, false, false, true, true});
  for (std::size_t i = 0; i < maxNodeVal; ++i) {
    // Check that the partial trace computation performs no additional lookups
    // on the bottom qubits that are not eliminated
    ASSERT_EQ(uniqueTable.getStats(i).lookups, lookupValues[i]);
  }
}

TEST(DDPackageTest, PartialTraceComplexity) {
  // In the worst case, the partial trace computation scales with the number of
  // paths in the DD. This situation arises particularly when tracing out the
  // bottom qubits.
  const std::size_t numQubits = 9;
  auto dd = std::make_unique<dd::Package<>>(numQubits);
  auto& uniqueTable = dd->getUniqueTable<dd::mNode>();
  const auto hGate = dd->makeGateDD(dd::H_MAT, 0);
  auto hKron = hGate;
  for (std::size_t i = 0; i < numQubits - 2; ++i) {
    hKron = dd->kronecker(hKron, hGate, 1);
  }
  hKron = dd->kronecker(hKron, dd->makeIdent(), 1);

  const std::size_t maxNodeVal = 6;
  std::array<std::size_t, maxNodeVal + 1> lookupValues{};
  for (std::size_t i = 1; i <= maxNodeVal; ++i) {
    // Store the number of lookups performed so far for levels 1 through 6
    lookupValues[i] = uniqueTable.getStats(i).lookups;
  }

  dd->partialTrace(
      hKron, {true, false, false, false, false, false, false, true, true});
  for (std::size_t i = 1; i < maxNodeVal; ++i) {
    // Check that the number of lookups scales with the number of paths in the
    // DD
    ASSERT_EQ(uniqueTable.getStats(i).lookups,
              lookupValues[i] +
                  static_cast<std::size_t>(std::pow(4, (maxNodeVal - i))));
  }
}

TEST(DDPackageTest, StateGenerationManipulation) {
  const std::size_t nqubits = 6;
  auto dd = std::make_unique<dd::Package<>>(nqubits);
  auto b = std::vector<bool>(nqubits, false);
  b[0] = b[1] = true;
  auto e = dd->makeBasisState(nqubits, b);
  auto f = dd->makeBasisState(nqubits,
                              {dd::BasisStates::zero, dd::BasisStates::one,
                               dd::BasisStates::plus, dd::BasisStates::minus,
                               dd::BasisStates::left, dd::BasisStates::right});
  dd->incRef(e);
  dd->incRef(f);
  dd->vUniqueTable.print();
  dd->decRef(e);
  dd->decRef(f);
}

TEST(DDPackageTest, VectorSerializationTest) {
  auto dd = std::make_unique<dd::Package<>>(2);

  auto hGate = dd->makeGateDD(dd::H_MAT, 1);
  auto cxGate = dd->makeGateDD(dd::X_MAT, 1_pc, 0);
  auto zeroState = dd->makeZeroState(2);

  auto bellState = dd->multiply(dd->multiply(cxGate, hGate), zeroState);

  serialize(bellState, "bell_state.dd", false);
  auto deserializedBellState =
      dd->deserialize<dd::vNode>("bell_state.dd", false);
  EXPECT_EQ(bellState, deserializedBellState);
  std::filesystem::remove("bell_state.dd");

  serialize(bellState, "bell_state_binary.dd", true);
  deserializedBellState =
      dd->deserialize<dd::vNode>("bell_state_binary.dd", true);
  EXPECT_EQ(bellState, deserializedBellState);
  std::filesystem::remove("bell_state_binary.dd");
}

TEST(DDPackageTest, BellMatrix) {
  auto dd = std::make_unique<dd::Package<>>(2);

  auto hGate = dd->makeGateDD(dd::H_MAT, 1);
  auto cxGate = dd->makeGateDD(dd::X_MAT, 1_pc, 0);

  auto bellMatrix = dd->multiply(cxGate, hGate);

  bellMatrix.printMatrix(dd->qubits());

  ASSERT_EQ(bellMatrix.getValueByPath(dd->qubits(), "00"), dd::SQRT2_2);
  ASSERT_EQ(bellMatrix.getValueByPath(dd->qubits(), "02"), 0.);
  ASSERT_EQ(bellMatrix.getValueByPath(dd->qubits(), "20"), 0.);
  ASSERT_EQ(bellMatrix.getValueByPath(dd->qubits(), "22"), dd::SQRT2_2);

  ASSERT_EQ(bellMatrix.getValueByIndex(dd->qubits(), 0, 0), dd::SQRT2_2);
  ASSERT_EQ(bellMatrix.getValueByIndex(dd->qubits(), 1, 0), 0.);
  ASSERT_EQ(bellMatrix.getValueByIndex(dd->qubits(), 2, 0), 0.);
  ASSERT_EQ(bellMatrix.getValueByIndex(dd->qubits(), 3, 0), dd::SQRT2_2);

  ASSERT_EQ(bellMatrix.getValueByIndex(dd->qubits(), 0, 1), 0.);
  ASSERT_EQ(bellMatrix.getValueByIndex(dd->qubits(), 1, 1), dd::SQRT2_2);
  ASSERT_EQ(bellMatrix.getValueByIndex(dd->qubits(), 2, 1), dd::SQRT2_2);
  ASSERT_EQ(bellMatrix.getValueByIndex(dd->qubits(), 3, 1), 0.);

  ASSERT_EQ(bellMatrix.getValueByIndex(dd->qubits(), 0, 2), dd::SQRT2_2);
  ASSERT_EQ(bellMatrix.getValueByIndex(dd->qubits(), 1, 2), 0.);
  ASSERT_EQ(bellMatrix.getValueByIndex(dd->qubits(), 2, 2), 0.);
  ASSERT_EQ(bellMatrix.getValueByIndex(dd->qubits(), 3, 2), -dd::SQRT2_2);

  ASSERT_EQ(bellMatrix.getValueByIndex(dd->qubits(), 0, 3), 0.);
  ASSERT_EQ(bellMatrix.getValueByIndex(dd->qubits(), 1, 3), dd::SQRT2_2);
  ASSERT_EQ(bellMatrix.getValueByIndex(dd->qubits(), 2, 3), -dd::SQRT2_2);
  ASSERT_EQ(bellMatrix.getValueByIndex(dd->qubits(), 3, 3), 0.);

  auto goalRow0 =
      dd::CVec{{dd::SQRT2_2, 0.}, {0., 0.}, {dd::SQRT2_2, 0.}, {0., 0.}};
  auto goalRow1 =
      dd::CVec{{0., 0.}, {dd::SQRT2_2, 0.}, {0., 0.}, {dd::SQRT2_2, 0.}};
  auto goalRow2 =
      dd::CVec{{0., 0.}, {dd::SQRT2_2, 0.}, {0., 0.}, {-dd::SQRT2_2, 0.}};
  auto goalRow3 =
      dd::CVec{{dd::SQRT2_2, 0.}, {0., 0.}, {-dd::SQRT2_2, 0.}, {0., 0.}};
  auto goalMatrix = dd::CMat{goalRow0, goalRow1, goalRow2, goalRow3};
  ASSERT_EQ(bellMatrix.getMatrix(dd->qubits()), goalMatrix);

  export2Dot(bellMatrix, "bell_matrix_colored_labels.dot", true, true, false,
             false, false);
  export2Dot(bellMatrix, "bell_matrix_colored_labels_classic.dot", true, true,
             true, false, false);
  export2Dot(bellMatrix, "bell_matrix_mono_labels.dot", false, true, false,
             false, false);
  export2Dot(bellMatrix, "bell_matrix_mono_labels_classic.dot", false, true,
             true, false, false);
  export2Dot(bellMatrix, "bell_matrix_colored.dot", true, false, false, false,
             false);
  export2Dot(bellMatrix, "bell_matrix_colored_classic.dot", true, false, true,
             false, false);
  export2Dot(bellMatrix, "bell_matrix_mono.dot", false, false, false, false,
             false);
  export2Dot(bellMatrix, "bell_matrix_mono_classic.dot", false, false, true,
             false, false);
  export2Dot(bellMatrix, "bell_matrix_memory.dot", false, true, true, true,
             false);

  const auto filenames = {"bell_matrix_colored_labels.dot",
                          "bell_matrix_colored_labels_classic.dot",
                          "bell_matrix_mono_labels.dot",
                          "bell_matrix_mono_labels_classic.dot",
                          "bell_matrix_colored.dot",
                          "bell_matrix_colored_classic.dot",
                          "bell_matrix_mono.dot",
                          "bell_matrix_mono_classic.dot",
                          "bell_matrix_memory.dot"};

  for (const auto* const filename : filenames) {
    std::ifstream ifs(filename);
    ASSERT_TRUE(ifs.good());
    ASSERT_NE(ifs.peek(), std::ifstream::traits_type::eof());
    ifs.close();
    std::filesystem::remove(filename);
  }

  dd::printStatistics(dd.get());
}

TEST(DDPackageTest, MatrixSerializationTest) {
  auto dd = std::make_unique<dd::Package<>>(2);

  auto hGate = dd->makeGateDD(dd::H_MAT, 1);
  auto cxGate = dd->makeGateDD(dd::X_MAT, 1_pc, 0);

  auto bellMatrix = dd->multiply(cxGate, hGate);

  serialize(bellMatrix, "bell_matrix.dd", false);
  auto deserializedBellMatrix =
      dd->deserialize<dd::mNode>("bell_matrix.dd", false);
  EXPECT_EQ(bellMatrix, deserializedBellMatrix);
  std::filesystem::remove("bell_matrix.dd");

  serialize(bellMatrix, "bell_matrix_binary.dd", true);
  deserializedBellMatrix =
      dd->deserialize<dd::mNode>("bell_matrix_binary.dd", true);
  EXPECT_EQ(bellMatrix, deserializedBellMatrix);
  std::filesystem::remove("bell_matrix_binary.dd");
}

TEST(DDPackageTest, SerializationErrors) {
  auto dd = std::make_unique<dd::Package<>>(2);

  auto hGate = dd->makeGateDD(dd::H_MAT, 1);
  auto cxGate = dd->makeGateDD(dd::X_MAT, 1_pc, 0);
  auto zeroState = dd->makeZeroState(2);
  auto bellState = dd->multiply(dd->multiply(cxGate, hGate), zeroState);

  // test non-existing file
  EXPECT_THROW(serialize(bellState, "./path/that/does/not/exist/filename.dd"),
               std::invalid_argument);
  EXPECT_THROW(dd->deserialize<dd::vNode>(
                   "./path/that/does/not/exist/filename.dd", true),
               std::invalid_argument);

  // test wrong version number
  std::stringstream ss{};
  ss << 2 << "\n";
  EXPECT_THROW(dd->deserialize<dd::vNode>(ss, false), std::runtime_error);
  ss << 2 << "\n";
  EXPECT_THROW(dd->deserialize<dd::mNode>(ss, false), std::runtime_error);

  ss.str("");
  std::remove_const_t<decltype(dd::SERIALIZATION_VERSION)> version = 2;
  ss.write(reinterpret_cast<const char*>(&version),
           sizeof(decltype(dd::SERIALIZATION_VERSION)));
  EXPECT_THROW(dd->deserialize<dd::vNode>(ss, true), std::runtime_error);
  ss.write(reinterpret_cast<const char*>(&version),
           sizeof(decltype(dd::SERIALIZATION_VERSION)));
  EXPECT_THROW(dd->deserialize<dd::mNode>(ss, true), std::runtime_error);

  // test wrong format
  ss.str("");
  ss << "1\n";
  ss << "not_complex\n";
  EXPECT_THROW(dd->deserialize<dd::vNode>(ss), std::runtime_error);
  ss << "1\n";
  ss << "not_complex\n";
  EXPECT_THROW(dd->deserialize<dd::mNode>(ss), std::runtime_error);

  ss.str("");
  ss << "1\n";
  ss << "1.0\n";
  ss << "no_node_here\n";
  EXPECT_THROW(dd->deserialize<dd::vNode>(ss), std::runtime_error);
  ss << "1\n";
  ss << "1.0\n";
  ss << "no_node_here\n";
  EXPECT_THROW(dd->deserialize<dd::mNode>(ss), std::runtime_error);
}

TEST(DDPackageTest, Ancillaries) {
  auto dd = std::make_unique<dd::Package<>>(4);
  auto hGate = dd->makeGateDD(dd::H_MAT, 0);
  auto cxGate = dd->makeGateDD(dd::X_MAT, 0_pc, 1);
  auto bellMatrix = dd->multiply(cxGate, hGate);

  dd->incRef(bellMatrix);
  auto reducedBellMatrix =
      dd->reduceAncillae(bellMatrix, {false, false, false, false});
  EXPECT_EQ(bellMatrix, reducedBellMatrix);

  dd->incRef(bellMatrix);
  reducedBellMatrix =
      dd->reduceAncillae(bellMatrix, {false, false, true, true});
  EXPECT_TRUE(reducedBellMatrix.p->e[1].isZeroTerminal());
  EXPECT_TRUE(reducedBellMatrix.p->e[2].isZeroTerminal());
  EXPECT_TRUE(reducedBellMatrix.p->e[3].isZeroTerminal());

  EXPECT_EQ(reducedBellMatrix.p->e[0].p->e[0].p, bellMatrix.p);
  EXPECT_TRUE(reducedBellMatrix.p->e[0].p->e[1].isZeroTerminal());
  EXPECT_TRUE(reducedBellMatrix.p->e[0].p->e[2].isZeroTerminal());
  EXPECT_TRUE(reducedBellMatrix.p->e[0].p->e[3].isZeroTerminal());

  dd->incRef(bellMatrix);
  reducedBellMatrix =
      dd->reduceAncillae(bellMatrix, {false, false, true, true}, false);
  EXPECT_TRUE(reducedBellMatrix.p->e[1].isZeroTerminal());
  EXPECT_TRUE(reducedBellMatrix.p->e[2].isZeroTerminal());
  EXPECT_TRUE(reducedBellMatrix.p->e[3].isZeroTerminal());

  EXPECT_EQ(reducedBellMatrix.p->e[0].p->e[0].p, bellMatrix.p);
  EXPECT_TRUE(reducedBellMatrix.p->e[0].p->e[1].isZeroTerminal());
  EXPECT_TRUE(reducedBellMatrix.p->e[0].p->e[2].isZeroTerminal());
  EXPECT_TRUE(reducedBellMatrix.p->e[0].p->e[3].isZeroTerminal());
}

TEST(DDPackageTest, GarbageVector) {
  auto dd = std::make_unique<dd::Package<>>(4);
  auto hGate = dd->makeGateDD(dd::H_MAT, 0);
  auto cxGate = dd->makeGateDD(dd::X_MAT, 0_pc, 1);
  auto zeroState = dd->makeZeroState(2);
  auto bellState = dd->multiply(dd->multiply(cxGate, hGate), zeroState);
  std::cout << "Bell State:\n";
  bellState.printVector();

  dd->incRef(bellState);
  auto reducedBellState =
      dd->reduceGarbage(bellState, {false, false, false, false});
  EXPECT_EQ(bellState, reducedBellState);
  dd->incRef(bellState);
  reducedBellState = dd->reduceGarbage(bellState, {false, false, true, false});
  EXPECT_EQ(bellState, reducedBellState);

  dd->incRef(bellState);
  reducedBellState = dd->reduceGarbage(bellState, {false, true, false, false});
  auto vec = reducedBellState.getVector();
  std::cout << "Reduced Bell State (q1 garbage):\n";
  reducedBellState.printVector();
  EXPECT_EQ(vec[2], 0.);
  EXPECT_EQ(vec[3], 0.);

  dd->incRef(bellState);
  reducedBellState = dd->reduceGarbage(bellState, {true, false, false, false});
  std::cout << "Reduced Bell State (q0 garbage):\n";
  reducedBellState.printVector();
  vec = reducedBellState.getVector();
  EXPECT_EQ(vec[1], 0.);
  EXPECT_EQ(vec[3], 0.);
}

TEST(DDPackageTest, GarbageMatrix) {
  auto dd = std::make_unique<dd::Package<>>(4);
  auto hGate = dd->makeGateDD(dd::H_MAT, 0);
  auto cxGate = dd->makeGateDD(dd::X_MAT, 0_pc, 1);
  auto bellMatrix = dd->multiply(cxGate, hGate);

  dd->incRef(bellMatrix);
  auto reducedBellMatrix =
      dd->reduceGarbage(bellMatrix, {false, false, false, false});
  EXPECT_EQ(bellMatrix, reducedBellMatrix);
  dd->incRef(bellMatrix);
  reducedBellMatrix =
      dd->reduceGarbage(bellMatrix, {false, false, true, false});
  EXPECT_NE(bellMatrix, reducedBellMatrix);

  dd->incRef(bellMatrix);
  reducedBellMatrix =
      dd->reduceGarbage(bellMatrix, {false, true, false, false});
  auto mat = reducedBellMatrix.getMatrix(2);
  auto zero = dd::CVec{{0., 0.}, {0., 0.}, {0., 0.}, {0., 0.}};
  EXPECT_EQ(mat[2], zero);
  EXPECT_EQ(mat[3], zero);

  dd->incRef(bellMatrix);
  reducedBellMatrix =
      dd->reduceGarbage(bellMatrix, {true, false, false, false});
  mat = reducedBellMatrix.getMatrix(2);
  EXPECT_EQ(mat[1], zero);
  EXPECT_EQ(mat[3], zero);

  dd->incRef(bellMatrix);
  reducedBellMatrix =
      dd->reduceGarbage(bellMatrix, {false, true, false, false}, false);
  EXPECT_TRUE(reducedBellMatrix.p->e[1].isZeroTerminal());
  EXPECT_TRUE(reducedBellMatrix.p->e[3].isZeroTerminal());
}

TEST(DDPackageTest, ReduceGarbageVector) {
  auto dd = std::make_unique<dd::Package<>>(3);
  auto xGate = dd->makeGateDD(dd::X_MAT, 2);
  auto hGate = dd->makeGateDD(dd::H_MAT, 2);
  auto zeroState = dd->makeZeroState(3);
  auto initialState = dd->multiply(dd->multiply(hGate, xGate), zeroState);
  std::cout << "Initial State:\n";
  initialState.printVector();

  dd->incRef(initialState);
  auto reducedState = dd->reduceGarbage(initialState, {false, true, true});
  std::cout << "After reduceGarbage():\n";
  reducedState.printVector();
  EXPECT_EQ(reducedState, dd->makeZeroState(3));

  dd->incRef(initialState);
  auto reducedState2 =
      dd->reduceGarbage(initialState, {false, true, true}, true);

  EXPECT_EQ(reducedState2, dd->makeZeroState(3));
}

TEST(DDPackageTest, ReduceGarbageVectorTGate) {
  const auto nqubits = 2U;
  const auto dd = std::make_unique<dd::Package<>>(nqubits);
  const auto xGate0 = dd->makeGateDD(dd::X_MAT, 0);
  const auto xGate1 = dd->makeGateDD(dd::X_MAT, 1);
  const auto tdgGate0 = dd->makeGateDD(dd::TDG_MAT, 0);

  auto zeroState = dd->makeZeroState(nqubits);
  auto initialState = dd->multiply(
      dd->multiply(tdgGate0, dd->multiply(xGate0, xGate1)), zeroState);
  std::cout << "Initial State:\n";
  initialState.printVector();

  dd->incRef(initialState);
  auto reducedState = dd->reduceGarbage(initialState, {false, false}, true);
  std::cout << "After reduceGarbage():\n";
  reducedState.printVector();
  EXPECT_EQ(reducedState,
            dd->multiply(dd->multiply(xGate0, xGate1), zeroState));
}

TEST(DDPackageTest, ReduceGarbageMatrix) {
  auto dd = std::make_unique<dd::Package<>>(3);
  auto hGate = dd->makeGateDD(dd::H_MAT, 0);
  auto cNotGate = dd->makeGateDD(dd::X_MAT, qc::Controls{0}, 1);

  auto initialState = dd->multiply(hGate, cNotGate);

  std::cout << "Initial State:\n";
  initialState.printMatrix(dd->qubits());

  dd->incRef(initialState);
  auto reducedState1 =
      dd->reduceGarbage(initialState, {false, true, true}, true, true);
  std::cout << "After reduceGarbage(q1 and q2 are garbage):\n";
  reducedState1.printMatrix(dd->qubits());

  auto expectedMatrix1 =
      dd::CMat{{dd::SQRT2_2, dd::SQRT2_2, dd::SQRT2_2, dd::SQRT2_2, dd::SQRT2_2,
                dd::SQRT2_2, dd::SQRT2_2, dd::SQRT2_2},
               {dd::SQRT2_2, dd::SQRT2_2, dd::SQRT2_2, dd::SQRT2_2, dd::SQRT2_2,
                dd::SQRT2_2, dd::SQRT2_2, dd::SQRT2_2},
               {0, 0, 0, 0, 0, 0, 0, 0},
               {0, 0, 0, 0, 0, 0, 0, 0},
               {0, 0, 0, 0, 0, 0, 0, 0},
               {0, 0, 0, 0, 0, 0, 0, 0},
               {0, 0, 0, 0, 0, 0, 0, 0},
               {0, 0, 0, 0, 0, 0, 0, 0}};
  EXPECT_EQ(reducedState1.getMatrix(dd->qubits()), expectedMatrix1);

  dd->incRef(initialState);
  auto reducedState2 =
      dd->reduceGarbage(initialState, {true, false, false}, true, true);
  std::cout << "After reduceGarbage(q0 is garbage):\n";
  reducedState2.printMatrix(dd->qubits());

  auto expectedMatrix2 =
      dd::CMat{{1, 0, 0, 1, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0},
               {0, 1, 1, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0},
               {0, 0, 0, 0, 1, 0, 0, 1}, {0, 0, 0, 0, 0, 0, 0, 0},
               {0, 0, 0, 0, 0, 1, 1, 0}, {0, 0, 0, 0, 0, 0, 0, 0}};
  EXPECT_EQ(reducedState2.getMatrix(dd->qubits()), expectedMatrix2);
}

TEST(DDPackageTest, ReduceGarbageMatrix2) {
  const auto nqubits = 3U;
  const auto dd = std::make_unique<dd::Package<>>(nqubits);
  const auto controlledSwapGate =
      dd->makeTwoQubitGateDD(dd::SWAP_MAT, qc::Controls{1}, 0, 2);
  const auto hGate = dd->makeGateDD(dd::H_MAT, 0);
  const auto zGate = dd->makeGateDD(dd::Z_MAT, 2);
  const auto xGate = dd->makeGateDD(dd::X_MAT, 1);
  const auto controlledHGate = dd->makeGateDD(dd::H_MAT, qc::Controls{1}, 0);

  auto c1 = dd->multiply(
      controlledSwapGate,
      dd->multiply(hGate, dd->multiply(zGate, controlledSwapGate)));
  auto c2 = dd->multiply(controlledHGate, xGate);

  std::cout << "c1:\n";
  c1.printMatrix(dd->qubits());
  std::cout << "reduceGarbage:\n";
  dd->incRef(c1);
  auto c1Reduced = dd->reduceGarbage(c1, {false, true, true}, true, true);
  c1Reduced.printMatrix(dd->qubits());

  std::cout << "c2:\n";
  c2.printMatrix(dd->qubits());
  std::cout << "reduceGarbage:\n";
  dd->incRef(c2);
  auto c2Reduced = dd->reduceGarbage(c2, {false, true, true}, true, true);
  c2Reduced.printMatrix(dd->qubits());

  EXPECT_EQ(c1Reduced, c2Reduced);
}

TEST(DDPackageTest, ReduceGarbageMatrixNoGarbage) {
  const auto nqubits = 2U;
  const auto dd = std::make_unique<dd::Package<>>(nqubits);
  const auto tdgGate0 = dd->makeGateDD(dd::TDG_MAT, 0);
  const auto tdgGate1 = dd->makeGateDD(dd::TDG_MAT, 1);

  auto c1 = dd->makeIdent();
  auto c2 = dd->multiply(tdgGate0, tdgGate1);

  std::cout << "c2:\n";
  c2.printMatrix(dd->qubits());
  std::cout << "reduceGarbage:\n";
  dd->incRef(c2);
  auto c2Reduced = dd->reduceGarbage(c2, {false, false}, true, true);
  c2Reduced.printMatrix(dd->qubits());

  EXPECT_EQ(c1, c2Reduced);
}

TEST(DDPackageTest, ReduceGarbageMatrixTGate) {
  const auto nqubits = 2U;
  const auto dd = std::make_unique<dd::Package<>>(nqubits);
  const auto tdgGate0 = dd->makeGateDD(dd::TDG_MAT, 0);
  const auto tdgGate1 = dd->makeGateDD(dd::TDG_MAT, 1);

  auto c1 = dd->makeIdent();
  auto c2 = dd->multiply(tdgGate0, tdgGate1);

  std::cout << "c1:\n";
  c1.printMatrix(dd->qubits());
  std::cout << "reduceGarbage:\n";
  dd->incRef(c1);
  auto c1Reduced = dd->reduceGarbage(c1, {false, true}, true, true);
  c1Reduced.printMatrix(dd->qubits());

  std::cout << "c2:\n";
  c2.printMatrix(dd->qubits());
  std::cout << "reduceGarbage:\n";
  dd->incRef(c2);
  auto c2Reduced = dd->reduceGarbage(c2, {false, true}, true, true);
  c2Reduced.printMatrix(dd->qubits());

  EXPECT_EQ(c1Reduced, c2Reduced);
}

TEST(DDPackageTest, InvalidMakeBasisStateAndGate) {
  auto nqubits = 2U;
  auto dd = std::make_unique<dd::Package<>>(nqubits);
  auto basisState = std::vector<dd::BasisStates>{dd::BasisStates::zero};
  EXPECT_THROW(dd->makeBasisState(nqubits, basisState), std::runtime_error);
  EXPECT_THROW(dd->makeZeroState(3), std::runtime_error);
  EXPECT_THROW(dd->makeBasisState(3, {true, true, true}), std::runtime_error);
  EXPECT_THROW(
      dd->makeBasisState(3, {dd::BasisStates::one, dd::BasisStates::one,
                             dd::BasisStates::one}),
      std::runtime_error);
  EXPECT_THROW(dd->makeGateDD(dd::X_MAT, 3), std::runtime_error);
}

TEST(DDPackageTest, InvalidDecRef) {
  auto dd = std::make_unique<dd::Package<>>(2);
  auto e = dd->makeGateDD(dd::H_MAT, 0);
  EXPECT_DEBUG_DEATH(
      dd->decRef(e),
      "Reference count of Node must not be zero before decrement");
}

TEST(DDPackageTest, PackageReset) {
  auto dd = std::make_unique<dd::Package<>>(1);

  // one node in unique table of variable 0
  auto xGate = dd->makeGateDD(dd::X_MAT, 0);

  const auto& unique = dd->mUniqueTable.getTables();
  const auto& table = unique[0];
  auto ihash = decltype(dd->mUniqueTable)::hash(xGate.p);
  const auto* node = table[ihash];
  std::cout << ihash << ": " << reinterpret_cast<uintptr_t>(xGate.p) << "\n";
  // node should be the first in this unique table bucket
  EXPECT_EQ(node, xGate.p);
  dd->reset();
  // after clearing the tables, they should be empty
  EXPECT_EQ(table[ihash], nullptr);
  xGate = dd->makeGateDD(dd::X_MAT, 0);
  const auto* node2 = table[ihash];
  // after recreating the DD, it should receive the same node
  EXPECT_EQ(node2, node);
}

TEST(DDPackageTest, MaxRefCount) {
  auto dd = std::make_unique<dd::Package<>>(1);
  auto e = dd->makeGateDD(dd::X_MAT, 0);
  // ref count saturates at this value
  e.p->ref = std::numeric_limits<decltype(e.p->ref)>::max();
  dd->incRef(e);
  EXPECT_EQ(e.p->ref, std::numeric_limits<decltype(e.p->ref)>::max());
}

TEST(DDPackageTest, Inverse) {
  auto dd = std::make_unique<dd::Package<>>(1);
  auto x = dd->makeGateDD(dd::X_MAT, 0);
  auto xdag = dd->conjugateTranspose(x);
  EXPECT_EQ(x, xdag);
  dd->garbageCollect();
  // nothing should have been collected since the threshold is not reached
  EXPECT_EQ(dd->mUniqueTable.getNumEntries(), 1);
  dd->incRef(x);
  dd->garbageCollect(true);
  // nothing should have been collected since the lone node has a non-zero ref
  // count
  EXPECT_EQ(dd->mUniqueTable.getNumEntries(), 1);
  dd->decRef(x);
  dd->garbageCollect(true);
  // now the node should have been collected
  EXPECT_EQ(dd->mUniqueTable.getNumEntries(), 0);
}

TEST(DDPackageTest, UniqueTableAllocation) {
  auto dd = std::make_unique<dd::Package<>>(1);

  auto allocs = dd->vMemoryManager.getStats().numAllocated;
  std::cout << allocs << "\n";
  std::vector<dd::vNode*> nodes{allocs};
  // get all the nodes that are pre-allocated
  for (auto i = 0U; i < allocs; ++i) {
    nodes[i] = dd->vMemoryManager.get();
  }

  // trigger new allocation
  const auto* node = dd->vMemoryManager.get();
  ASSERT_NE(node, nullptr);
  EXPECT_EQ(dd->vMemoryManager.getStats().numAllocated,
            (1. + dd::MemoryManager<dd::vNode>::GROWTH_FACTOR) *
                static_cast<double>(allocs));

  // clearing the unique table should reduce the allocated size to the original
  // size
  dd->vMemoryManager.reset();
  EXPECT_EQ(dd->vMemoryManager.getStats().numAllocated, allocs);
}

TEST(DDPackageTest, SpecialCaseTerminal) {
  auto dd = std::make_unique<dd::Package<>>(2);
  auto one = dd::vEdge::one();
  dd::export2Dot(one, "oneColored.dot", true, false, false, false, false);
  dd::export2Dot(one, "oneClassic.dot", false, false, false, false, false);
  dd::export2Dot(one, "oneMemory.dot", true, true, false, true, false);

  const auto filenames = {
      "oneColored.dot",
      "oneClassic.dot",
      "oneMemory.dot",
  };

  for (const auto* const filename : filenames) {
    std::ifstream ifs(filename);
    ASSERT_TRUE(ifs.good());
    ASSERT_NE(ifs.peek(), std::ifstream::traits_type::eof());
    ifs.close();
    std::filesystem::remove(filename);
  }

  EXPECT_EQ(dd->vUniqueTable.lookup(one.p), one.p);

  auto zero = dd::vEdge::zero();
  EXPECT_TRUE(dd->kronecker(zero, one, 0).isZeroTerminal());
  EXPECT_TRUE(dd->kronecker(one, one, 0).isOneTerminal());

  EXPECT_EQ(one.getValueByPath(0, ""), 1.);
  EXPECT_EQ(one.getValueByIndex(0), 1.);
  EXPECT_EQ(dd::mEdge::one().getValueByIndex(0, 0, 0), 1.);

  EXPECT_EQ(dd->innerProduct(zero, zero), dd::ComplexValue(0.));
}

TEST(DDPackageTest, KroneckerProduct) {
  auto dd = std::make_unique<dd::Package<>>(2);
  auto x = dd->makeGateDD(dd::X_MAT, 0);
  auto kronecker = dd->kronecker(x, x, 1);
  EXPECT_EQ(kronecker.p->v, 1);
  EXPECT_TRUE(kronecker.p->e[0].isZeroTerminal());
  EXPECT_EQ(kronecker.p->e[0], kronecker.p->e[3]);
  EXPECT_EQ(kronecker.p->e[1], kronecker.p->e[2]);
  EXPECT_EQ(kronecker.p->e[1].p->v, 0);
  EXPECT_TRUE(kronecker.p->e[1].p->e[0].isZeroTerminal());
  EXPECT_EQ(kronecker.p->e[1].p->e[0], kronecker.p->e[1].p->e[3]);
  EXPECT_TRUE(kronecker.p->e[1].p->e[1].isOneTerminal());
  EXPECT_EQ(kronecker.p->e[1].p->e[1], kronecker.p->e[1].p->e[2]);

  auto kronecker2 = dd->kronecker(x, x, 1);
  EXPECT_EQ(kronecker, kronecker2);
}

TEST(DDPackageTest, KroneckerProductVectors) {
  auto dd = std::make_unique<dd::Package<>>(2);
  auto zeroState = dd->makeZeroState(1);
  auto kronecker = dd->kronecker(zeroState, zeroState, 1);

  auto expected = dd->makeZeroState(2);
  EXPECT_EQ(kronecker, expected);
}

TEST(DDPackageTest, KroneckerIdentityHandling) {
  auto dd = std::make_unique<dd::Package<>>(3U);
  // create a Hadamard gate on the middle qubit
  auto h = dd->makeGateDD(dd::H_MAT, 1U);
  // create a single qubit identity
  auto id = dd->makeIdent();
  // kronecker both DDs
  const auto combined = dd->kronecker(h, id, 1);
  const auto matrix = combined.getMatrix(dd->qubits());
  const auto expectedMatrix = dd::CMat{
      {dd::SQRT2_2, 0, 0, 0, dd::SQRT2_2, 0, 0, 0},
      {0, dd::SQRT2_2, 0, 0, 0, dd::SQRT2_2, 0, 0},
      {0, 0, dd::SQRT2_2, 0, 0, 0, dd::SQRT2_2, 0},
      {0, 0, 0, dd::SQRT2_2, 0, 0, 0, dd::SQRT2_2},
      {dd::SQRT2_2, 0, 0, 0, -dd::SQRT2_2, 0, 0, 0},
      {0, dd::SQRT2_2, 0, 0, 0, -dd::SQRT2_2, 0, 0},
      {0, 0, dd::SQRT2_2, 0, 0, 0, -dd::SQRT2_2, 0},
      {0, 0, 0, dd::SQRT2_2, 0, 0, 0, -dd::SQRT2_2},
  };
  EXPECT_EQ(matrix, expectedMatrix);
}

TEST(DDPackageTest, NearZeroNormalize) {
  auto dd = std::make_unique<dd::Package<>>(2);
  const dd::fp nearZero = dd::RealNumber::eps / 10;
  dd::vEdge ve{};
  ve.p = dd->vMemoryManager.get();
  ve.p->v = 1;
  ve.w = dd::Complex::one();
  std::array<dd::vCachedEdge, dd::RADIX> edges{};
  for (auto& edge : edges) {
    edge.p = dd->vMemoryManager.get();
    edge.p->v = 0;
    edge.w = nearZero;
    edge.p->e = {dd::vEdge::one(), dd::vEdge::one()};
  }
  auto veNormalizedCached =
      dd::vCachedEdge::normalize(ve.p, edges, dd->vMemoryManager, dd->cn);
  EXPECT_EQ(veNormalizedCached, dd::vCachedEdge::zero());

  std::array<dd::vEdge, dd::RADIX> edges2{};
  for (auto& edge : edges2) {
    edge.p = dd->vMemoryManager.get();
    edge.p->v = 0;
    edge.w = dd->cn.lookup(nearZero);
    edge.p->e = {dd::vEdge::one(), dd::vEdge::one()};
  }
  auto veNormalized =
      dd::vEdge::normalize(ve.p, edges2, dd->vMemoryManager, dd->cn);
  EXPECT_TRUE(veNormalized.isZeroTerminal());

  dd::mEdge me{};
  me.p = dd->mMemoryManager.get();
  me.p->v = 1;
  me.w = dd::Complex::one();
  std::array<dd::mCachedEdge, dd::NEDGE> edges3{};
  for (auto& edge : edges3) {
    edge.p = dd->mMemoryManager.get();
    edge.p->v = 0;
    edge.w = nearZero;
    edge.p->e = {dd::mEdge::one(), dd::mEdge::one(), dd::mEdge::one(),
                 dd::mEdge::one()};
  }
  auto meNormalizedCached =
      dd::mCachedEdge::normalize(me.p, edges3, dd->mMemoryManager, dd->cn);
  EXPECT_EQ(meNormalizedCached, dd::mCachedEdge::zero());

  me.p = dd->mMemoryManager.get();
  std::array<dd::mEdge, 4> edges4{};
  for (auto& edge : edges4) {
    edge.p = dd->mMemoryManager.get();
    edge.p->v = 0;
    edge.w = dd->cn.lookup(nearZero, 0.);
    edge.p->e = {dd::mEdge::one(), dd::mEdge::one(), dd::mEdge::one(),
                 dd::mEdge::one()};
  }
  auto meNormalized =
      dd::mEdge::normalize(me.p, edges4, dd->mMemoryManager, dd->cn);
  EXPECT_TRUE(meNormalized.isZeroTerminal());
}

TEST(DDPackageTest, DestructiveMeasurementAll) {
  auto dd = std::make_unique<dd::Package<>>(4);
  auto hGate0 = dd->makeGateDD(dd::H_MAT, 0);
  auto hGate1 = dd->makeGateDD(dd::H_MAT, 1);
  auto plusMatrix = dd->multiply(hGate0, hGate1);
  auto zeroState = dd->makeZeroState(2);
  auto plusState = dd->multiply(plusMatrix, zeroState);
  dd->incRef(plusState);

  std::mt19937_64 mt{0}; // NOLINT(ms

  const dd::CVec vBefore = plusState.getVector();

  ASSERT_EQ(vBefore[0], vBefore[1]);
  ASSERT_EQ(vBefore[0], vBefore[2]);
  ASSERT_EQ(vBefore[0], vBefore[3]);

  const std::string m = dd->measureAll(plusState, true, mt);

  const dd::CVec vAfter = plusState.getVector();
  const int i = std::stoi(m, nullptr, 2);

  ASSERT_EQ(vAfter[static_cast<std::size_t>(i)], 1.);
}

TEST(DDPackageTest, DestructiveMeasurementOne) {
  auto dd = std::make_unique<dd::Package<>>(4);
  auto hGate0 = dd->makeGateDD(dd::H_MAT, 0);
  auto hGate1 = dd->makeGateDD(dd::H_MAT, 1);
  auto plusMatrix = dd->multiply(hGate0, hGate1);
  auto zeroState = dd->makeZeroState(2);
  auto plusState = dd->multiply(plusMatrix, zeroState);
  dd->incRef(plusState);

  std::mt19937_64 mt{0}; // NOLINT(cert-msc51-cpp)

  const char m = dd->measureOneCollapsing(plusState, 0, true, mt);
  const dd::CVec vAfter = plusState.getVector();

  ASSERT_EQ(m, '0');
  ASSERT_EQ(vAfter[0], dd::SQRT2_2);
  ASSERT_EQ(vAfter[2], dd::SQRT2_2);
  ASSERT_EQ(vAfter[1], 0.);
  ASSERT_EQ(vAfter[3], 0.);
}

TEST(DDPackageTest, DestructiveMeasurementOneArbitraryNormalization) {
  auto dd = std::make_unique<dd::Package<>>(4);
  auto hGate0 = dd->makeGateDD(dd::H_MAT, 0);
  auto hGate1 = dd->makeGateDD(dd::H_MAT, 1);
  auto plusMatrix = dd->multiply(hGate0, hGate1);
  auto zeroState = dd->makeZeroState(2);
  auto plusState = dd->multiply(plusMatrix, zeroState);
  dd->incRef(plusState);

  std::mt19937_64 mt{0}; // NOLINT(cert-msc51-cpp)

  const char m = dd->measureOneCollapsing(plusState, 0, false, mt);
  const dd::CVec vAfter = plusState.getVector();

  ASSERT_EQ(m, '0');
  ASSERT_EQ(vAfter[0], dd::SQRT2_2);
  ASSERT_EQ(vAfter[2], dd::SQRT2_2);
  ASSERT_EQ(vAfter[1], 0.);
  ASSERT_EQ(vAfter[3], 0.);
}

TEST(DDPackageTest, ExportPolarPhaseFormatted) {
  std::ostringstream phaseString;

  // zero case
  dd::printPhaseFormatted(phaseString, 0);
  EXPECT_STREQ(phaseString.str().c_str(), "ℯ(iπ 0)");
  phaseString.str("");

  // one cases
  dd::printPhaseFormatted(phaseString, 0.5 * dd::PI);
  EXPECT_STREQ(phaseString.str().c_str(), "ℯ(iπ/2)");
  phaseString.str("");

  dd::printPhaseFormatted(phaseString, -0.5 * dd::PI);
  EXPECT_STREQ(phaseString.str().c_str(), "ℯ(-iπ/2)");
  phaseString.str("");

  dd::printPhaseFormatted(phaseString, dd::PI);
  EXPECT_STREQ(phaseString.str().c_str(), "ℯ(iπ)");
  phaseString.str("");

  dd::printPhaseFormatted(phaseString, -dd::PI);
  EXPECT_STREQ(phaseString.str().c_str(), "ℯ(-iπ)");
  phaseString.str("");

  // 1/sqrt(2) cases
  dd::printPhaseFormatted(phaseString, dd::SQRT2_2 * dd::PI);
  EXPECT_STREQ(phaseString.str().c_str(), "ℯ(iπ/√2)");
  phaseString.str("");

  dd::printPhaseFormatted(phaseString, 2 * dd::SQRT2_2 * dd::PI);
  EXPECT_STREQ(phaseString.str().c_str(), "ℯ(iπ 2/√2)");
  phaseString.str("");

  dd::printPhaseFormatted(phaseString, 0.5 * dd::SQRT2_2 * dd::PI);
  EXPECT_STREQ(phaseString.str().c_str(), "ℯ(iπ/(2√2))");
  phaseString.str("");

  dd::printPhaseFormatted(phaseString, 0.75 * dd::SQRT2_2 * dd::PI);
  EXPECT_STREQ(phaseString.str().c_str(), "ℯ(iπ 3/(4√2))");
  phaseString.str("");

  // pi cases mhhh pie
  dd::printPhaseFormatted(phaseString, dd::PI);
  EXPECT_STREQ(phaseString.str().c_str(), "ℯ(iπ)");
  phaseString.str("");

  dd::printPhaseFormatted(phaseString, 2 * dd::PI);
  EXPECT_STREQ(phaseString.str().c_str(), "ℯ(iπ 2)");
  phaseString.str("");

  dd::printPhaseFormatted(phaseString, 0.5 * dd::PI);
  EXPECT_STREQ(phaseString.str().c_str(), "ℯ(iπ/2)");
  phaseString.str("");

  dd::printPhaseFormatted(phaseString, 0.75 * dd::PI);
  EXPECT_STREQ(phaseString.str().c_str(), "ℯ(iπ 3/4)");
  phaseString.str("");

  dd::printPhaseFormatted(phaseString, 0.25 * dd::PI);
  EXPECT_STREQ(phaseString.str().c_str(), "ℯ(iπ/4)");
  phaseString.str("");

  // general case
  dd::printPhaseFormatted(phaseString, 0.12345 * dd::PI);
  EXPECT_STREQ(phaseString.str().c_str(), "ℯ(iπ 0.12345)");
  phaseString.str("");
}

TEST(DDPackageTest, BasicNumericInstabilityTest) {
  const dd::fp zero = 0.0;
  const dd::fp half = 0.5;
  const dd::fp one = 1.0;
  const dd::fp two = 2.0;

  std::cout << std::setprecision(std::numeric_limits<dd::fp>::max_digits10);

  std::cout << "The 1/sqrt(2) constant used in this package is " << dd::SQRT2_2
            << ", which is the closest floating point value to the actual "
               "value of 1/sqrt(2).\n";
  std::cout << "Computing std::sqrt(0.5) actually computes this value, i.e. "
            << std::sqrt(half) << "\n";
  EXPECT_EQ(dd::SQRT2_2, std::sqrt(half));

  std::cout << "However, computing 1/std::sqrt(2.) leads to "
            << one / std::sqrt(two)
            << ", which differs by 1 ULP from std::sqrt(0.5)\n";
  EXPECT_EQ(one / std::sqrt(two), std::nextafter(std::sqrt(half), zero));

  std::cout << "In the same fashion, computing std::sqrt(2.) leads to "
            << std::sqrt(two) << ", while computing 1/std::sqrt(0.5) leads to "
            << one / std::sqrt(half) << ", which differ by exactly 1 ULP\n";
  EXPECT_EQ(std::sqrt(two), std::nextafter(one / std::sqrt(half), two));

  std::cout << "Another inaccuracy occurs when computing 1/sqrt(2) * "
               "1/sqrt(2), which should equal to 0.5 but is off by 1 ULP: "
            << std::sqrt(half) * std::sqrt(half) << "\n";
  EXPECT_EQ(std::sqrt(half) * std::sqrt(half), std::nextafter(half, one));

  std::cout << "This inaccuracy even persists when computing std::sqrt(0.5) * "
               "std::sqrt(0.5): "
            << std::sqrt(half) * std::sqrt(half) << "\n";
  EXPECT_EQ(std::sqrt(half) * std::sqrt(half), std::nextafter(half, one));

  std::cout << "Interestingly, calculating powers of dd::SQRT2_2 can be "
               "conducted very precisely, i.e., with an error of only 1 ULP.\n";
  dd::fp accumulator = dd::SQRT2_2 * dd::SQRT2_2;
  const std::size_t nq = 64;
  for (std::size_t i = 1; i < nq; i += 2) {
    const std::size_t power = (i + 1) / 2;
    const std::size_t denom = static_cast<std::size_t>(1U) << power;
    const dd::fp target = 1. / static_cast<double>(denom);
    const dd::fp diff = std::abs(target - accumulator);
    const auto ulps = dd::ulpDistance(accumulator, target);
    std::cout << accumulator << ", numerical error: " << diff
              << ", ulps: " << ulps << "\n";
    EXPECT_EQ(ulps, 1);
    accumulator *= dd::SQRT2_2;
    accumulator *= dd::SQRT2_2;
  }
}

TEST(DDPackageTest, BasicNumericStabilityTest) {
  using limits = std::numeric_limits<dd::fp>;

  auto dd = std::make_unique<dd::Package<>>(1);
  auto tol = dd::RealNumber::eps;
  dd::ComplexNumbers::setTolerance(limits::epsilon());
  auto state = dd->makeZeroState(1);
  auto h = dd->makeGateDD(dd::H_MAT, 0);
  auto state1 = dd->multiply(h, state);
  auto z = dd->makeGateDD(dd::Z_MAT, 0);
  auto result = dd->multiply(z, state1);

  const auto topWeight = result.w.toString(false, limits::max_digits10);
  const auto leftWeight =
      result.p->e[0].w.toString(false, limits::max_digits10);
  const auto rightWeight =
      result.p->e[1].w.toString(false, limits::max_digits10);
  std::cout << topWeight << " | " << leftWeight << " | " << rightWeight << "\n";
  EXPECT_EQ(topWeight, "1");
  std::ostringstream oss{};
  oss << std::setprecision(limits::max_digits10) << dd::SQRT2_2;
  EXPECT_EQ(leftWeight, oss.str());
  oss.str("");
  oss << -dd::SQRT2_2;
  EXPECT_EQ(rightWeight, oss.str());
  // restore tolerance
  dd::ComplexNumbers::setTolerance(tol);
}

TEST(DDPackageTest, NormalizationNumericStabilityTest) {
  auto dd = std::make_unique<dd::Package<>>(1);
  for (std::size_t x = 23; x <= 50; ++x) {
    const auto lambda = dd::PI / static_cast<dd::fp>(1ULL << x);
    std::cout << std::setprecision(17) << "x: " << x << " | lambda: " << lambda
              << " | cos(lambda): " << std::cos(lambda)
              << " | sin(lambda): " << std::sin(lambda) << "\n";
    auto p = dd->makeGateDD(dd::pMat(lambda), 0);
    auto pdag = dd->makeGateDD(dd::pMat(-lambda), 0);
    auto result = dd->multiply(p, pdag);
    EXPECT_TRUE(result.isIdentity());
    dd->cUniqueTable.clear();
    dd->cMemoryManager.reset();
  }
}

TEST(DDPackageTest, FidelityOfMeasurementOutcomes) {
  auto dd = std::make_unique<dd::Package<>>(3);

  auto hGate = dd->makeGateDD(dd::H_MAT, 2);
  auto cxGate1 = dd->makeGateDD(dd::X_MAT, 2_pc, 1);
  auto cxGate2 = dd->makeGateDD(dd::X_MAT, 1_pc, 0);
  auto zeroState = dd->makeZeroState(3);

  auto ghzState = dd->multiply(
      cxGate2, dd->multiply(cxGate1, dd->multiply(hGate, zeroState)));

  dd::SparsePVec probs{};
  probs[0] = 0.5;
  probs[7] = 0.5;
  auto fidelity = dd->fidelityOfMeasurementOutcomes(ghzState, probs);
  EXPECT_NEAR(fidelity, 1.0, dd::RealNumber::eps);
}

TEST(DDPackageTest, CloseToIdentity) {
  auto dd = std::make_unique<dd::Package<>>(3);
  auto id = dd->makeIdent();
  EXPECT_TRUE(dd->isCloseToIdentity(id));
  dd::mEdge close{};
  close.p = id.p;
  close.w = dd->cn.lookup(1e-11, 0);
  auto id2 = dd->makeDDNode(
      1, std::array{id, dd::mEdge::zero(), dd::mEdge::zero(), close});
  EXPECT_TRUE(dd->isCloseToIdentity(id2));

  auto noId = dd->makeDDNode(
      1, std::array{dd::mEdge::zero(), id, dd::mEdge::zero(), close});
  EXPECT_FALSE(dd->isCloseToIdentity(noId));

  dd::mEdge notClose{};
  notClose.p = id.p;
  notClose.w = dd->cn.lookup(1e-9, 0);
  auto noId2 = dd->makeDDNode(
      1, std::array{notClose, dd::mEdge::zero(), dd::mEdge::zero(), close});
  EXPECT_FALSE(dd->isCloseToIdentity(noId2));

  auto noId3 = dd->makeDDNode(
      1, std::array{close, dd::mEdge::zero(), dd::mEdge::zero(), notClose});
  EXPECT_FALSE(dd->isCloseToIdentity(noId3));

  auto notClose2 =
      dd->makeDDNode(0, std::array{dd::mEdge::zero(), dd::mEdge::one(),
                                   dd::mEdge::one(), dd::mEdge::zero()});
  auto notClose3 = dd->makeDDNode(1, std::array{notClose2, dd::mEdge::zero(),
                                                dd::mEdge::zero(), notClose2});
  EXPECT_FALSE(dd->isCloseToIdentity(notClose3));
}

TEST(DDPackageTest, CloseToIdentityWithGarbageAtTheBeginning) {
  const dd::fp tol = 1.0E-10;
  const auto nqubits = 3U;
  auto dd = std::make_unique<dd::Package<>>(nqubits);
  auto controlledSwapGate =
      dd->makeTwoQubitGateDD(dd::SWAP_MAT, qc::Controls{1}, 0, 2);
  auto hGate = dd->makeGateDD(dd::H_MAT, 0);
  auto zGate = dd->makeGateDD(dd::Z_MAT, 2);
  auto xGate = dd->makeGateDD(dd::X_MAT, 1);
  auto controlledHGate = dd->makeGateDD(dd::H_MAT, qc::Controls{1}, 0);

  auto c1 = dd->multiply(
      controlledSwapGate,
      dd->multiply(hGate, dd->multiply(zGate, controlledSwapGate)));
  auto c2 = dd->multiply(controlledHGate, xGate);

  auto c1MultipliedWithC2 = dd->multiply(c1, dd->conjugateTranspose(c2));

  EXPECT_TRUE(dd->isCloseToIdentity(c1MultipliedWithC2, tol,
                                    {false, true, true}, false));
  EXPECT_FALSE(dd->isCloseToIdentity(c1MultipliedWithC2, tol,
                                     {false, false, true}, false));
}

TEST(DDPackageTest, CloseToIdentityWithGarbageAtTheEnd) {
  const dd::fp tol = 1.0E-10;
  const auto nqubits = 3U;
  auto dd = std::make_unique<dd::Package<>>(nqubits);

  auto controlledSwapGate =
      dd->makeTwoQubitGateDD(dd::SWAP_MAT, qc::Controls{1}, 0, 2);
  auto xGate = dd->makeGateDD(dd::X_MAT, 1);

  auto hGate2 = dd->makeGateDD(dd::H_MAT, 2);
  auto zGate2 = dd->makeGateDD(dd::Z_MAT, 0);
  auto controlledHGate2 = dd->makeGateDD(dd::H_MAT, qc::Controls{1}, 2);

  auto c3 = dd->multiply(
      controlledSwapGate,
      dd->multiply(hGate2, dd->multiply(zGate2, controlledSwapGate)));
  auto c4 = dd->multiply(controlledHGate2, xGate);

  auto c3MultipliedWithC4 = dd->multiply(c3, dd->conjugateTranspose(c4));

  EXPECT_FALSE(dd->isCloseToIdentity(c3MultipliedWithC4, tol,
                                     {false, true, true}, false));
  EXPECT_FALSE(dd->isCloseToIdentity(c3MultipliedWithC4, tol,
                                     {true, false, true}, false));
  EXPECT_TRUE(dd->isCloseToIdentity(c3MultipliedWithC4, tol,
                                    {true, true, false}, false));
}

TEST(DDPackageTest, CloseToIdentityWithGarbageInTheMiddle) {
  const dd::fp tol = 1.0E-10;
  const auto nqubits = 3U;
  auto dd = std::make_unique<dd::Package<>>(nqubits);

  auto zGate = dd->makeGateDD(dd::Z_MAT, 2);

  auto controlledSwapGate3 =
      dd->makeTwoQubitGateDD(dd::SWAP_MAT, qc::Controls{0}, 1, 2);
  auto hGate3 = dd->makeGateDD(dd::H_MAT, 1);
  auto xGate3 = dd->makeGateDD(dd::X_MAT, 0);
  auto controlledHGate3 = dd->makeGateDD(dd::H_MAT, qc::Controls{0}, 1);

  auto c5 = dd->multiply(
      controlledSwapGate3,
      dd->multiply(hGate3, dd->multiply(zGate, controlledSwapGate3)));
  auto c6 = dd->multiply(controlledHGate3, xGate3);

  auto c5MultipliedWithC6 = dd->multiply(c5, dd->conjugateTranspose(c6));

  EXPECT_FALSE(dd->isCloseToIdentity(c5MultipliedWithC6, tol,
                                     {false, true, true}, false));
  EXPECT_FALSE(dd->isCloseToIdentity(c5MultipliedWithC6, tol,
                                     {true, true, false}, false));
  EXPECT_TRUE(dd->isCloseToIdentity(c5MultipliedWithC6, tol,
                                    {true, false, true}, false));
}

TEST(DDPackageTest, dNodeMultiply) {
  // Multiply dNode with mNode (MxMxM)
  const auto nrQubits = 3U;
  auto dd =
      std::make_unique<dd::Package<dd::DensityMatrixSimulatorDDPackageConfig>>(
          nrQubits);
  // Make zero density matrix
  auto state = dd->makeZeroDensityOperator(dd->qubits());
  dd->incRef(state);
  std::vector<dd::mEdge> operations = {};
  operations.emplace_back(dd->makeGateDD(dd::H_MAT, 0));
  operations.emplace_back(dd->makeGateDD(dd::H_MAT, 1));
  operations.emplace_back(dd->makeGateDD(dd::H_MAT, 2));
  operations.emplace_back(dd->makeGateDD(dd::Z_MAT, 2));

  for (const auto& op : operations) {
    dd->applyOperationToDensity(state, op);
  }

  const auto stateDensityMatrix = state.getMatrix(dd->qubits());

  for (const auto& stateVector : stateDensityMatrix) {
    for (const auto& cValue : stateVector) {
      std::cout << "r:" << cValue.real() << " i:" << cValue.imag();
    }
    std::cout << "\n";
  }

  for (std::size_t i = 0; i < (1 << nrQubits); i++) {
    for (std::size_t j = 0; j < (1 << nrQubits); j++) {
      EXPECT_EQ(std::abs(stateDensityMatrix[i][j].imag()), 0);
      if ((i < 4 && j < 4) || (i >= 4 && j >= 4)) {
        EXPECT_TRUE(stateDensityMatrix[i][j].real() > 0);
      } else {
        EXPECT_TRUE(stateDensityMatrix[i][j].real() < 0);
      }
      EXPECT_TRUE(std::abs(std::abs(stateDensityMatrix[i][j]) - 0.125) <
                  0.000001);
    }
  }

  const auto probVector = state.getSparseProbabilityVector(nrQubits, 0.001);
  const double tolerance = 1e-10;
  for (const auto& [s, prob] : probVector) {
    std::cout << s << ": " << prob << "\n";
    EXPECT_NEAR(prob, 0.125, tolerance);
  }
}

TEST(DDPackageTest, dNodeMultiply2) {
  // Multiply dNode with mNode (MxMxM)
  const auto nrQubits = 3U;
  auto dd =
      std::make_unique<dd::Package<dd::DensityMatrixSimulatorDDPackageConfig>>(
          nrQubits);
  // Make zero density matrix
  auto state = dd->makeZeroDensityOperator(dd->qubits());
  dd->incRef(state);
  std::vector<dd::mEdge> operations = {};
  operations.emplace_back(dd->makeGateDD(dd::H_MAT, 0));
  operations.emplace_back(dd->makeGateDD(dd::H_MAT, 1));
  operations.emplace_back(dd->makeGateDD(dd::H_MAT, 2));
  operations.emplace_back(dd->makeGateDD(dd::Z_MAT, 2));

  for (const auto& op : operations) {
    dd->applyOperationToDensity(state, op);
  }
  operations[0].printMatrix(dd->qubits());

  const auto stateDensityMatrix = state.getMatrix(dd->qubits());

  for (std::size_t i = 0; i < (1 << nrQubits); i++) {
    for (std::size_t j = 0; j < (1 << nrQubits); j++) {
      EXPECT_TRUE(std::abs(stateDensityMatrix[i][j].imag()) == 0);
      if ((i < 4 && j < 4) || (i >= 4 && j >= 4)) {
        EXPECT_TRUE(stateDensityMatrix[i][j].real() > 0);
      } else {
        EXPECT_TRUE(stateDensityMatrix[i][j].real() < 0);
      }
      EXPECT_TRUE(std::abs(std::abs(stateDensityMatrix[i][j]) - 0.125) <
                  0.000001);
    }
  }
  const auto probVector = state.getSparseProbabilityVector(nrQubits, 0.001);
  const double tolerance = 1e-10;
  for (const auto& [s, prob] : probVector) {
    std::cout << s << ": " << prob << "\n";
    EXPECT_NEAR(prob, 0.125, tolerance);
  }
}

TEST(DDPackageTest, dNodeMulCache1) {
  // Make caching test with dNodes
  const auto nrQubits = 1U;
  auto dd =
      std::make_unique<dd::Package<dd::DensityMatrixSimulatorDDPackageConfig>>(
          nrQubits);
  // Make zero density matrix
  auto state = dd->makeZeroDensityOperator(nrQubits);
  dd->incRef(state);

  const auto operation = dd->makeGateDD(dd::H_MAT, 0);
  dd->applyOperationToDensity(state, operation);

  state = dd->makeZeroDensityOperator(nrQubits);
  auto& computeTable = dd->getMultiplicationComputeTable<dd::dNode>();

  const auto& densityMatrix0 =
      dd::densityFromMatrixEdge(dd->conjugateTranspose(operation));

  const auto* cachedResult =
      computeTable.lookup(state.p, densityMatrix0.p, false);
  ASSERT_NE(cachedResult, nullptr);
  ASSERT_NE(cachedResult->p, nullptr);
  state = dd->multiply(state, densityMatrix0, false);
  ASSERT_NE(state.p, nullptr);
  ASSERT_EQ(state.p, cachedResult->p);

  const auto densityMatrix1 = dd::densityFromMatrixEdge(operation);
  const auto* cachedResult1 =
      computeTable.lookup(densityMatrix1.p, state.p, true);
  ASSERT_NE(cachedResult1, nullptr);
  ASSERT_NE(cachedResult1->p, nullptr);
  const auto state2 = dd->multiply(densityMatrix1, state, true);
  ASSERT_NE(state2.p, nullptr);
  ASSERT_EQ(state2.p, cachedResult1->p);

  // try a repeated lookup
  const auto* cachedResult2 =
      computeTable.lookup(densityMatrix1.p, state.p, true);
  ASSERT_NE(cachedResult2, nullptr);
  ASSERT_NE(cachedResult2->p, nullptr);
  ASSERT_EQ(cachedResult2->p, cachedResult1->p);

  computeTable.clear();
  const auto* cachedResult3 =
      computeTable.lookup(densityMatrix1.p, state.p, true);
  ASSERT_EQ(cachedResult3, nullptr);
}

TEST(DDPackageTest, dNoiseCache) {
  // Test the flags for dnode, vnode and mnodes
  const auto nrQubits = 1U;
  auto dd = std::make_unique<dd::Package<>>(nrQubits);
  // Make zero density matrix
  const auto initialState = dd->makeZeroDensityOperator(nrQubits);
  dd->incRef(initialState);

  // nothing pre-cached
  const std::vector<dd::Qubit> target = {0};
  const auto cachedNoise = dd->densityNoise.lookup(initialState, target);
  ASSERT_EQ(cachedNoise.p, nullptr);

  auto state = initialState;
  const auto operation = dd->makeGateDD(dd::X_MAT, 0);
  dd->applyOperationToDensity(state, operation);
  dd->densityNoise.insert(initialState, state, target);

  // noise pre-cached
  const auto cachedNoise1 = dd->densityNoise.lookup(initialState, target);
  ASSERT_NE(cachedNoise1.p, nullptr);
  ASSERT_EQ(cachedNoise1.p, state.p);

  // no noise pre-cached after clear
  dd->densityNoise.clear();
  const auto cachedNoise2 = dd->densityNoise.lookup(initialState, target);
  ASSERT_EQ(cachedNoise2.p, nullptr);
}

TEST(DDPackageTest, calCulpDistance) {
  const auto nrQubits = 1U;
  auto dd = std::make_unique<dd::Package<>>(nrQubits);
  auto tmp0 = dd::ulpDistance(1 + 1e-12, 1);
  auto tmp1 = dd::ulpDistance(1, 1);
  EXPECT_TRUE(tmp0 > 0);
  EXPECT_EQ(tmp1, 0);
}

TEST(DDPackageTest, dStochCache) {
  const auto nrQubits = 4U;
  auto dd = std::make_unique<
      dd::Package<dd::StochasticNoiseSimulatorDDPackageConfig>>(nrQubits);

  std::vector<dd::mEdge> operations = {};
  operations.emplace_back(dd->makeGateDD(dd::X_MAT, 0));
  operations.emplace_back(dd->makeGateDD(dd::Z_MAT, 1));
  operations.emplace_back(dd->makeGateDD(dd::Y_MAT, 2));
  operations.emplace_back(dd->makeGateDD(dd::H_MAT, 3));

  dd->stochasticNoiseOperationCache.insert(
      0, 0, operations[0]); // insert X operations with target 0
  dd->stochasticNoiseOperationCache.insert(
      1, 1, operations[1]); // insert Z operations with target 1
  dd->stochasticNoiseOperationCache.insert(
      2, 2, operations[2]); // insert Y operations with target 2
  dd->stochasticNoiseOperationCache.insert(
      3, 3, operations[3]); // insert H operations with target 3

  for (std::uint8_t i = 0; i < 4; i++) {
    for (dd::Qubit j = 0; j < 4; j++) {
      const auto* op = dd->stochasticNoiseOperationCache.lookup(i, j);
      if (static_cast<dd::Qubit>(i) == j) {
        EXPECT_TRUE(op != nullptr && op->p == operations[i].p);
      } else {
        EXPECT_EQ(op, nullptr);
      }
    }
  }

  dd->stochasticNoiseOperationCache.clear();
  for (std::uint8_t i = 0; i < 4; i++) {
    for (dd::Qubit j = 0; j < 4; j++) {
      auto* op = dd->stochasticNoiseOperationCache.lookup(i, j);
      EXPECT_EQ(op, nullptr);
    }
  }
}

TEST(DDPackageTest, stateFromVectorBell) {
  auto dd = std::make_unique<dd::Package<>>(2);
  const auto v =
      std::vector<std::complex<dd::fp>>{dd::SQRT2_2, 0, 0, dd::SQRT2_2};
  const auto s = dd->makeStateFromVector(v);
  ASSERT_NE(s.p, nullptr);
  EXPECT_EQ(s.p->v, 1);
  EXPECT_EQ(s.p->e[0].w.r->value, dd::SQRT2_2);
  EXPECT_EQ(s.p->e[0].w.i->value, 0);
  EXPECT_EQ(s.p->e[1].w.r->value, dd::SQRT2_2);
  EXPECT_EQ(s.p->e[1].w.i->value, 0);
  ASSERT_NE(s.p->e[0].p, nullptr);
  EXPECT_EQ(s.p->e[0].p->e[0].w.r->value, 1);
  EXPECT_EQ(s.p->e[0].p->e[0].w.i->value, 0);
  EXPECT_EQ(s.p->e[0].p->e[1].w.r->value, 0);
  EXPECT_EQ(s.p->e[0].p->e[1].w.i->value, 0);
  ASSERT_NE(s.p->e[1].p, nullptr);
  EXPECT_EQ(s.p->e[1].p->e[0].w.r->value, 0);
  EXPECT_EQ(s.p->e[1].p->e[0].w.i->value, 0);
  EXPECT_EQ(s.p->e[1].p->e[1].w.r->value, 1);
  EXPECT_EQ(s.p->e[1].p->e[1].w.i->value, 0);
}

TEST(DDPackageTest, stateFromVectorEmpty) {
  auto dd = std::make_unique<dd::Package<>>(1);
  auto v = std::vector<std::complex<dd::fp>>{};
  EXPECT_TRUE(dd->makeStateFromVector(v).isOneTerminal());
}

TEST(DDPackageTest, stateFromVectorNoPowerOfTwo) {
  auto dd = std::make_unique<dd::Package<>>(3);
  auto v = std::vector<std::complex<dd::fp>>{1, 2, 3, 4, 5};
  EXPECT_THROW(dd->makeStateFromVector(v), std::invalid_argument);
}

TEST(DDPackageTest, stateFromScalar) {
  auto dd = std::make_unique<dd::Package<>>(1);
  auto s = dd->makeStateFromVector({1});
  EXPECT_TRUE(s.isTerminal());
  EXPECT_EQ(s.w.r->value, 1);
  EXPECT_EQ(s.w.i->value, 0);
}

TEST(DDPackageTest, expectationValueGlobalOperators) {
  const dd::Qubit maxQubits = 3;
  auto dd = std::make_unique<dd::Package<>>(maxQubits);
  for (dd::Qubit nrQubits = 1; nrQubits < maxQubits + 1; ++nrQubits) {
    const auto zeroState = dd->makeZeroState(nrQubits);

    // Definition global operators
    const auto singleSiteX = dd->makeGateDD(dd::X_MAT, 0);
    auto globalX = singleSiteX;

    const auto singleSiteZ = dd->makeGateDD(dd::Z_MAT, 0);
    auto globalZ = singleSiteZ;

    const auto singleSiteHadamard = dd->makeGateDD(dd::H_MAT, 0);
    auto globalHadamard = singleSiteHadamard;

    for (dd::Qubit i = 1; i < nrQubits; ++i) {
      globalX = dd->kronecker(globalX, singleSiteX, 1);
      globalZ = dd->kronecker(globalZ, singleSiteZ, 1);
      globalHadamard = dd->kronecker(globalHadamard, singleSiteHadamard, 1);
    }

    // Global Expectation values
    EXPECT_EQ(dd->expectationValue(globalX, zeroState), 0);
    EXPECT_EQ(dd->expectationValue(globalZ, zeroState), 1);
    EXPECT_EQ(dd->expectationValue(globalHadamard, zeroState),
              std::pow(dd::SQRT2_2, nrQubits));
  }
}

TEST(DDPackageTest, expectationValueLocalOperators) {
  const dd::Qubit maxQubits = 3;
  auto dd = std::make_unique<dd::Package<>>(maxQubits);
  for (dd::Qubit nrQubits = 1; nrQubits < maxQubits + 1; ++nrQubits) {
    const auto zeroState = dd->makeZeroState(nrQubits);

    // Local expectation values at each site
    for (dd::Qubit site = 0; site < nrQubits - 1; ++site) {
      // Definition local operators
      auto xGate = dd->makeGateDD(dd::X_MAT, site);
      auto zGate = dd->makeGateDD(dd::Z_MAT, site);
      auto hadamard = dd->makeGateDD(dd::H_MAT, site);

      EXPECT_EQ(dd->expectationValue(xGate, zeroState), 0);
      EXPECT_EQ(dd->expectationValue(zGate, zeroState), 1);
      EXPECT_EQ(dd->expectationValue(hadamard, zeroState), dd::SQRT2_2);
    }
  }
}

TEST(DDPackageTest, expectationValueExceptions) {
  const auto nrQubits = 2U;

  auto dd = std::make_unique<dd::Package<>>(nrQubits);
  const auto zeroState = dd->makeZeroState(nrQubits - 1);
  const auto xGate = dd->makeGateDD(dd::X_MAT, 1);

  EXPECT_ANY_THROW(dd->expectationValue(xGate, zeroState));
}

TEST(DDPackageTest, DDFromSingleQubitMatrix) {
  const auto inputMatrix =
      dd::CMat{{dd::SQRT2_2, dd::SQRT2_2}, {dd::SQRT2_2, -dd::SQRT2_2}};

  const auto nrQubits = 1U;
  const auto dd = std::make_unique<dd::Package<>>(nrQubits);
  const auto matDD = dd->makeDDFromMatrix(inputMatrix);
  const auto outputMatrix = matDD.getMatrix(dd->qubits());

  EXPECT_EQ(inputMatrix, outputMatrix);
}

TEST(DDPackageTest, DDFromTwoQubitMatrix) {
  const auto inputMatrix =
      dd::CMat{{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 0, 1}, {0, 0, 1, 0}};

  const auto nrQubits = 2U;
  const auto dd = std::make_unique<dd::Package<>>(nrQubits);
  const auto matDD = dd->makeDDFromMatrix(inputMatrix);
  const auto outputMatrix = matDD.getMatrix(dd->qubits());

  EXPECT_EQ(inputMatrix, outputMatrix);
}

TEST(DDPackageTest, DDFromTwoQubitAsymmetricalMatrix) {
  const auto inputMatrix = dd::CMat{{dd::SQRT2_2, dd::SQRT2_2, 0, 0},
                                    {-dd::SQRT2_2, dd::SQRT2_2, 0, 0},
                                    {0, 0, dd::SQRT2_2, -dd::SQRT2_2},
                                    {0, 0, dd::SQRT2_2, dd::SQRT2_2}};

  const auto nrQubits = 2U;
  const auto dd = std::make_unique<dd::Package<>>(nrQubits);
  const auto matDD = dd->makeDDFromMatrix(inputMatrix);
  const auto outputMatrix = matDD.getMatrix(dd->qubits());

  EXPECT_EQ(inputMatrix, outputMatrix);
}

TEST(DDPackageTest, DDFromThreeQubitMatrix) {
  const auto inputMatrix =
      dd::CMat{{1, 0, 0, 0, 0, 0, 0, 0}, {0, 1, 0, 0, 0, 0, 0, 0},
               {0, 0, 1, 0, 0, 0, 0, 0}, {0, 0, 0, 1, 0, 0, 0, 0},
               {0, 0, 0, 0, 1, 0, 0, 0}, {0, 0, 0, 0, 0, 1, 0, 0},
               {0, 0, 0, 0, 0, 0, 0, 1}, {0, 0, 0, 0, 0, 0, 1, 0}};

  const auto nrQubits = 3U;
  const auto dd = std::make_unique<dd::Package<>>(nrQubits);
  const auto matDD = dd->makeDDFromMatrix(inputMatrix);

  const auto outputMatrix = matDD.getMatrix(dd->qubits());

  EXPECT_EQ(inputMatrix, outputMatrix);
}

TEST(DDPackageTest, DDFromEmptyMatrix) {
  const auto inputMatrix = dd::CMat{};

  const auto nrQubits = 3U;
  const auto dd = std::make_unique<dd::Package<>>(nrQubits);
  EXPECT_TRUE(dd->makeDDFromMatrix(inputMatrix).isOneTerminal());
}

TEST(DDPackageTest, DDFromNonPowerOfTwoMatrix) {
  auto inputMatrix = dd::CMat{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}};

  const auto nrQubits = 3U;
  const auto dd = std::make_unique<dd::Package<>>(nrQubits);
  EXPECT_THROW(dd->makeDDFromMatrix(inputMatrix), std::invalid_argument);
}

TEST(DDPackageTest, DDFromNonSquareMatrix) {
  const auto inputMatrix = dd::CMat{{0, 1, 2, 3}, {4, 5, 6, 7}};

  const auto nrQubits = 3U;
  const auto dd = std::make_unique<dd::Package<>>(nrQubits);
  EXPECT_THROW(dd->makeDDFromMatrix(inputMatrix), std::invalid_argument);
}

TEST(DDPackageTest, DDFromSingleElementMatrix) {
  const auto inputMatrix = dd::CMat{{1}};

  const auto nrQubits = 1U;
  const auto dd = std::make_unique<dd::Package<>>(nrQubits);

  EXPECT_TRUE(dd->makeDDFromMatrix(inputMatrix).isOneTerminal());
}

TEST(DDPackageTest, TwoQubitControlledGateDDConstruction) {
  const auto nrQubits = 5U;
  const auto dd = std::make_unique<dd::Package<>>(nrQubits);

  const auto gateMatrices = std::vector{std::pair{dd::X_MAT, dd::CX_MAT},
                                        std::pair{dd::Z_MAT, dd::CZ_MAT}};

  // For every combination of control and target, test that the DD created by
  // makeTwoQubitGateDD is equal to the DD created by makeGateDD. This should
  // cover every scenario of the makeTwoQubitGateDD function.
  for (const auto& [gateMatrix, controlledGateMatrix] : gateMatrices) {
    for (dd::Qubit control = 0; control < nrQubits; ++control) {
      for (dd::Qubit target = 0; target < nrQubits; ++target) {
        if (control == target) {
          continue;
        }
        const auto controlledGateDD =
            dd->makeTwoQubitGateDD(controlledGateMatrix, control, target);
        const auto gateDD = dd->makeGateDD(
            gateMatrix, qc::Control{static_cast<qc::Qubit>(control)}, target);
        EXPECT_EQ(controlledGateDD, gateDD);
      }
    }
  }
}

TEST(DDPackageTest, TwoQubitControlledGateDDConstructionNegativeControls) {
  const auto nrQubits = 5U;
  const auto dd = std::make_unique<dd::Package<>>(nrQubits);

  const auto gateMatrices = std::vector{std::pair{dd::X_MAT, dd::CX_MAT},
                                        std::pair{dd::Z_MAT, dd::CZ_MAT}};

  // For every combination of controls, control type, and target, test that the
  // DD created by makeTwoQubitGateDD is equal to the DD created by makeGateDD.
  // This should cover every scenario of the makeTwoQubitGateDD function.
  for (const auto& [gateMatrix, controlledGateMatrix] : gateMatrices) {
    for (dd::Qubit control0 = 0; control0 < nrQubits; ++control0) {
      for (dd::Qubit control1 = 0; control1 < nrQubits; ++control1) {
        if (control0 == control1) {
          continue;
        }
        for (dd::Qubit target = 0; target < nrQubits; ++target) {
          if (control0 == target || control1 == target) {
            continue;
          }
          for (const auto controlType :
               {qc::Control::Type::Pos, qc::Control::Type::Neg}) {
            const auto controlledGateDD = dd->makeTwoQubitGateDD(
                controlledGateMatrix, qc::Controls{{control0, controlType}},
                control1, target);
            const auto gateDD = dd->makeGateDD(
                gateMatrix, qc::Controls{{control0, controlType}, control1},
                target);
            EXPECT_EQ(controlledGateDD, gateDD);
          }
        }
      }
    }
  }
}

TEST(DDPackageTest, SWAPGateDDConstruction) {
  const auto nrQubits = 5U;
  const auto dd = std::make_unique<dd::Package<>>(nrQubits);

  for (dd::Qubit control = 0; control < nrQubits; ++control) {
    for (dd::Qubit target = 0; target < nrQubits; ++target) {
      if (control == target) {
        continue;
      }
      const auto swapGateDD =
          dd->makeTwoQubitGateDD(dd::SWAP_MAT, control, target);

      auto c = qc::Controls{control};
      auto gateDD = dd->makeGateDD(dd::X_MAT, c, target);
      c.erase(qc::Control{control});
      c.insert(qc::Control{target});
      gateDD = dd->multiply(
          gateDD, dd->multiply(dd->makeGateDD(dd::X_MAT, c, control), gateDD));

      EXPECT_EQ(swapGateDD, gateDD);
    }
  }
}

TEST(DDPackageTest, PeresGateDDConstruction) {
  const auto nrQubits = 5U;
  const auto dd = std::make_unique<dd::Package<>>(nrQubits);

  for (dd::Qubit control = 0; control < nrQubits; ++control) {
    for (dd::Qubit target = 0; target < nrQubits; ++target) {
      if (control == target) {
        continue;
      }
      const auto peresGateDD =
          dd->makeTwoQubitGateDD(dd::PERES_MAT, control, target);

      const auto c = qc::Controls{control};
      auto gateDD = dd->makeGateDD(dd::X_MAT, c, target);
      gateDD = dd->multiply(dd->makeGateDD(dd::X_MAT, control), gateDD);

      EXPECT_EQ(peresGateDD, gateDD);

      const auto peresInvDD =
          dd->makeTwoQubitGateDD(dd::PERESDG_MAT, control, target);

      auto gateInvDD = dd->makeGateDD(dd::X_MAT, control);
      gateInvDD = dd->multiply(dd->makeGateDD(dd::X_MAT, c, target), gateInvDD);

      EXPECT_EQ(peresInvDD, gateInvDD);
    }
  }
}

TEST(DDPackageTest, iSWAPGateDDConstruction) {
  const auto nrQubits = 5U;
  const auto dd = std::make_unique<dd::Package<>>(nrQubits);

  for (dd::Qubit control = 0; control < nrQubits; ++control) {
    for (dd::Qubit target = 0; target < nrQubits; ++target) {
      if (control == target) {
        continue;
      }
      const auto iswapGateDD =
          dd->makeTwoQubitGateDD(dd::ISWAP_MAT, control, target);

      auto gateDD = dd->makeGateDD(dd::S_MAT, target); // S q[1]
      gateDD =
          dd->multiply(gateDD, dd->makeGateDD(dd::S_MAT, control)); // S q[0]
      gateDD =
          dd->multiply(gateDD, dd->makeGateDD(dd::H_MAT, control)); // H q[0]
      auto c = qc::Controls{control};
      gateDD = dd->multiply(gateDD, dd->makeGateDD(dd::X_MAT, c,
                                                   target)); // CX q[0], q[1]
      c.erase(qc::Control{control});
      c.insert(qc::Control{target});
      gateDD = dd->multiply(gateDD, dd->makeGateDD(dd::X_MAT, c,
                                                   control)); // CX q[1], q[0]
      gateDD =
          dd->multiply(gateDD, dd->makeGateDD(dd::H_MAT, target)); // H q[1]

      EXPECT_EQ(iswapGateDD, gateDD);

      const auto iswapInvGateDD =
          dd->makeTwoQubitGateDD(dd::ISWAPDG_MAT, control, target);

      auto gateInvDD = dd->makeGateDD(dd::H_MAT, target); // H q[1]
      c = qc::Controls{target};
      gateInvDD =
          dd->multiply(gateInvDD, dd->makeGateDD(dd::X_MAT, c,
                                                 control)); // CX q[1], q[0]
      c.erase(qc::Control{target});
      c.insert(qc::Control{control});
      gateInvDD =
          dd->multiply(gateInvDD, dd->makeGateDD(dd::X_MAT, c,
                                                 target)); // CX q[0], q[1]
      gateInvDD =
          dd->multiply(gateInvDD, dd->makeGateDD(dd::H_MAT, control)); // H q[0]
      gateInvDD = dd->multiply(gateInvDD, dd->makeGateDD(dd::SDG_MAT,
                                                         control)); // Sdag q[0]
      gateInvDD = dd->multiply(gateInvDD, dd->makeGateDD(dd::SDG_MAT,
                                                         target)); // Sdag q[1]

      EXPECT_EQ(iswapInvGateDD, gateInvDD);
    }
  }
}

TEST(DDPackageTest, DCXGateDDConstruction) {
  const auto nrQubits = 5U;
  const auto dd = std::make_unique<dd::Package<>>(nrQubits);

  for (dd::Qubit control = 0; control < nrQubits; ++control) {
    for (dd::Qubit target = 0; target < nrQubits; ++target) {
      if (control == target) {
        continue;
      }
      const auto dcxGateDD =
          dd->makeTwoQubitGateDD(dd::DCX_MAT, control, target);

      auto c = qc::Controls{};
      c.insert(qc::Control{control});
      auto gateDD = dd->makeGateDD(dd::X_MAT, c, target);
      c.erase(qc::Control{control});
      c.insert(qc::Control{target});
      gateDD = dd->multiply(gateDD, dd->makeGateDD(dd::X_MAT, c, control));

      EXPECT_EQ(dcxGateDD, gateDD);
    }
  }
}

TEST(DDPackageTest, RZZGateDDConstruction) {
  const auto nrQubits = 5U;
  const auto dd = std::make_unique<dd::Package<>>(nrQubits);

  const auto params = {0., dd::PI_2, dd::PI, 2 * dd::PI};

  for (dd::Qubit control = 0; control < nrQubits; ++control) {
    for (dd::Qubit target = 0; target < nrQubits; ++target) {
      if (control == target) {
        continue;
      }
      for (const auto& param : params) {
        const auto rzzGateDD =
            dd->makeTwoQubitGateDD(dd::rzzMat(param), control, target);

        auto c = qc::Controls{};
        c.insert(qc::Control{control});
        auto gateDD = dd->makeGateDD(dd::X_MAT, c, target);
        c.erase(qc::Control{control});
        gateDD =
            dd->multiply(gateDD, dd->makeGateDD(dd::rzMat(param), c, target));
        c.insert(qc::Control{control});
        gateDD = dd->multiply(gateDD, dd->makeGateDD(dd::X_MAT, c, target));

        EXPECT_EQ(rzzGateDD, gateDD);
      }
    }
  }

  auto identity = dd->makeIdent();
  auto rzzZero = dd->makeTwoQubitGateDD(dd::rzzMat(0.), 0, 1);
  EXPECT_EQ(rzzZero, identity);

  auto rzzTwoPi = dd->makeTwoQubitGateDD(dd::rzzMat(2 * dd::PI), 0, 1);
  EXPECT_EQ(rzzTwoPi.p, identity.p);
  EXPECT_EQ(dd::RealNumber::val(rzzTwoPi.w.r), -1.);

  auto rzzPi = dd->makeTwoQubitGateDD(dd::rzzMat(dd::PI), 0, 1);
  auto zz = dd->makeGateDD(dd::Z_MAT, qc::Controls{}, 0);
  zz = dd->multiply(zz, dd->makeGateDD(dd::Z_MAT, qc::Controls{}, 1));
  EXPECT_EQ(rzzPi.p, zz.p);
}

TEST(DDPackageTest, RYYGateDDConstruction) {
  const auto nrQubits = 5U;
  const auto dd = std::make_unique<dd::Package<>>(nrQubits);

  const auto params = {0., dd::PI_2, dd::PI};

  for (dd::Qubit control = 0; control < nrQubits; ++control) {
    for (dd::Qubit target = 0; target < nrQubits; ++target) {
      if (control == target) {
        continue;
      }
      for (const auto& param : params) {
        const auto ryyGateDD =
            dd->makeTwoQubitGateDD(dd::ryyMat(param), control, target);

        // no controls are necessary on the RX gates since they cancel if the
        // controls are 0.
        auto gateDD = dd->makeGateDD(dd::rxMat(dd::PI_2), control);
        gateDD =
            dd->multiply(gateDD, dd->makeGateDD(dd::rxMat(dd::PI_2), target));
        gateDD = dd->multiply(
            gateDD, dd->makeTwoQubitGateDD(dd::rzzMat(param), control, target));
        gateDD =
            dd->multiply(gateDD, dd->makeGateDD(dd::rxMat(-dd::PI_2), target));
        gateDD =
            dd->multiply(gateDD, dd->makeGateDD(dd::rxMat(-dd::PI_2), control));

        EXPECT_EQ(ryyGateDD, gateDD);
      }
    }
  }

  auto identity = dd->makeIdent();
  auto ryyZero = dd->makeTwoQubitGateDD(dd::ryyMat(0.), 0, 1);
  EXPECT_EQ(ryyZero, identity);

  auto ryyPi = dd->makeTwoQubitGateDD(dd::ryyMat(dd::PI), 0, 1);
  auto yy = dd->makeGateDD(dd::Y_MAT, qc::Controls{}, 0);
  yy = dd->multiply(yy, dd->makeGateDD(dd::Y_MAT, qc::Controls{}, 1));
  EXPECT_EQ(ryyPi.p, yy.p);
}

TEST(DDPackageTest, RXXGateDDConstruction) {
  const auto nrQubits = 5U;
  const auto dd = std::make_unique<dd::Package<>>(nrQubits);

  const auto params = {0., dd::PI_2, dd::PI};

  for (dd::Qubit control = 0; control < nrQubits; ++control) {
    for (dd::Qubit target = 0; target < nrQubits; ++target) {
      if (control == target) {
        continue;
      }
      for (const auto& param : params) {
        const auto rxxGateDD =
            dd->makeTwoQubitGateDD(dd::rxxMat(param), control, target);

        auto gateDD = dd->makeGateDD(dd::H_MAT, control);
        gateDD = dd->multiply(gateDD, dd->makeGateDD(dd::H_MAT, target));
        gateDD = dd->multiply(
            gateDD, dd->makeTwoQubitGateDD(dd::rzzMat(param), control, target));
        gateDD = dd->multiply(gateDD, dd->makeGateDD(dd::H_MAT, target));
        gateDD = dd->multiply(gateDD, dd->makeGateDD(dd::H_MAT, control));

        EXPECT_EQ(rxxGateDD, gateDD);
      }
    }
  }

  auto identity = dd->makeIdent();
  auto rxxZero = dd->makeTwoQubitGateDD(dd::rxxMat(0.), 0, 1);
  EXPECT_EQ(rxxZero, identity);

  auto rxxPi = dd->makeTwoQubitGateDD(dd::rxxMat(dd::PI), 0, 1);
  auto xx = dd->makeGateDD(dd::X_MAT, qc::Controls{}, 0);
  xx = dd->multiply(xx, dd->makeGateDD(dd::X_MAT, qc::Controls{}, 1));
  EXPECT_EQ(rxxPi.p, xx.p);
}

TEST(DDPackageTest, RZXGateDDConstruction) {
  const auto nrQubits = 5U;
  const auto dd = std::make_unique<dd::Package<>>(nrQubits);

  const auto params = {0., dd::PI_2, dd::PI};

  for (dd::Qubit control = 0; control < nrQubits; ++control) {
    for (dd::Qubit target = 0; target < nrQubits; ++target) {
      if (control == target) {
        continue;
      }
      for (const auto& param : params) {
        const auto rzxGateDD =
            dd->makeTwoQubitGateDD(dd::rzxMat(param), control, target);

        // no controls are necessary on the H gates since they cancel if the
        // controls are 0.
        auto gateDD = dd->makeGateDD(dd::H_MAT, target);
        gateDD = dd->multiply(
            gateDD, dd->makeTwoQubitGateDD(dd::rzzMat(param), control, target));
        gateDD = dd->multiply(gateDD, dd->makeGateDD(dd::H_MAT, target));

        EXPECT_EQ(rzxGateDD, gateDD);
      }
    }
  }

  auto identity = dd->makeIdent();
  auto rzxZero = dd->makeTwoQubitGateDD(dd::rzxMat(0.), 0, 1);
  EXPECT_EQ(rzxZero, identity);

  auto rzxPi = dd->makeTwoQubitGateDD(dd::rzxMat(dd::PI), 0, 1);
  auto zx = dd->makeGateDD(dd::Z_MAT, qc::Controls{}, 0);
  zx = dd->multiply(zx, dd->makeGateDD(dd::X_MAT, qc::Controls{}, 1));
  EXPECT_EQ(rzxPi.p, zx.p);
}

TEST(DDPackageTest, ECRGateDDConstruction) {
  const auto nrQubits = 5U;
  const auto dd = std::make_unique<dd::Package<>>(nrQubits);

  for (dd::Qubit control = 0; control < nrQubits; ++control) {
    for (dd::Qubit target = 0; target < nrQubits; ++target) {
      if (control == target) {
        continue;
      }

      const auto ecrGateDD =
          dd->makeTwoQubitGateDD(dd::ECR_MAT, control, target);

      auto gateDD =
          dd->makeTwoQubitGateDD(dd::rzxMat(-dd::PI_4), control, target);
      gateDD = dd->multiply(gateDD, dd->makeGateDD(dd::X_MAT, control));
      gateDD = dd->multiply(gateDD, dd->makeTwoQubitGateDD(dd::rzxMat(dd::PI_4),
                                                           control, target));

      EXPECT_EQ(ecrGateDD, gateDD);
    }
  }
}

TEST(DDPackageTest, XXMinusYYGateDDConstruction) {
  const auto nrQubits = 5U;
  const auto dd = std::make_unique<dd::Package<>>(nrQubits);

  const auto thetaAngles = {0., dd::PI_2, dd::PI};
  const auto betaAngles = {0., dd::PI_2, dd::PI};

  for (dd::Qubit control = 0; control < nrQubits; ++control) {
    for (dd::Qubit target = 0; target < nrQubits; ++target) {
      if (control == target) {
        continue;
      }

      for (const auto& theta : thetaAngles) {
        for (const auto& beta : betaAngles) {
          const auto xxMinusYYGateDD = dd->makeTwoQubitGateDD(
              dd::xxMinusYYMat(theta, beta), control, target);

          auto gateDD = dd->makeGateDD(dd::rzMat(-beta), target);
          gateDD = dd->multiply(gateDD,
                                dd->makeGateDD(dd::rzMat(-dd::PI_2), control));
          gateDD = dd->multiply(gateDD, dd->makeGateDD(dd::SX_MAT, control));
          gateDD = dd->multiply(gateDD,
                                dd->makeGateDD(dd::rzMat(dd::PI_2), control));
          gateDD = dd->multiply(gateDD, dd->makeGateDD(dd::S_MAT, target));
          gateDD = dd->multiply(
              gateDD, dd->makeGateDD(dd::X_MAT, qc::Control{control}, target));
          // only the following two gates need to be controlled by the controls
          // since the other gates cancel if the controls are 0.
          gateDD =
              dd->multiply(gateDD, dd->makeGateDD(dd::ryMat(-theta / 2.),
                                                  qc::Controls{}, control));
          gateDD = dd->multiply(gateDD, dd->makeGateDD(dd::ryMat(theta / 2.),
                                                       qc::Controls{}, target));

          gateDD = dd->multiply(
              gateDD, dd->makeGateDD(dd::X_MAT, qc::Control{control}, target));
          gateDD = dd->multiply(gateDD, dd->makeGateDD(dd::SDG_MAT, target));
          gateDD = dd->multiply(gateDD,
                                dd->makeGateDD(dd::rzMat(-dd::PI_2), control));
          gateDD = dd->multiply(gateDD, dd->makeGateDD(dd::SXDG_MAT, control));
          gateDD = dd->multiply(gateDD,
                                dd->makeGateDD(dd::rzMat(dd::PI_2), control));
          gateDD =
              dd->multiply(gateDD, dd->makeGateDD(dd::rzMat(beta), target));

          EXPECT_EQ(xxMinusYYGateDD, gateDD);
        }
      }
    }
  }
}

TEST(DDPackageTest, XXPlusYYGateDDConstruction) {
  const auto nrQubits = 5U;
  const auto dd = std::make_unique<dd::Package<>>(nrQubits);

  const auto thetaAngles = {0., dd::PI_2, dd::PI};
  const auto betaAngles = {0., dd::PI_2, dd::PI};

  for (dd::Qubit control = 0; control < nrQubits; ++control) {
    for (dd::Qubit target = 0; target < nrQubits; ++target) {
      if (control == target) {
        continue;
      }

      for (const auto& theta : thetaAngles) {
        for (const auto& beta : betaAngles) {
          const auto xxPlusYYGateDD = dd->makeTwoQubitGateDD(
              dd::xxPlusYYMat(theta, beta), control, target);
          auto gateDD = dd->makeGateDD(dd::rzMat(beta), target);
          gateDD = dd->multiply(gateDD,
                                dd->makeGateDD(dd::rzMat(-dd::PI_2), control));
          gateDD = dd->multiply(gateDD, dd->makeGateDD(dd::SX_MAT, control));
          gateDD = dd->multiply(gateDD,
                                dd->makeGateDD(dd::rzMat(dd::PI_2), control));
          gateDD = dd->multiply(gateDD, dd->makeGateDD(dd::S_MAT, target));
          gateDD = dd->multiply(
              gateDD, dd->makeGateDD(dd::X_MAT, qc::Control{control}, target));
          // only the following two gates need to be controlled by the controls
          // since the other gates cancel if the controls are 0.
          gateDD =
              dd->multiply(gateDD, dd->makeGateDD(dd::ryMat(theta / 2.),
                                                  qc::Controls{}, control));
          gateDD = dd->multiply(gateDD, dd->makeGateDD(dd::ryMat(theta / 2.),
                                                       qc::Controls{}, target));

          gateDD = dd->multiply(
              gateDD, dd->makeGateDD(dd::X_MAT, qc::Control{control}, target));
          gateDD = dd->multiply(gateDD, dd->makeGateDD(dd::SDG_MAT, target));
          gateDD = dd->multiply(gateDD,
                                dd->makeGateDD(dd::rzMat(-dd::PI_2), control));
          gateDD = dd->multiply(gateDD, dd->makeGateDD(dd::SXDG_MAT, control));
          gateDD = dd->multiply(gateDD,
                                dd->makeGateDD(dd::rzMat(dd::PI_2), control));
          gateDD =
              dd->multiply(gateDD, dd->makeGateDD(dd::rzMat(-beta), target));

          EXPECT_EQ(xxPlusYYGateDD, gateDD);
        }
      }
    }
  }
}

TEST(DDPackageTest, TwoQubitGateCreationFailure) {
  const auto nrQubits = 1U;
  const auto dd = std::make_unique<dd::Package<>>(nrQubits);

  EXPECT_THROW(dd->makeTwoQubitGateDD(dd::CX_MAT, 0, 1), std::runtime_error);
}

TEST(DDPackageTest, InnerProductTopNodeConjugation) {
  // Test comes from experimental results
  // 2 qubit state is rotated Rxx(-2) equivalent to
  // Ising model evolution up to a time T=1
  const auto nrQubits = 2U;
  const auto dd = std::make_unique<dd::Package<>>(nrQubits);
  const auto zeroState = dd->makeZeroState(nrQubits);
  const auto rxx = dd->makeTwoQubitGateDD(dd::rxxMat(-2), 0, 1);
  const auto op = dd->makeGateDD(dd::Z_MAT, 0);

  const auto evolvedState = dd->multiply(rxx, zeroState);

  // Actual evolution results in approximately -0.416
  // If the top node in the inner product is not conjugated properly,
  // it will result in +0.416.
  EXPECT_NEAR(dd->expectationValue(op, evolvedState), -0.416, 0.001);
}

/**
 * @brief This is a regression test for a long lasting memory leak in the DD
 * package.
 * @details The memory leak was caused by a bug in the normalization routine
 * which was not properly returning a node to the memory manager. This occurred
 * whenever the multiplication of two DDs resulted in a zero terminal.
 */
TEST(DDPackageTest, DDNodeLeakRegressionTest) {
  const auto nqubits = 1U;
  auto dd = std::make_unique<dd::Package<>>(nqubits);

  auto dd1 = dd->makeGateDD(dd::MEAS_ZERO_MAT, 0U);
  auto dd2 = dd->makeGateDD(dd::MEAS_ONE_MAT, 0U);
  dd->multiply(dd1, dd2);
  dd->garbageCollect(true);
  EXPECT_EQ(dd->mMemoryManager.getStats().numUsed, 0U);
}

/**
 * @brief This is a regression test for a compute table bug with terminals.
 * @details The bug was caused by the assumption that `result.p == nullptr`
 * indicates that the lookup was unsuccessful. However, this is not the case
 * anymore since terminal DD nodes were replaced by a `nullptr` pointer.
 */
TEST(DDPackageTest, CTPerformanceRegressionTest) {
  const auto nqubits = 1U;
  auto dd = std::make_unique<dd::Package<>>(nqubits);

  auto dd1 = dd->makeGateDD(dd::MEAS_ZERO_MAT, 0U);
  auto dd2 = dd->makeGateDD(dd::MEAS_ONE_MAT, 0U);
  const auto repetitions = 10U;
  for (auto i = 0U; i < repetitions; ++i) {
    dd->multiply(dd1, dd2);
  }
  auto& ct = dd->matrixMatrixMultiplication;
  EXPECT_EQ(ct.getStats().lookups, repetitions);
  EXPECT_EQ(ct.getStats().hits, repetitions - 1U);

  // This additional check makes sure that no nodes are leaked.
  dd->garbageCollect(true);
  EXPECT_EQ(dd->mMemoryManager.getStats().numUsed, 0U);
}

TEST(DDPackageTest, DataStructureStatistics) {
  const auto nqubits = 1U;
  auto dd = std::make_unique<dd::Package<>>(nqubits);
  const auto stats = dd::getDataStructureStatistics(dd.get());

  EXPECT_EQ(stats["vNode"]["size_B"], 64U);
  EXPECT_EQ(stats["vNode"]["alignment_B"], 8U);
  EXPECT_EQ(stats["mNode"]["size_B"], 112U);
  EXPECT_EQ(stats["mNode"]["alignment_B"], 8U);
  EXPECT_EQ(stats["dNode"]["size_B"], 112U);
  EXPECT_EQ(stats["dNode"]["alignment_B"], 8U);
  EXPECT_EQ(stats["vEdge"]["size_B"], 24U);
  EXPECT_EQ(stats["vEdge"]["alignment_B"], 8U);
  EXPECT_EQ(stats["mEdge"]["size_B"], 24U);
  EXPECT_EQ(stats["mEdge"]["alignment_B"], 8U);
  EXPECT_EQ(stats["dEdge"]["size_B"], 24U);
  EXPECT_EQ(stats["dEdge"]["alignment_B"], 8U);
  EXPECT_EQ(stats["RealNumber"]["size_B"], 24U);
  EXPECT_EQ(stats["RealNumber"]["alignment_B"], 8U);
}

TEST(DDPackageTest, DDStatistics) {
  const auto nqubits = 2U;
  auto dd = std::make_unique<dd::Package<>>(nqubits);
  const auto dummyGate = dd->makeGateDD(dd::X_MAT, 0U);
  EXPECT_NE(dummyGate.p, nullptr);
  const auto stats = dd::getStatistics(dd.get(), true);

  std::cout << stats.dump(2) << "\n";
  EXPECT_TRUE(stats.contains("vector"));
  ASSERT_TRUE(stats.contains("matrix"));
  EXPECT_TRUE(stats.contains("density_matrix"));
  EXPECT_TRUE(stats.contains("real_numbers"));
  EXPECT_TRUE(stats.contains("compute_tables"));
  const auto& matrixStats = stats["matrix"];
  ASSERT_TRUE(matrixStats.contains("unique_table"));
  const auto& uniqueTableStats = matrixStats["unique_table"];
  EXPECT_TRUE(uniqueTableStats.contains("0"));
  EXPECT_TRUE(uniqueTableStats.contains("1"));
  EXPECT_TRUE(uniqueTableStats.contains("total"));
  ASSERT_TRUE(uniqueTableStats["0"].contains("num_buckets"));
  EXPECT_GT(uniqueTableStats["0"]["num_buckets"], 0);
  ASSERT_TRUE(uniqueTableStats["total"].contains("num_buckets"));
  EXPECT_GT(uniqueTableStats["total"]["num_buckets"], 0);
}

TEST(DDPackageTest, ReduceAncillaRegression) {
  auto dd = std::make_unique<dd::Package<>>(2);
  const auto inputMatrix =
      dd::CMat{{1, 1, 1, 1}, {1, -1, 1, -1}, {1, 1, -1, -1}, {1, -1, -1, 1}};
  auto inputDD = dd->makeDDFromMatrix(inputMatrix);
  dd->incRef(inputDD);
  const auto outputDD = dd->reduceAncillae(inputDD, {true, false});

  const auto outputMatrix = outputDD.getMatrix(dd->qubits());
  const auto expected =
      dd::CMat{{1, 0, 1, 0}, {1, 0, 1, 0}, {1, 0, -1, 0}, {1, 0, -1, 0}};

  EXPECT_EQ(outputMatrix, expected);
}

TEST(DDPackageTest, VectorConjugate) {
  auto dd = std::make_unique<dd::Package<>>(2);

  EXPECT_EQ(dd->conjugate(dd::vEdge::zero()), dd::vEdge::zero());

  EXPECT_EQ(dd->conjugate(dd::vEdge::one()), dd::vEdge::one());
  EXPECT_EQ(dd->conjugate(dd::vEdge::terminal(dd->cn.lookup(0., 1.))),
            dd::vEdge::terminal(dd->cn.lookup(0., -1.)));

  dd::CVec vec{{0., 0.5},
               {0.5 * dd::SQRT2_2, 0.5 * dd::SQRT2_2},
               {0., -0.5},
               {-0.5 * dd::SQRT2_2, -0.5 * dd::SQRT2_2}};

  auto vecDD = dd->makeStateFromVector(vec);
  std::cout << "Vector:\n";
  vecDD.printVector();
  auto conjVecDD = dd->conjugate(vecDD);
  std::cout << "Conjugated vector:\n";
  conjVecDD.printVector();

  auto conjVec = conjVecDD.getVector();
  const dd::fp tolerance = 1e-10;
  for (auto i = 0U; i < vec.size(); ++i) {
    EXPECT_NEAR(conjVec[i].real(), vec[i].real(), tolerance);
    EXPECT_NEAR(conjVec[i].imag(), -vec[i].imag(), tolerance);
  }
}

TEST(DDPackageTest, ReduceAncillaIdentity) {
  auto dd = std::make_unique<dd::Package<>>(2);
  auto inputDD = dd->makeIdent();
  const auto outputDD = dd->reduceAncillae(inputDD, {true, true});

  const auto outputMatrix = outputDD.getMatrix(dd->qubits());
  const auto expected =
      dd::CMat{{1, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};

  EXPECT_EQ(outputMatrix, expected);
}

TEST(DDPackageTest, ReduceAnicllaIdentityBeforeFirstNode) {
  auto dd = std::make_unique<dd::Package<>>(2);
  auto xGate = dd->makeGateDD(dd::X_MAT, 0);
  auto outputDD = dd->reduceAncillae(xGate, {false, true});

  auto outputMatrix = outputDD.getMatrix(dd->qubits());
  auto expected =
      dd::CMat{{0, 1, 0, 0}, {1, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};
  EXPECT_EQ(outputMatrix, expected);
}

TEST(DDPackageTest, ReduceAnicllaIdentityAfterLastNode) {
  auto dd = std::make_unique<dd::Package<>>(2);
  auto xGate = dd->makeGateDD(dd::X_MAT, 1);
  dd->incRef(xGate);
  auto outputDD = dd->reduceAncillae(xGate, {true, false});

  auto outputMatrix = outputDD.getMatrix(dd->qubits());
  auto expected =
      dd::CMat{{0, 0, 1, 0}, {0, 0, 0, 0}, {1, 0, 0, 0}, {0, 0, 0, 0}};
  EXPECT_EQ(outputMatrix, expected);
}

TEST(DDPackageTest, ReduceAncillaIdentityBetweenTwoNodes) {
  auto dd = std::make_unique<dd::Package<>>(3);
  auto xGate0 = dd->makeGateDD(dd::X_MAT, 0);
  auto xGate2 = dd->makeGateDD(dd::X_MAT, 2);
  auto state = dd->multiply(xGate0, xGate2);

  dd->incRef(state);
  auto outputDD = dd->reduceAncillae(state, {false, true, false});
  auto outputMatrix = outputDD.getMatrix(dd->qubits());
  auto expected = dd::CMat{{0, 0, 0, 0, 0, 1, 0, 0}, {0, 0, 0, 0, 1, 0, 0, 0},
                           {0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0},
                           {0, 1, 0, 0, 0, 0, 0, 0}, {1, 0, 0, 0, 0, 0, 0, 0},
                           {0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0}};
  EXPECT_EQ(outputMatrix, expected);
}

TEST(DDPackageTest, ReduceGarbageIdentity) {
  auto dd = std::make_unique<dd::Package<>>(2);
  auto inputDD = dd->makeIdent();
  auto outputDD = dd->reduceGarbage(inputDD, {true, true});

  auto outputMatrix = outputDD.getMatrix(dd->qubits());
  auto expected =
      dd::CMat{{1, 1, 1, 1}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};
  EXPECT_EQ(outputMatrix, expected);

  // test also for non-regular garbage reduction as well
  outputDD = dd->reduceGarbage(inputDD, {true, true}, false);

  outputMatrix = outputDD.getMatrix(dd->qubits());
  expected = dd::CMat{{1, 0, 0, 0}, {1, 0, 0, 0}, {1, 0, 0, 0}, {1, 0, 0, 0}};
  EXPECT_EQ(outputMatrix, expected);
}

TEST(DDPackageTest, ReduceGarbageIdentityBeforeFirstNode) {
  auto dd = std::make_unique<dd::Package<>>(2);
  auto xGate = dd->makeGateDD(dd::X_MAT, 0);
  auto outputDD = dd->reduceGarbage(xGate, {false, true});

  auto outputMatrix = outputDD.getMatrix(dd->qubits());
  auto expected =
      dd::CMat{{0, 1, 0, 1}, {1, 0, 1, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};
  EXPECT_EQ(outputMatrix, expected);

  // test also for non-regular garbage reduction as well
  outputDD = dd->reduceGarbage(xGate, {false, true}, false);

  outputMatrix = outputDD.getMatrix(dd->qubits());
  expected = dd::CMat{{0, 1, 0, 0}, {1, 0, 0, 0}, {0, 1, 0, 0}, {1, 0, 0, 0}};
  EXPECT_EQ(outputMatrix, expected);
}

TEST(DDPackageTest, ReduceGarbageIdentityAfterLastNode) {
  auto dd = std::make_unique<dd::Package<>>(2);
  auto xGate = dd->makeGateDD(dd::X_MAT, 1);
  dd->incRef(xGate);
  auto outputDD = dd->reduceGarbage(xGate, {true, false});

  auto outputMatrix = outputDD.getMatrix(dd->qubits());
  auto expected =
      dd::CMat{{0, 0, 1, 1}, {0, 0, 0, 0}, {1, 1, 0, 0}, {0, 0, 0, 0}};
  EXPECT_EQ(outputMatrix, expected);

  // test also for non-regular garbage reduction as well
  dd->incRef(xGate);
  outputDD = dd->reduceGarbage(xGate, {true, false}, false);

  outputMatrix = outputDD.getMatrix(dd->qubits());
  expected = dd::CMat{{0, 0, 1, 0}, {0, 0, 1, 0}, {1, 0, 0, 0}, {1, 0, 0, 0}};
  EXPECT_EQ(outputMatrix, expected);
}

TEST(DDPackageTest, ReduceGarbageIdentityBetweenTwoNodes) {
  auto dd = std::make_unique<dd::Package<>>(3);
  auto xGate0 = dd->makeGateDD(dd::X_MAT, 0);
  auto xGate2 = dd->makeGateDD(dd::X_MAT, 2);
  auto state = dd->multiply(xGate0, xGate2);

  dd->incRef(state);
  auto outputDD = dd->reduceGarbage(state, {false, true, false});
  auto outputMatrix = outputDD.getMatrix(dd->qubits());
  auto expected = dd::CMat{{0, 0, 0, 0, 0, 1, 0, 1}, {0, 0, 0, 0, 1, 0, 1, 0},
                           {0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0},
                           {0, 1, 0, 1, 0, 0, 0, 0}, {1, 0, 1, 0, 0, 0, 0, 0},
                           {0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0}};
  EXPECT_EQ(outputMatrix, expected);

  // test also for non-regular garbage reduction as well
  dd->incRef(state);
  outputDD = dd->reduceGarbage(state, {false, true, false}, false);

  outputMatrix = outputDD.getMatrix(dd->qubits());
  expected = dd::CMat{{0, 0, 0, 0, 0, 1, 0, 0}, {0, 0, 0, 0, 1, 0, 0, 0},
                      {0, 0, 0, 0, 0, 1, 0, 0}, {0, 0, 0, 0, 1, 0, 0, 0},
                      {0, 1, 0, 0, 0, 0, 0, 0}, {1, 0, 0, 0, 0, 0, 0, 0},
                      {0, 1, 0, 0, 0, 0, 0, 0}, {1, 0, 0, 0, 0, 0, 0, 0}};
  EXPECT_EQ(outputMatrix, expected);
}
