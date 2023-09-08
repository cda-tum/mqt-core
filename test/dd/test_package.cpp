#include "dd/Export.hpp"
#include "dd/GateMatrixDefinitions.hpp"
#include "dd/Package.hpp"
#include "dd/statistics/PackageStatistics.hpp"
#include "operations/Control.hpp"

#include <gtest/gtest.h>
#include <iomanip>
#include <memory>
#include <random>
#include <sstream>

using namespace qc::literals;

TEST(DDPackageTest, RequestInvalidPackageSize) {
  EXPECT_THROW(auto dd = std::make_unique<dd::Package<>>(
                   dd::Package<>::MAX_POSSIBLE_QUBITS + 2),
               std::invalid_argument);
}

TEST(DDPackageTest, TrivialTest) {
  auto dd = std::make_unique<dd::Package<>>(2);
  EXPECT_EQ(dd->qubits(), 2);

  auto xGate = dd->makeGateDD(dd::Xmat, 1, 0);
  auto hGate = dd->makeGateDD(dd::Hmat, 1, 0);

  ASSERT_EQ(dd->getValueByPath(hGate, "0"), (dd::ComplexValue{dd::SQRT2_2, 0}));

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

  auto hGate = dd->makeGateDD(dd::Hmat, 2, 1);
  auto cxGate = dd->makeGateDD(dd::Xmat, 2, 1_pc, 0);
  auto zeroState = dd->makeZeroState(2);

  auto bellState = dd->multiply(dd->multiply(cxGate, hGate), zeroState);
  dd->printVector(bellState);

  // repeated calculation is practically for free
  auto bellState2 = dd->multiply(dd->multiply(cxGate, hGate), zeroState);
  EXPECT_EQ(bellState, bellState2);

  ASSERT_EQ(dd->getValueByPath(bellState, "00"),
            (dd::ComplexValue{dd::SQRT2_2, 0}));
  ASSERT_EQ(dd->getValueByPath(bellState, "01"), (dd::ComplexValue{0, 0}));
  ASSERT_EQ(dd->getValueByPath(bellState, "10"), (dd::ComplexValue{0, 0}));
  ASSERT_EQ(dd->getValueByPath(bellState, "11"),
            (dd::ComplexValue{dd::SQRT2_2, 0}));

  ASSERT_EQ(dd->getValueByPath(bellState, 0),
            (dd::ComplexValue{dd::SQRT2_2, 0}));
  ASSERT_EQ(dd->getValueByPath(bellState, 1), (dd::ComplexValue{0, 0}));
  ASSERT_EQ(dd->getValueByPath(bellState, 2), (dd::ComplexValue{0, 0}));
  ASSERT_EQ(dd->getValueByPath(bellState, 3),
            (dd::ComplexValue{dd::SQRT2_2, 0}));

  auto goalState =
      dd::CVec{{dd::SQRT2_2, 0.}, {0., 0.}, {0., 0.}, {dd::SQRT2_2, 0.}};
  ASSERT_EQ(dd->getVector(bellState), goalState);

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

  dd::printStatistics(dd.get());
}

TEST(DDPackageTest, QFTState) {
  auto dd = std::make_unique<dd::Package<>>(3);

  auto h0Gate = dd->makeGateDD(dd::Hmat, 3, 0);
  auto s0Gate = dd->makeGateDD(dd::Smat, 3, 1_pc, 0);
  auto t0Gate = dd->makeGateDD(dd::Tmat, 3, 2_pc, 0);
  auto h1Gate = dd->makeGateDD(dd::Hmat, 3, 1);
  auto s1Gate = dd->makeGateDD(dd::Smat, 3, 2_pc, 1);
  auto h2Gate = dd->makeGateDD(dd::Hmat, 3, 2);
  auto swapGate = dd->makeSWAPDD(3, qc::Controls{}, 0, 2);

  auto qftOp = dd->multiply(s0Gate, h0Gate);
  qftOp = dd->multiply(t0Gate, qftOp);
  qftOp = dd->multiply(h1Gate, qftOp);
  qftOp = dd->multiply(s1Gate, qftOp);
  qftOp = dd->multiply(h2Gate, qftOp);

  qftOp = dd->multiply(swapGate, qftOp);
  auto qftState = dd->multiply(qftOp, dd->makeZeroState(3));

  dd->printVector(qftState);

  for (dd::Qubit qubit = 0; qubit < 7; ++qubit) {
    ASSERT_NEAR(dd->getValueByPath(qftState, static_cast<std::size_t>(qubit)).r,
                0.5 * dd::SQRT2_2, dd::RealNumber::eps);
    ASSERT_EQ(dd->getValueByPath(qftState, static_cast<std::size_t>(qubit)).i,
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

  dd::printStatistics(dd.get());
}

TEST(DDPackageTest, CorruptedBellState) {
  auto dd = std::make_unique<dd::Package<>>(2);

  auto hGate = dd->makeGateDD(dd::Hmat, 2, 1);
  auto cxGate = dd->makeGateDD(dd::Xmat, 2, 1_pc, 0);
  auto zeroState = dd->makeZeroState(2);

  auto bellState = dd->multiply(dd->multiply(cxGate, hGate), zeroState);

  bellState.w = dd->cn.getTemporary(0.5, 0);
  // prints a warning
  std::mt19937_64 mt; // NOLINT(cert-msc51-cpp)
  std::cout << dd->measureAll(bellState, false, mt) << "\n";

  bellState.w = dd::Complex::zero;

  ASSERT_THROW(dd->measureAll(bellState, false, mt), std::runtime_error);

  ASSERT_THROW(dd->measureOneCollapsing(bellState, 0, true, mt),
               std::runtime_error);
}

TEST(DDPackageTest, NegativeControl) {
  auto dd = std::make_unique<dd::Package<>>(2);

  auto xGate = dd->makeGateDD(dd::Xmat, 2, 1_nc, 0);
  auto zeroState = dd->makeZeroState(2);
  auto state01 = dd->multiply(xGate, zeroState);
  EXPECT_EQ(dd->getValueByPath(state01, 0b01).r, 1.);
}

TEST(DDPackageTest, IdentityTrace) {
  auto dd = std::make_unique<dd::Package<>>(4);
  auto fullTrace = dd->trace(dd->makeIdent(4));

  ASSERT_EQ(fullTrace, (dd::ComplexValue{16, 0}));
}

TEST(DDPackageTest, PartialIdentityTrace) {
  auto dd = std::make_unique<dd::Package<>>(2);
  auto tr = dd->partialTrace(dd->makeIdent(2), {false, true});
  auto mul = dd->multiply(tr, tr);
  EXPECT_EQ(dd::RealNumber::val(mul.w.r), 4.0);
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

  auto hGate = dd->makeGateDD(dd::Hmat, 2, 1);
  auto cxGate = dd->makeGateDD(dd::Xmat, 2, 1_pc, 0);
  auto zeroState = dd->makeZeroState(2);

  auto bellState = dd->multiply(dd->multiply(cxGate, hGate), zeroState);

  serialize(bellState, "bell_state.dd", false);
  auto deserializedBellState =
      dd->deserialize<dd::vNode>("bell_state.dd", false);
  EXPECT_EQ(bellState, deserializedBellState);

  serialize(bellState, "bell_state_binary.dd", true);
  deserializedBellState =
      dd->deserialize<dd::vNode>("bell_state_binary.dd", true);
  EXPECT_EQ(bellState, deserializedBellState);
}

TEST(DDPackageTest, BellMatrix) {
  auto dd = std::make_unique<dd::Package<>>(2);

  auto hGate = dd->makeGateDD(dd::Hmat, 2, 1);
  auto cxGate = dd->makeGateDD(dd::Xmat, 2, 1_pc, 0);

  auto bellMatrix = dd->multiply(cxGate, hGate);

  ASSERT_EQ(dd->getValueByPath(bellMatrix, "00"),
            (dd::ComplexValue{dd::SQRT2_2, 0}));
  ASSERT_EQ(dd->getValueByPath(bellMatrix, "02"), (dd::ComplexValue{0, 0}));
  ASSERT_EQ(dd->getValueByPath(bellMatrix, "20"), (dd::ComplexValue{0, 0}));
  ASSERT_EQ(dd->getValueByPath(bellMatrix, "22"),
            (dd::ComplexValue{dd::SQRT2_2, 0}));

  ASSERT_EQ(dd->getValueByPath(bellMatrix, 0, 0),
            (dd::ComplexValue{dd::SQRT2_2, 0}));
  ASSERT_EQ(dd->getValueByPath(bellMatrix, 1, 0), (dd::ComplexValue{0, 0}));
  ASSERT_EQ(dd->getValueByPath(bellMatrix, 2, 0), (dd::ComplexValue{0, 0}));
  ASSERT_EQ(dd->getValueByPath(bellMatrix, 3, 0),
            (dd::ComplexValue{dd::SQRT2_2, 0}));

  ASSERT_EQ(dd->getValueByPath(bellMatrix, 0, 1), (dd::ComplexValue{0, 0}));
  ASSERT_EQ(dd->getValueByPath(bellMatrix, 1, 1),
            (dd::ComplexValue{dd::SQRT2_2, 0}));
  ASSERT_EQ(dd->getValueByPath(bellMatrix, 2, 1),
            (dd::ComplexValue{dd::SQRT2_2, 0}));
  ASSERT_EQ(dd->getValueByPath(bellMatrix, 3, 1), (dd::ComplexValue{0, 0}));

  ASSERT_EQ(dd->getValueByPath(bellMatrix, 0, 2),
            (dd::ComplexValue{dd::SQRT2_2, 0}));
  ASSERT_EQ(dd->getValueByPath(bellMatrix, 1, 2), (dd::ComplexValue{0, 0}));
  ASSERT_EQ(dd->getValueByPath(bellMatrix, 2, 2), (dd::ComplexValue{0, 0}));
  ASSERT_EQ(dd->getValueByPath(bellMatrix, 3, 2),
            (dd::ComplexValue{-dd::SQRT2_2, 0}));

  ASSERT_EQ(dd->getValueByPath(bellMatrix, 0, 3), (dd::ComplexValue{0, 0}));
  ASSERT_EQ(dd->getValueByPath(bellMatrix, 1, 3),
            (dd::ComplexValue{dd::SQRT2_2, 0}));
  ASSERT_EQ(dd->getValueByPath(bellMatrix, 2, 3),
            (dd::ComplexValue{-dd::SQRT2_2, 0}));
  ASSERT_EQ(dd->getValueByPath(bellMatrix, 3, 3), (dd::ComplexValue{0, 0}));

  auto goalRow0 =
      dd::CVec{{dd::SQRT2_2, 0.}, {0., 0.}, {dd::SQRT2_2, 0.}, {0., 0.}};
  auto goalRow1 =
      dd::CVec{{0., 0.}, {dd::SQRT2_2, 0.}, {0., 0.}, {dd::SQRT2_2, 0.}};
  auto goalRow2 =
      dd::CVec{{0., 0.}, {dd::SQRT2_2, 0.}, {0., 0.}, {-dd::SQRT2_2, 0.}};
  auto goalRow3 =
      dd::CVec{{dd::SQRT2_2, 0.}, {0., 0.}, {-dd::SQRT2_2, 0.}, {0., 0.}};
  auto goalMatrix = dd::CMat{goalRow0, goalRow1, goalRow2, goalRow3};
  ASSERT_EQ(dd->getMatrix(bellMatrix), goalMatrix);

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

  dd::printStatistics(dd.get());
}

TEST(DDPackageTest, MatrixSerializationTest) {
  auto dd = std::make_unique<dd::Package<>>(2);

  auto hGate = dd->makeGateDD(dd::Hmat, 2, 1);
  auto cxGate = dd->makeGateDD(dd::Xmat, 2, 1_pc, 0);

  auto bellMatrix = dd->multiply(cxGate, hGate);

  serialize(bellMatrix, "bell_matrix.dd", false);
  auto deserializedBellMatrix =
      dd->deserialize<dd::mNode>("bell_matrix.dd", false);
  EXPECT_EQ(bellMatrix, deserializedBellMatrix);

  serialize(bellMatrix, "bell_matrix_binary.dd", true);
  deserializedBellMatrix =
      dd->deserialize<dd::mNode>("bell_matrix_binary.dd", true);
  EXPECT_EQ(bellMatrix, deserializedBellMatrix);
}

TEST(DDPackageTest, SerializationErrors) {
  auto dd = std::make_unique<dd::Package<>>(2);

  auto hGate = dd->makeGateDD(dd::Hmat, 2, 1);
  auto cxGate = dd->makeGateDD(dd::Xmat, 2, 1_pc, 0);
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

TEST(DDPackageTest, TestConsistency) {
  auto dd = std::make_unique<dd::Package<>>(2);

  auto hGate = dd->makeGateDD(dd::Hmat, 2, 1);
  auto cxGate = dd->makeGateDD(dd::Xmat, 2, 1_pc, 0);
  auto zeroState = dd->makeZeroState(2);

  auto bellMatrix = dd->multiply(cxGate, hGate);
  dd->incRef(bellMatrix);
  auto local = dd->isLocallyConsistent(bellMatrix);
  EXPECT_TRUE(local);
  auto global = dd->isGloballyConsistent(bellMatrix);
  EXPECT_TRUE(global);
  dd->debugnode(bellMatrix.p);

  auto bellState = dd->multiply(bellMatrix, zeroState);
  dd->incRef(bellState);
  local = dd->isLocallyConsistent(bellState);
  EXPECT_TRUE(local);
  global = dd->isGloballyConsistent(bellState);
  EXPECT_TRUE(global);
  dd->debugnode(bellState.p);
}

TEST(DDPackageTest, Extend) {
  auto dd = std::make_unique<dd::Package<>>(4);

  auto id = dd->makeIdent(3);
  EXPECT_EQ(id.p->v, 2);
  EXPECT_EQ(id.p->e[0], id.p->e[3]);
  EXPECT_EQ(id.p->e[1], id.p->e[2]);
  EXPECT_TRUE(id.p->isIdentity());

  auto ext = dd->extend(id, 0, 1);
  EXPECT_EQ(ext.p->v, 3);
  EXPECT_EQ(ext.p->e[0], ext.p->e[3]);
  EXPECT_EQ(ext.p->e[1], ext.p->e[2]);
  EXPECT_TRUE(ext.p->isIdentity());
}

TEST(DDPackageTest, Identity) {
  auto dd = std::make_unique<dd::Package<>>(4);

  EXPECT_TRUE(dd->makeIdent(0).isOneTerminal());

  auto id3 = dd->makeIdent(3);
  EXPECT_EQ(dd->makeIdent(0, 2), id3);
  const auto& table = dd->getIdentityTable();
  EXPECT_NE(table[2].p, nullptr);

  auto id2 = dd->makeIdent(0, 1); // should be found in idTable
  EXPECT_EQ(dd->makeIdent(2), id2);

  auto id4 = dd->makeIdent(0, 3); // should use id3 and extend it
  EXPECT_EQ(dd->makeIdent(0, 3), id4);
  EXPECT_NE(table[3].p, nullptr);

  auto idCached = dd->makeIdent(4);
  EXPECT_EQ(id4, idCached);
}

TEST(DDPackageTest, TestLocalInconsistency) {
  auto dd = std::make_unique<dd::Package<>>(3);

  auto hGate = dd->makeGateDD(dd::Hmat, 2, 0);
  auto cxGate = dd->makeGateDD(dd::Xmat, 2, 0_pc, 1);
  auto zeroState = dd->makeZeroState(2);

  auto bellState = dd->multiply(dd->multiply(cxGate, hGate), zeroState);
  auto local = dd->isLocallyConsistent(bellState);
  EXPECT_FALSE(local);
  bellState.p->ref = 1;
  local = dd->isLocallyConsistent(bellState);
  EXPECT_FALSE(local);
  bellState.p->ref = 0;
  dd->incRef(bellState);

  bellState.p->v = 2;
  local = dd->isLocallyConsistent(bellState);
  EXPECT_FALSE(local);
  bellState.p->v = 1;

  bellState.p->e[0].w.r->ref = 0;
  local = dd->isLocallyConsistent(bellState);
  EXPECT_FALSE(local);
  bellState.p->e[0].w.r->ref = 1;
}

TEST(DDPackageTest, Ancillaries) {
  auto dd = std::make_unique<dd::Package<>>(4);
  auto hGate = dd->makeGateDD(dd::Hmat, 2, 0);
  auto cxGate = dd->makeGateDD(dd::Xmat, 2, 0_pc, 1);
  auto bellMatrix = dd->multiply(cxGate, hGate);

  dd->incRef(bellMatrix);
  auto reducedBellMatrix =
      dd->reduceAncillae(bellMatrix, {false, false, false, false});
  EXPECT_EQ(bellMatrix, reducedBellMatrix);
  dd->incRef(bellMatrix);
  reducedBellMatrix =
      dd->reduceAncillae(bellMatrix, {false, false, true, true});
  EXPECT_EQ(bellMatrix, reducedBellMatrix);

  auto extendedBellMatrix = dd->extend(bellMatrix, 2);
  dd->incRef(extendedBellMatrix);
  reducedBellMatrix =
      dd->reduceAncillae(extendedBellMatrix, {false, false, true, true});
  EXPECT_TRUE(reducedBellMatrix.p->e[1].isZeroTerminal());
  EXPECT_TRUE(reducedBellMatrix.p->e[2].isZeroTerminal());
  EXPECT_TRUE(reducedBellMatrix.p->e[3].isZeroTerminal());

  EXPECT_EQ(reducedBellMatrix.p->e[0].p->e[0].p, bellMatrix.p);
  EXPECT_TRUE(reducedBellMatrix.p->e[0].p->e[1].isZeroTerminal());
  EXPECT_TRUE(reducedBellMatrix.p->e[0].p->e[2].isZeroTerminal());
  EXPECT_TRUE(reducedBellMatrix.p->e[0].p->e[3].isZeroTerminal());

  dd->incRef(extendedBellMatrix);
  reducedBellMatrix =
      dd->reduceAncillae(extendedBellMatrix, {false, false, true, true}, false);
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
  auto hGate = dd->makeGateDD(dd::Hmat, 2, 0);
  auto cxGate = dd->makeGateDD(dd::Xmat, 2, 0_pc, 1);
  auto zeroState = dd->makeZeroState(2);
  auto bellState = dd->multiply(dd->multiply(cxGate, hGate), zeroState);
  dd->printVector(bellState);

  dd->incRef(bellState);
  auto reducedBellState =
      dd->reduceGarbage(bellState, {false, false, false, false});
  EXPECT_EQ(bellState, reducedBellState);
  dd->incRef(bellState);
  reducedBellState = dd->reduceGarbage(bellState, {false, false, true, false});
  EXPECT_EQ(bellState, reducedBellState);

  dd->incRef(bellState);
  reducedBellState = dd->reduceGarbage(bellState, {false, true, false, false});
  auto vec = dd->getVector(reducedBellState);
  dd->printVector(reducedBellState);
  EXPECT_EQ(vec[2], static_cast<std::complex<dd::fp>>(dd::complex_zero));
  EXPECT_EQ(vec[3], static_cast<std::complex<dd::fp>>(dd::complex_zero));

  dd->incRef(bellState);
  reducedBellState = dd->reduceGarbage(bellState, {true, false, false, false});
  dd->printVector(reducedBellState);
  vec = dd->getVector(reducedBellState);
  EXPECT_EQ(vec[1], static_cast<std::complex<dd::fp>>(dd::complex_zero));
  EXPECT_EQ(vec[3], static_cast<std::complex<dd::fp>>(dd::complex_zero));
}

TEST(DDPackageTest, GarbageMatrix) {
  auto dd = std::make_unique<dd::Package<>>(4);
  auto hGate = dd->makeGateDD(dd::Hmat, 2, 0);
  auto cxGate = dd->makeGateDD(dd::Xmat, 2, 0_pc, 1);
  auto bellMatrix = dd->multiply(cxGate, hGate);

  dd->incRef(bellMatrix);
  auto reducedBellMatrix =
      dd->reduceGarbage(bellMatrix, {false, false, false, false});
  EXPECT_EQ(bellMatrix, reducedBellMatrix);
  dd->incRef(bellMatrix);
  reducedBellMatrix =
      dd->reduceGarbage(bellMatrix, {false, false, true, false});
  EXPECT_EQ(bellMatrix, reducedBellMatrix);

  dd->incRef(bellMatrix);
  reducedBellMatrix =
      dd->reduceGarbage(bellMatrix, {false, true, false, false});
  auto mat = dd->getMatrix(reducedBellMatrix);
  auto zero = dd::CVec{{0., 0.}, {0., 0.}, {0., 0.}, {0., 0.}};
  EXPECT_EQ(mat[2], zero);
  EXPECT_EQ(mat[3], zero);

  dd->incRef(bellMatrix);
  reducedBellMatrix =
      dd->reduceGarbage(bellMatrix, {true, false, false, false});
  mat = dd->getMatrix(reducedBellMatrix);
  EXPECT_EQ(mat[1], zero);
  EXPECT_EQ(mat[3], zero);

  dd->incRef(bellMatrix);
  reducedBellMatrix =
      dd->reduceGarbage(bellMatrix, {false, true, false, false}, false);
  EXPECT_TRUE(reducedBellMatrix.p->e[1].isZeroTerminal());
  EXPECT_TRUE(reducedBellMatrix.p->e[3].isZeroTerminal());
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
  EXPECT_THROW(dd->makeGateDD(dd::Xmat, 3, 0), std::runtime_error);
}

TEST(DDPackageTest, InvalidDecRef) {
  auto dd = std::make_unique<dd::Package<>>(2);
  auto e = dd->makeIdent(2);
  EXPECT_DEBUG_DEATH(
      dd->decRef(e),
      "Reference count of Node must not be zero before decrement");
}

TEST(DDPackageTest, PackageReset) {
  auto dd = std::make_unique<dd::Package<>>(1);

  // one node in unique table of variable 0
  auto iGate = dd->makeIdent(1);
  const auto& unique = dd->mUniqueTable.getTables();
  const auto& table = unique[0];
  auto ihash = decltype(dd->mUniqueTable)::hash(iGate.p);
  const auto* node = table[ihash];
  std::cout << ihash << ": " << reinterpret_cast<uintptr_t>(iGate.p) << "\n";
  // node should be the first in this unique table bucket
  EXPECT_EQ(node, iGate.p);
  dd->reset();
  // after clearing the tables, they should be empty
  EXPECT_EQ(table[ihash], nullptr);
  iGate = dd->makeIdent(1);
  const auto* node2 = table[ihash];
  // after recreating the DD, it should receive the same node
  EXPECT_EQ(node2, node);
}

TEST(DDPackageTest, MaxRefCount) {
  auto dd = std::make_unique<dd::Package<>>(1);
  auto e = dd->makeIdent(1);
  // ref count saturates at this value
  e.p->ref = std::numeric_limits<decltype(e.p->ref)>::max();
  dd->incRef(e);
  EXPECT_EQ(e.p->ref, std::numeric_limits<decltype(e.p->ref)>::max());
}

TEST(DDPackageTest, Inverse) {
  auto dd = std::make_unique<dd::Package<>>(1);
  auto x = dd->makeGateDD(dd::Xmat, 1, 0);
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

TEST(DDPackageTest, MatrixTranspose) {
  auto dd = std::make_unique<dd::Package<>>(2);
  auto cx = dd->makeGateDD(dd::Xmat, 2, 1_pc, 0);

  // transposing a symmetric matrix shall yield a symmetric matrix
  auto cxTransposed = dd->transpose(cx);
  EXPECT_EQ(cxTransposed, cx);

  // the Y gate is not symmetric
  auto y = dd->makeGateDD(dd::Ymat, 2, 0);
  auto yTransposed = dd->transpose(y);
  EXPECT_NE(yTransposed, y);

  // transposing twice should yield the original matrix
  auto yTT = dd->transpose(yTransposed);
  EXPECT_EQ(yTT, y);

  // perform the same computation again -> trigger a compute table hit
  auto yAgain = dd->transpose(yTransposed);
  EXPECT_EQ(yAgain, y);
}

TEST(DDPackageTest, SpecialCaseTerminal) {
  auto dd = std::make_unique<dd::Package<>>(2);
  auto one = dd::vEdge::one;
  dd::export2Dot(one, "oneColored.dot", true, false, false, false, false);
  dd::export2Dot(one, "oneClassic.dot", false, false, false, false, false);
  dd::export2Dot(one, "oneMemory.dot", true, true, false, true, false);

  EXPECT_EQ(dd->vUniqueTable.lookup(one), one);

  auto zero = dd::vEdge::zero;
  EXPECT_EQ(dd->kronecker(zero, one), zero);
  EXPECT_EQ(dd->kronecker(one, one), one);

  dd->debugnode(one.p);
  const dd::ComplexValue cOne{1.0, 0.0};
  EXPECT_EQ(dd->getValueByPath(one, ""), cOne);
  EXPECT_EQ(dd->getValueByPath(one, 0), cOne);
  EXPECT_EQ(dd->getValueByPath(dd::mEdge::one, 0, 0), cOne);

  const dd::ComplexValue cZero{0.0, 0.0};
  EXPECT_EQ(dd->innerProduct(zero, zero), cZero);
}

TEST(DDPackageTest, KroneckerProduct) {
  auto dd = std::make_unique<dd::Package<>>(2);
  auto x = dd->makeGateDD(dd::Xmat, 1, 0);
  auto kronecker = dd->kronecker(x, x);
  EXPECT_EQ(kronecker.p->v, 1);
  EXPECT_EQ(kronecker.p->e[0], dd::mEdge::zero);
  EXPECT_EQ(kronecker.p->e[0], kronecker.p->e[3]);
  EXPECT_EQ(kronecker.p->e[1], kronecker.p->e[2]);
  EXPECT_EQ(kronecker.p->e[1].p->v, 0);
  EXPECT_EQ(kronecker.p->e[1].p->e[0], dd::mEdge::zero);
  EXPECT_EQ(kronecker.p->e[1].p->e[0], kronecker.p->e[1].p->e[3]);
  EXPECT_EQ(kronecker.p->e[1].p->e[1], dd::mEdge::one);
  EXPECT_EQ(kronecker.p->e[1].p->e[1], kronecker.p->e[1].p->e[2]);

  auto kronecker2 = dd->kronecker(x, x);
  EXPECT_EQ(kronecker, kronecker2);
}

TEST(DDPackageTest, NearZeroNormalize) {
  auto dd = std::make_unique<dd::Package<>>(2);
  const dd::fp nearZero = dd::RealNumber::eps / 10;
  dd::vEdge ve{};
  ve.p = dd->vMemoryManager.get();
  ve.p->v = 1;
  ve.w = dd::Complex::one;
  for (auto& edge : ve.p->e) {
    edge.p = dd->vMemoryManager.get();
    edge.p->v = 0;
    edge.w = dd->cn.getCached(nearZero, 0.);
    edge.p->e = {dd::vEdge::one, dd::vEdge::one};
  }
  auto veNormalizedCached = dd->normalize(ve, true);
  EXPECT_EQ(veNormalizedCached, dd::vEdge::zero);

  for (auto& edge : ve.p->e) {
    edge.p = dd->vMemoryManager.get();
    edge.p->v = 0;
    edge.w = dd->cn.lookup(nearZero, 0.);
    edge.p->e = {dd::vEdge::one, dd::vEdge::one};
  }
  auto veNormalized = dd->normalize(ve, false);
  EXPECT_EQ(veNormalized, dd::vEdge::zero);

  dd::mEdge me{};
  me.p = dd->mMemoryManager.get();
  me.p->v = 1;
  me.w = dd::Complex::one;
  for (auto& edge : me.p->e) {
    edge.p = dd->mMemoryManager.get();
    edge.p->v = 0;
    edge.w = dd->cn.getCached(nearZero, 0.);
    edge.p->e = {dd::mEdge::one, dd::mEdge::one, dd::mEdge::one,
                 dd::mEdge::one};
  }
  auto meNormalizedCached = dd->normalize(me, true);
  EXPECT_EQ(meNormalizedCached, dd::mEdge::zero);

  me.p = dd->mMemoryManager.get();
  for (auto& edge : me.p->e) {
    edge.p = dd->mMemoryManager.get();
    edge.p->v = 0;
    edge.w = dd->cn.lookup(nearZero, 0.);
    edge.p->e = {dd::mEdge::one, dd::mEdge::one, dd::mEdge::one,
                 dd::mEdge::one};
  }
  auto meNormalized = dd->normalize(me, false);
  EXPECT_EQ(meNormalized, dd::mEdge::zero);
}

TEST(DDPackageTest, DestructiveMeasurementAll) {
  auto dd = std::make_unique<dd::Package<>>(4);
  auto hGate0 = dd->makeGateDD(dd::Hmat, 2, 0);
  auto hGate1 = dd->makeGateDD(dd::Hmat, 2, 1);
  auto plusMatrix = dd->multiply(hGate0, hGate1);
  auto zeroState = dd->makeZeroState(2);
  auto plusState = dd->multiply(plusMatrix, zeroState);
  dd->incRef(plusState);

  std::mt19937_64 mt{0}; // NOLINT(ms

  const dd::CVec vBefore = dd->getVector(plusState);

  ASSERT_EQ(vBefore[0], vBefore[1]);
  ASSERT_EQ(vBefore[0], vBefore[2]);
  ASSERT_EQ(vBefore[0], vBefore[3]);

  const std::string m = dd->measureAll(plusState, true, mt);

  const dd::CVec vAfter = dd->getVector(plusState);
  const int i = std::stoi(m, nullptr, 2);

  ASSERT_EQ(vAfter[static_cast<std::size_t>(i)],
            static_cast<std::complex<dd::fp>>(dd::complex_one));
}

TEST(DDPackageTest, DestructiveMeasurementOne) {
  auto dd = std::make_unique<dd::Package<>>(4);
  auto hGate0 = dd->makeGateDD(dd::Hmat, 2, 0);
  auto hGate1 = dd->makeGateDD(dd::Hmat, 2, 1);
  auto plusMatrix = dd->multiply(hGate0, hGate1);
  auto zeroState = dd->makeZeroState(2);
  auto plusState = dd->multiply(plusMatrix, zeroState);
  dd->incRef(plusState);

  std::mt19937_64 mt{0}; // NOLINT(cert-msc51-cpp)

  const char m = dd->measureOneCollapsing(plusState, 0, true, mt);
  const dd::CVec vAfter = dd->getVector(plusState);

  ASSERT_EQ(m, '0');
  ASSERT_EQ(vAfter[0], static_cast<std::complex<dd::fp>>(dd::complex_SQRT2_2));
  ASSERT_EQ(vAfter[2], static_cast<std::complex<dd::fp>>(dd::complex_SQRT2_2));
  ASSERT_EQ(vAfter[1], static_cast<std::complex<dd::fp>>(dd::complex_zero));
  ASSERT_EQ(vAfter[3], static_cast<std::complex<dd::fp>>(dd::complex_zero));
}

TEST(DDPackageTest, DestructiveMeasurementOneArbitraryNormalization) {
  auto dd = std::make_unique<dd::Package<>>(4);
  auto hGate0 = dd->makeGateDD(dd::Hmat, 2, 0);
  auto hGate1 = dd->makeGateDD(dd::Hmat, 2, 1);
  auto plusMatrix = dd->multiply(hGate0, hGate1);
  auto zeroState = dd->makeZeroState(2);
  auto plusState = dd->multiply(plusMatrix, zeroState);
  dd->incRef(plusState);

  std::mt19937_64 mt{0}; // NOLINT(cert-msc51-cpp)

  const char m = dd->measureOneCollapsing(plusState, 0, false, mt);
  const dd::CVec vAfter = dd->getVector(plusState);

  ASSERT_EQ(m, '0');
  ASSERT_EQ(vAfter[0], static_cast<std::complex<dd::fp>>(dd::complex_SQRT2_2));
  ASSERT_EQ(vAfter[2], static_cast<std::complex<dd::fp>>(dd::complex_SQRT2_2));
  ASSERT_EQ(vAfter[1], static_cast<std::complex<dd::fp>>(dd::complex_zero));
  ASSERT_EQ(vAfter[3], static_cast<std::complex<dd::fp>>(dd::complex_zero));
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
  auto h = dd->makeGateDD(dd::Hmat, 1, 0);
  auto state1 = dd->multiply(h, state);
  auto z = dd->makeGateDD(dd::Zmat, 1, 0);
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
    auto p = dd->makeGateDD(dd::Phasemat(lambda), 1, 0);
    auto pdag = dd->makeGateDD(dd::Phasemat(-lambda), 1, 0);
    auto result = dd->multiply(p, pdag);
    EXPECT_TRUE(result.p->isIdentity());
    dd->cUniqueTable.clear();
    dd->cCacheManager.reset();
    dd->cMemoryManager.reset();
  }
}

TEST(DDPackageTest, FidelityOfMeasurementOutcomes) {
  auto dd = std::make_unique<dd::Package<>>(3);

  auto hGate = dd->makeGateDD(dd::Hmat, 3, 2);
  auto cxGate1 = dd->makeGateDD(dd::Xmat, 3, 2_pc, 1);
  auto cxGate2 = dd->makeGateDD(dd::Xmat, 3, 1_pc, 0);
  auto zeroState = dd->makeZeroState(3);

  auto ghzState = dd->multiply(
      cxGate2, dd->multiply(cxGate1, dd->multiply(hGate, zeroState)));

  dd::ProbabilityVector probs{};
  probs[0] = 0.5;
  probs[7] = 0.5;
  auto fidelity = dd->fidelityOfMeasurementOutcomes(ghzState, probs);
  EXPECT_NEAR(fidelity, 1.0, dd::RealNumber::eps);
}

TEST(DDPackageTest, CloseToIdentity) {
  auto dd = std::make_unique<dd::Package<>>(3);
  auto id = dd->makeIdent(1);
  EXPECT_TRUE(dd->isCloseToIdentity(id));
  dd::mEdge close{};
  close.p = id.p;
  close.w = dd->cn.lookup(1e-11, 0);
  auto id2 = dd->makeDDNode(
      1, std::array{id, dd::mEdge::zero, dd::mEdge::zero, close});
  EXPECT_TRUE(dd->isCloseToIdentity(id2));

  auto noId = dd->makeDDNode(
      1, std::array{dd::mEdge::zero, id, dd::mEdge::zero, close});
  EXPECT_FALSE(dd->isCloseToIdentity(noId));

  dd::mEdge notClose{};
  notClose.p = id.p;
  notClose.w = dd->cn.lookup(1e-9, 0);
  auto noId2 = dd->makeDDNode(
      1, std::array{notClose, dd::mEdge::zero, dd::mEdge::zero, close});
  EXPECT_FALSE(dd->isCloseToIdentity(noId2));

  auto noId3 = dd->makeDDNode(
      1, std::array{close, dd::mEdge::zero, dd::mEdge::zero, notClose});
  EXPECT_FALSE(dd->isCloseToIdentity(noId3));

  auto notClose2 =
      dd->makeDDNode(0, std::array{dd::mEdge::zero, dd::mEdge::one,
                                   dd::mEdge::one, dd::mEdge::zero});
  auto notClose3 = dd->makeDDNode(
      1, std::array{notClose2, dd::mEdge::zero, dd::mEdge::zero, notClose2});
  EXPECT_FALSE(dd->isCloseToIdentity(notClose3));
}

struct DensityMatrixSimulatorDDPackageConfigTesting
    : public dd::DDPackageConfig {
  static constexpr std::size_t UT_DM_NBUCKET = 65536U;
  static constexpr std::size_t UT_DM_INITIAL_ALLOCATION_SIZE = 4096U;

  static constexpr std::size_t CT_DM_DM_MULT_NBUCKET = 16384U;
  static constexpr std::size_t CT_DM_ADD_NBUCKET = 16384U;
  static constexpr std::size_t CT_DM_NOISE_NBUCKET = 4096U;

  static constexpr std::size_t UT_MAT_NBUCKET = 16384U;
  static constexpr std::size_t CT_MAT_ADD_NBUCKET = 4096U;
  static constexpr std::size_t CT_VEC_ADD_NBUCKET = 4096U;
  static constexpr std::size_t CT_MAT_TRANS_NBUCKET = 4096U;
  static constexpr std::size_t CT_MAT_CONJ_TRANS_NBUCKET = 4096U;

  static constexpr std::size_t CT_MAT_MAT_MULT_NBUCKET = 1U;
  static constexpr std::size_t CT_MAT_VEC_MULT_NBUCKET = 1U;
  static constexpr std::size_t UT_VEC_NBUCKET = 1U;
  static constexpr std::size_t UT_VEC_INITIAL_ALLOCATION_SIZE = 1U;
  static constexpr std::size_t UT_MAT_INITIAL_ALLOCATION_SIZE = 1U;
  static constexpr std::size_t CT_VEC_KRON_NBUCKET = 1U;
  static constexpr std::size_t CT_MAT_KRON_NBUCKET = 1U;
  static constexpr std::size_t CT_VEC_INNER_PROD_NBUCKET = 1U;
  static constexpr std::size_t STOCHASTIC_CACHE_OPS = 1U;
};

using DensityMatrixPackageTest =
    dd::Package<DensityMatrixSimulatorDDPackageConfigTesting>;

TEST(DDPackageTest, dNodeMultiply) {
  // Multiply dNode with mNode (MxMxM)
  const auto nrQubits = 3U;
  auto dd = std::make_unique<DensityMatrixPackageTest>(nrQubits);
  // Make zero density matrix
  auto state = dd->makeZeroDensityOperator(dd->qubits());
  dd->incRef(state);
  std::vector<dd::mEdge> operations = {};
  operations.emplace_back(dd->makeGateDD(dd::Hmat, nrQubits, 0));
  operations.emplace_back(dd->makeGateDD(dd::Hmat, nrQubits, 1));
  operations.emplace_back(dd->makeGateDD(dd::Hmat, nrQubits, 2));
  operations.emplace_back(dd->makeGateDD(dd::Zmat, nrQubits, 2));

  for (const auto& op : operations) {
    dd->applyOperationToDensity(state, op, true);
  }

  const auto stateDensityMatrix = dd->getDensityMatrix(state);

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

  const auto probVector = dd->getProbVectorFromDensityMatrix(state, 0.001);
  const double tolerance = 1e-10;
  for (const auto& prob : probVector) {
    std::cout << prob.first << ": " << prob.second << "\n";
    EXPECT_NEAR(prob.second, 0.125, tolerance);
  }
}

TEST(DDPackageTest, dNodeMultiply2) {
  // Multiply dNode with mNode (MxMxM)
  const auto nrQubits = 3U;
  auto dd = std::make_unique<DensityMatrixPackageTest>(nrQubits);
  // Make zero density matrix
  auto state = dd->makeZeroDensityOperator(dd->qubits());
  dd->incRef(state);
  std::vector<dd::mEdge> operations = {};
  operations.emplace_back(dd->makeGateDD(dd::Hmat, nrQubits, 0));
  operations.emplace_back(dd->makeGateDD(dd::Hmat, nrQubits, 1));
  operations.emplace_back(dd->makeGateDD(dd::Hmat, nrQubits, 2));
  operations.emplace_back(dd->makeGateDD(dd::Zmat, nrQubits, 2));

  for (const auto& op : operations) {
    dd->applyOperationToDensity(state, op, true);
  }
  dd->printMatrix(operations[0]);

  const auto stateDensityMatrix = dd->getDensityMatrix(state);

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
  auto probVector = dd->getProbVectorFromDensityMatrix(state, 0.001);
  const double tolerance = 1e-10;
  for (const auto& prob : probVector) {
    std::cout << prob.first << ": " << prob.second << "\n";
    EXPECT_NEAR(prob.second, 0.125, tolerance);
  }
}

TEST(DDPackageTest, dNodeMulCache1) {
  // Make caching test with dNodes
  const auto nrQubits = 1U;
  auto dd = std::make_unique<DensityMatrixPackageTest>(nrQubits);
  // Make zero density matrix
  auto state = dd->makeZeroDensityOperator(nrQubits);
  dd->incRef(state);

  const auto operation = dd->makeGateDD(dd::Hmat, nrQubits, 0);
  dd->applyOperationToDensity(state, operation, true);

  state = dd->makeZeroDensityOperator(nrQubits);
  auto& computeTable =
      dd->getMultiplicationComputeTable<dd::dNode, dd::dNode>();

  const auto& densityMatrix0 =
      dd::densityFromMatrixEdge(dd->conjugateTranspose(operation));

  const auto xCopy = dd::dEdge{state.p, dd::Complex::one};
  const auto yCopy = dd::dEdge{densityMatrix0.p, dd::Complex::one};
  const auto* cachedResult = computeTable.lookup(xCopy, yCopy, false);
  ASSERT_NE(cachedResult, nullptr);
  ASSERT_NE(cachedResult->p, nullptr);
  state = dd->multiply(state, densityMatrix0, 0, false);
  ASSERT_NE(state.p, nullptr);
  ASSERT_EQ(state.p, cachedResult->p);

  const auto densityMatrix1 = dd::densityFromMatrixEdge(operation);
  const auto xCopy1 = dd::dEdge{densityMatrix1.p, dd::Complex::one};
  const auto yCopy1 = dd::dEdge{state.p, dd::Complex::one};
  const auto* cachedResult1 = computeTable.lookup(xCopy1, yCopy1, true);
  ASSERT_NE(cachedResult1, nullptr);
  ASSERT_NE(cachedResult1->p, nullptr);
  state = dd->multiply(densityMatrix1, state, 0, true);
  ASSERT_NE(state.p, nullptr);
  ASSERT_EQ(state.p, cachedResult1->p);

  // try a repeated lookup
  const auto* cachedResult2 = computeTable.lookup(xCopy1, yCopy1, true);
  ASSERT_NE(cachedResult2, nullptr);
  ASSERT_NE(cachedResult2->p, nullptr);
  ASSERT_EQ(cachedResult2->p, cachedResult1->p);

  computeTable.clear();
  const auto* cachedResult3 = computeTable.lookup(xCopy1, yCopy1, true);
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
  const auto operation = dd->makeGateDD(dd::Xmat, nrQubits, 0);
  dd->applyOperationToDensity(state, operation, true);
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

struct StochPackageConfig : public dd::DDPackageConfig {
  static constexpr std::size_t STOCHASTIC_CACHE_OPS = 36;
};

using stochPackage = dd::Package<StochPackageConfig>;

TEST(DDPackageTest, dStochCache) {
  const auto nrQubits = 4U;
  auto dd = std::make_unique<stochPackage>(nrQubits);

  std::vector<dd::mEdge> operations = {};
  operations.emplace_back(dd->makeGateDD(dd::Xmat, nrQubits, 0));
  operations.emplace_back(dd->makeGateDD(dd::Zmat, nrQubits, 1));
  operations.emplace_back(dd->makeGateDD(dd::Ymat, nrQubits, 2));
  operations.emplace_back(dd->makeGateDD(dd::Hmat, nrQubits, 3));

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
  EXPECT_EQ(s.p->v, 1);
  EXPECT_EQ(s.p->e[0].w.r->value, dd::SQRT2_2);
  EXPECT_EQ(s.p->e[0].w.i->value, 0);
  EXPECT_EQ(s.p->e[1].w.r->value, dd::SQRT2_2);
  EXPECT_EQ(s.p->e[1].w.i->value, 0);
  EXPECT_EQ(s.p->e[0].p->e[0].w.r->value, 1);
  EXPECT_EQ(s.p->e[0].p->e[0].w.i->value, 0);
  EXPECT_EQ(s.p->e[0].p->e[1].w.r->value, 0);
  EXPECT_EQ(s.p->e[0].p->e[1].w.i->value, 0);
  EXPECT_EQ(s.p->e[1].p->e[0].w.r->value, 0);
  EXPECT_EQ(s.p->e[1].p->e[0].w.i->value, 0);
  EXPECT_EQ(s.p->e[1].p->e[1].w.r->value, 1);
  EXPECT_EQ(s.p->e[1].p->e[1].w.i->value, 0);
}

TEST(DDPackageTest, stateFromVectorEmpty) {
  auto dd = std::make_unique<dd::Package<>>(1);
  auto v = std::vector<std::complex<dd::fp>>{};
  EXPECT_EQ(dd->makeStateFromVector(v), dd::vEdge::one);
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
    const auto singleSiteX = dd->makeGateDD(dd::Xmat, 1, 0);
    auto globalX = singleSiteX;

    const auto singleSiteZ = dd->makeGateDD(dd::Zmat, 1, 0);
    auto globalZ = singleSiteZ;

    const auto singleSiteHadamard = dd->makeGateDD(dd::Hmat, 1, 0);
    auto globalHadamard = singleSiteHadamard;

    for (dd::Qubit i = 1; i < nrQubits; ++i) {
      globalX = dd->kronecker(globalX, singleSiteX);
      globalZ = dd->kronecker(globalZ, singleSiteZ);
      globalHadamard = dd->kronecker(globalHadamard, singleSiteHadamard);
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
      auto xGate = dd->makeGateDD(dd::Xmat, nrQubits, site);
      auto zGate = dd->makeGateDD(dd::Zmat, nrQubits, site);
      auto hadamard = dd->makeGateDD(dd::Hmat, nrQubits, site);

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
  const auto xGate = dd->makeGateDD(dd::Xmat, nrQubits, 0);

  EXPECT_ANY_THROW(dd->expectationValue(xGate, zeroState));
}

TEST(DDPackageTest, DDFromSingleQubitMatrix) {
  const auto inputMatrix =
      dd::CMat{{dd::SQRT2_2, dd::SQRT2_2}, {dd::SQRT2_2, -dd::SQRT2_2}};

  const auto nrQubits = 1U;
  const auto dd = std::make_unique<dd::Package<>>(nrQubits);
  const auto matDD = dd->makeDDFromMatrix(inputMatrix);

  const auto outputMatrix = dd->getMatrix(matDD);

  EXPECT_EQ(inputMatrix, outputMatrix);
}

TEST(DDPackageTest, DDFromTwoQubitMatrix) {
  const auto inputMatrix =
      dd::CMat{{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 0, 1}, {0, 0, 1, 0}};

  const auto nrQubits = 2U;
  const auto dd = std::make_unique<dd::Package<>>(nrQubits);
  const auto matDD = dd->makeDDFromMatrix(inputMatrix);
  const auto outputMatrix = dd->getMatrix(matDD);

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
  const auto outputMatrix = dd->getMatrix(matDD);

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

  const auto outputMatrix = dd->getMatrix(matDD);

  EXPECT_EQ(inputMatrix, outputMatrix);
}

TEST(DDPackageTest, DDFromEmptyMatrix) {
  const auto inputMatrix = dd::CMat{};

  const auto nrQubits = 3U;
  const auto dd = std::make_unique<dd::Package<>>(nrQubits);
  EXPECT_EQ(dd->makeDDFromMatrix(inputMatrix), dd::mEdge::one);
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

  EXPECT_EQ(dd->makeDDFromMatrix(inputMatrix), dd::mEdge::one);
}

TEST(DDPackageTest, TwoQubitControlledGateDDConstruction) {
  const auto nrQubits = 5U;
  const auto dd = std::make_unique<dd::Package<>>(nrQubits);

  const auto gateMatrices = std::vector{std::pair{dd::Xmat, dd::CXmat},
                                        std::pair{dd::Zmat, dd::CZmat}};

  // For every combination of control and target, test that the DD created by
  // makeTwoQubitGateDD is equal to the DD created by makeGateDD. This should
  // cover every scenario of the makeTwoQubitGateDD function.
  for (const auto& [gateMatrix, controlledGateMatrix] : gateMatrices) {
    for (dd::Qubit control = 0; control < nrQubits; ++control) {
      for (dd::Qubit target = 0; target < nrQubits; ++target) {
        if (control == target) {
          continue;
        }
        const auto controlledGateDD = dd->makeTwoQubitGateDD(
            controlledGateMatrix, nrQubits, control, target);
        const auto gateDD = dd->makeGateDD(
            gateMatrix, nrQubits, qc::Control{static_cast<qc::Qubit>(control)},
            target);
        EXPECT_EQ(controlledGateDD, gateDD);
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
      const auto swapGateDD = dd->makeSWAPDD(nrQubits, control, target);
      const auto gateDD =
          dd->makeSWAPDD(nrQubits, qc::Controls{}, control, target);
      EXPECT_EQ(swapGateDD, gateDD);
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
      const auto iswapGateDD = dd->makeiSWAPDD(nrQubits, control, target);
      const auto gateDD =
          dd->makeiSWAPDD(nrQubits, qc::Controls{}, control, target);
      EXPECT_EQ(iswapGateDD, gateDD);

      const auto iswapInvGateDD = dd->makeiSWAPinvDD(nrQubits, control, target);
      const auto gateInvDD =
          dd->makeiSWAPinvDD(nrQubits, qc::Controls{}, control, target);
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
      const auto dcxGateDD = dd->makeDCXDD(nrQubits, control, target);
      const auto gateDD =
          dd->makeDCXDD(nrQubits, qc::Controls{}, control, target);
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
        const auto rzzGateDD = dd->makeRZZDD(nrQubits, control, target, param);
        const auto gateDD =
            dd->makeRZZDD(nrQubits, qc::Controls{}, control, target, param);
        EXPECT_EQ(rzzGateDD, gateDD);
      }
    }
  }

  auto identity = dd->makeIdent(2);
  auto rzzZero = dd->makeRZZDD(2, 0, 1, 0.);
  EXPECT_EQ(rzzZero, identity);

  auto rzzTwoPi = dd->makeRZZDD(2, 0, 1, 2 * dd::PI);
  EXPECT_EQ(rzzTwoPi.p, identity.p);
  EXPECT_EQ(dd::RealNumber::val(rzzTwoPi.w.r), -1.);

  auto rzzPi = dd->makeRZZDD(2, 0, 1, dd::PI);
  auto zz = dd->makeGateDD(dd::Zmat, 2, qc::Controls{}, 0);
  zz = dd->multiply(zz, dd->makeGateDD(dd::Zmat, 2, qc::Controls{}, 1));
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
        const auto ryyGateDD = dd->makeRYYDD(nrQubits, control, target, param);
        const auto gateDD =
            dd->makeRYYDD(nrQubits, qc::Controls{}, control, target, param);
        EXPECT_EQ(ryyGateDD, gateDD);
      }
    }
  }

  auto identity = dd->makeIdent(2);
  auto ryyZero = dd->makeRYYDD(2, 0, 1, 0.);
  EXPECT_EQ(ryyZero, identity);

  auto ryyPi = dd->makeRYYDD(2, 0, 1, dd::PI);
  auto yy = dd->makeGateDD(dd::Ymat, 2, qc::Controls{}, 0);
  yy = dd->multiply(yy, dd->makeGateDD(dd::Ymat, 2, qc::Controls{}, 1));
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
        const auto rxxGateDD = dd->makeRXXDD(nrQubits, control, target, param);
        const auto gateDD =
            dd->makeRXXDD(nrQubits, qc::Controls{}, control, target, param);
        EXPECT_EQ(rxxGateDD, gateDD);
      }
    }
  }

  auto identity = dd->makeIdent(2);
  auto rxxZero = dd->makeRXXDD(2, 0, 1, 0.);
  EXPECT_EQ(rxxZero, identity);

  auto rxxPi = dd->makeRXXDD(2, 0, 1, dd::PI);
  auto xx = dd->makeGateDD(dd::Xmat, 2, qc::Controls{}, 0);
  xx = dd->multiply(xx, dd->makeGateDD(dd::Xmat, 2, qc::Controls{}, 1));
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
        const auto rzxGateDD = dd->makeRZXDD(nrQubits, control, target, param);
        const auto gateDD =
            dd->makeRZXDD(nrQubits, qc::Controls{}, control, target, param);
        EXPECT_EQ(rzxGateDD, gateDD);
      }
    }
  }

  auto identity = dd->makeIdent(2);
  auto rzxZero = dd->makeRZXDD(2, 0, 1, 0.);
  EXPECT_EQ(rzxZero, identity);

  auto rzxPi = dd->makeRZXDD(2, 0, 1, dd::PI);
  auto zx = dd->makeGateDD(dd::Zmat, 2, qc::Controls{}, 0);
  zx = dd->multiply(zx, dd->makeGateDD(dd::Xmat, 2, qc::Controls{}, 1));
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

      const auto ecrGateDD = dd->makeECRDD(nrQubits, control, target);
      const auto gateDD =
          dd->makeECRDD(nrQubits, qc::Controls{}, control, target);
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
          const auto xxMinusYYGateDD =
              dd->makeXXMinusYYDD(nrQubits, control, target, theta, beta);
          const auto gateDD = dd->makeXXMinusYYDD(nrQubits, qc::Controls{},
                                                  control, target, theta, beta);
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
          const auto xxPlusYYGateDD =
              dd->makeXXPlusYYDD(nrQubits, control, target, theta, beta);
          const auto gateDD = dd->makeXXPlusYYDD(nrQubits, qc::Controls{},
                                                 control, target, theta, beta);
          EXPECT_EQ(xxPlusYYGateDD, gateDD);
        }
      }
    }
  }
}

TEST(DDPackageTest, TwoQubitGateCreationFailure) {
  const auto nrQubits = 1U;
  const auto dd = std::make_unique<dd::Package<>>(nrQubits);

  EXPECT_THROW(dd->makeTwoQubitGateDD(dd::CXmat, 2, 0, 1), std::runtime_error);
}

TEST(DDPackageTest, InnerProductTopNodeConjugation) {
  // Test comes from experimental results
  // 2 qubit state is rotated Rxx(-2) equivalent to
  // Ising model evolution up to a time T=1
  const auto nrQubits = 2U;
  const auto dd = std::make_unique<dd::Package<>>(nrQubits);
  const auto zeroState = dd->makeZeroState(nrQubits);
  const auto rxx = dd->makeRXXDD(nrQubits, 0, 1, -2);
  const auto op = dd->makeGateDD(dd::Zmat, nrQubits, 0);

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
  auto matrix = dd::GateMatrix{dd::complex_one, dd::complex_zero,
                               dd::complex_zero, dd::complex_zero};
  auto dd = std::make_unique<dd::Package<>>(nqubits);

  auto dd1 = dd->makeGateDD(matrix, nqubits, 0U);
  matrix[0] = dd::complex_zero;
  matrix[3] = dd::complex_one;
  auto dd2 = dd->makeGateDD(matrix, nqubits, 0U);
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
  auto matrix = dd::GateMatrix{dd::complex_one, dd::complex_zero,
                               dd::complex_zero, dd::complex_zero};
  auto dd = std::make_unique<dd::Package<>>(nqubits);

  auto dd1 = dd->makeGateDD(matrix, nqubits, 0U);
  matrix[0] = dd::complex_zero;
  matrix[3] = dd::complex_one;
  auto dd2 = dd->makeGateDD(matrix, nqubits, 0U);
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
