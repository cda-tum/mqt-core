#include "dd/Benchmark.hpp"
#include "dd/Package.hpp"
#include "dd/Verification.hpp"

#include "gtest/gtest.h"
#include <iostream>
#include <vector>

using namespace qc::literals;

TEST(PartialEquivalenceTest, TrivialEquivalence) {
  const auto nqubits = 2U;
  const auto dd = std::make_unique<dd::Package<>>(nqubits);
  const auto inputMatrix =
      dd::CMat{{1, 1, 1, 1}, {1, -1, 1, -1}, {1, 1, -1, -1}, {1, -1, -1, 1}};
  const auto inputDD = dd->makeDDFromMatrix(inputMatrix);

  EXPECT_TRUE(dd->partialEquivalenceCheck(inputDD, inputDD, 1, 1));
  EXPECT_TRUE(dd->partialEquivalenceCheck(inputDD, inputDD, 2, 1));
  EXPECT_TRUE(dd->partialEquivalenceCheck(inputDD, inputDD, 1, 2));
  EXPECT_TRUE(dd->partialEquivalenceCheck(inputDD, inputDD, 2, 2));

  const auto hGate = dd->makeGateDD(dd::H_MAT, 2, 1);
  const auto cxGate = dd->makeGateDD(dd::X_MAT, 2, 1_pc, 0);
  const auto bellMatrix = dd->multiply(cxGate, hGate);
  EXPECT_FALSE(dd->partialEquivalenceCheck(inputDD, bellMatrix, 1, 1));
}

TEST(PartialEquivalenceTest, BasicPartialEquivalenceChecking) {
  const auto nqubits = 3U;
  auto dd = std::make_unique<dd::Package<>>(nqubits);
  // only the second qubit has differing gates in the two circuits,
  // therefore they should be equivalent if we only measure the first qubit
  const auto hGate = dd->makeGateDD(dd::H_MAT, 3, 1);
  const auto xGate = dd->makeGateDD(dd::X_MAT, 3, 1);
  const auto circuit1 = dd->multiply(xGate, hGate);
  const auto circuit2 = dd->makeIdent(3);

  EXPECT_TRUE(dd->partialEquivalenceCheck(circuit1, circuit2, 2, 1));
}

TEST(PartialEquivalenceTest, NotEquivalent) {
  const auto nqubits = 2U;
  const auto dd = std::make_unique<dd::Package<>>(nqubits);
  // the first qubit has differing gates in the two circuits,
  // therefore they should not be equivalent if we only measure the first qubit
  const auto hGate = dd->makeGateDD(dd::H_MAT, nqubits, 0);
  const auto xGate = dd->makeGateDD(dd::X_MAT, nqubits, 0);
  const auto circuit1 = dd->multiply(xGate, hGate);
  const auto circuit2 = dd->makeIdent(2);
  EXPECT_FALSE(dd->partialEquivalenceCheck(circuit1, circuit2, 2, 1));
  EXPECT_FALSE(dd->zeroAncillaePartialEquivalenceCheck(circuit1, circuit2, 1));
}

TEST(PartialEquivalenceTest, ExamplePaper) {
  const auto nqubits = 3U;
  const auto dd = std::make_unique<dd::Package<>>(nqubits);
  const auto controlledSwapGate =
      dd->makeTwoQubitGateDD(dd::SWAP_MAT, nqubits, qc::Controls{1}, 0, 2);
  const auto hGate = dd->makeGateDD(dd::H_MAT, nqubits, 0);
  const auto zGate = dd->makeGateDD(dd::Z_MAT, nqubits, 2);
  const auto xGate = dd->makeGateDD(dd::X_MAT, nqubits, 1);
  const auto controlledHGate =
      dd->makeGateDD(dd::H_MAT, nqubits, qc::Controls{1}, 0);

  const auto c1 = dd->multiply(
      controlledSwapGate,
      dd->multiply(hGate, dd->multiply(zGate, controlledSwapGate)));
  const auto c2 = dd->multiply(controlledHGate, xGate);

  EXPECT_TRUE(dd->partialEquivalenceCheck(c1, c2, 3, 1));
}

TEST(PartialEquivalenceTest, ExamplePaperZeroAncillae) {
  const auto nqubits = 3U;
  const auto dd = std::make_unique<dd::Package<>>(nqubits);
  const auto controlledSwapGate =
      dd->makeTwoQubitGateDD(dd::SWAP_MAT, nqubits, qc::Controls{1}, 0, 2);
  const auto hGate = dd->makeGateDD(dd::H_MAT, nqubits, 0);
  const auto zGate = dd->makeGateDD(dd::Z_MAT, nqubits, 2);
  const auto xGate = dd->makeGateDD(dd::X_MAT, nqubits, 1);
  const auto controlledHGate =
      dd->makeGateDD(dd::H_MAT, nqubits, qc::Controls{1}, 0);

  const auto c1 = dd->multiply(
      controlledSwapGate,
      dd->multiply(hGate, dd->multiply(zGate, controlledSwapGate)));
  const auto c2 = dd->multiply(controlledHGate, xGate);

  EXPECT_TRUE(dd->zeroAncillaePartialEquivalenceCheck(c1, c2, 1));
  EXPECT_FALSE(dd->zeroAncillaePartialEquivalenceCheck(c1, c2, 2));

  const auto hGate2 = dd->makeGateDD(dd::H_MAT, nqubits, 2);
  const auto zGate2 = dd->makeGateDD(dd::Z_MAT, nqubits, 0);
  const auto controlledHGate2 =
      dd->makeGateDD(dd::H_MAT, nqubits, qc::Controls{1}, 0);

  const auto c3 = dd->multiply(
      controlledSwapGate,
      dd->multiply(hGate2, dd->multiply(zGate2, controlledSwapGate)));
  const auto c4 = dd->multiply(controlledHGate2, xGate);

  EXPECT_FALSE(dd->zeroAncillaePartialEquivalenceCheck(c3, c4, 1));
}

TEST(PartialEquivalenceTest, DifferentNumberOfQubits) {
  const auto dd = std::make_unique<dd::Package<>>(3);
  const auto controlledSwapGate =
      dd->makeTwoQubitGateDD(dd::SWAP_MAT, 3, qc::Controls{1}, 0, 2);
  const auto hGate = dd->makeGateDD(dd::H_MAT, 3, 0);
  const auto zGate = dd->makeGateDD(dd::Z_MAT, 3, 2);
  const auto xGate = dd->makeGateDD(dd::X_MAT, 2, 1);
  const auto controlledHGate = dd->makeGateDD(dd::H_MAT, 2, qc::Controls{1}, 0);

  const auto c1 = dd->multiply(
      controlledSwapGate,
      dd->multiply(hGate, dd->multiply(zGate, controlledSwapGate)));
  const auto c2 = dd->multiply(controlledHGate, xGate);

  EXPECT_TRUE(dd->partialEquivalenceCheck(c1, c2, 3, 1));
  EXPECT_FALSE(dd->partialEquivalenceCheck(c2, c1, 3, 3));
  EXPECT_FALSE(dd->partialEquivalenceCheck(c2, dd::mEdge::zero(), 2, 1));
  EXPECT_FALSE(dd->partialEquivalenceCheck(c2, dd::mEdge::one(), 2, 1));
  EXPECT_FALSE(dd->partialEquivalenceCheck(dd::mEdge::one(), c1, 2, 1));
  EXPECT_TRUE(
      dd->partialEquivalenceCheck(dd::mEdge::one(), dd::mEdge::one(), 0, 1));
  EXPECT_TRUE(
      dd->partialEquivalenceCheck(dd::mEdge::one(), dd::mEdge::one(), 0, 0));
}

TEST(PartialEquivalenceTest, ComputeTableTest) {
  const auto nqubits = 3U;
  const auto dd = std::make_unique<dd::Package<>>(nqubits);
  const auto controlledSwapGate =
      dd->makeTwoQubitGateDD(dd::SWAP_MAT, nqubits, qc::Controls{1}, 2, 0);
  const auto hGate = dd->makeGateDD(dd::H_MAT, nqubits, 0);
  const auto zGate = dd->makeGateDD(dd::Z_MAT, nqubits, 2);
  const auto xGate = dd->makeGateDD(dd::X_MAT, nqubits, 1);
  const auto controlledHGate =
      dd->makeGateDD(dd::H_MAT, nqubits, qc::Controls{1}, 0);

  const auto c1 = dd->multiply(
      controlledSwapGate,
      dd->multiply(hGate, dd->multiply(zGate, controlledSwapGate)));
  const auto c2 = dd->multiply(controlledHGate, xGate);

  EXPECT_TRUE(dd->partialEquivalenceCheck(c1, c2, 3, 1));
  EXPECT_TRUE(dd->partialEquivalenceCheck(c1, c2, 3, 1));
  EXPECT_TRUE(dd->partialEquivalenceCheck(c1, c2, 3, 1));
  EXPECT_TRUE(dd->partialEquivalenceCheck(c1, c2, 3, 1));
  EXPECT_TRUE(dd->partialEquivalenceCheck(c1, c2, 3, 1));
}

TEST(PartialEquivalenceTest, MQTBenchGrover3Qubits) {
  const auto dd = std::make_unique<dd::Package<>>(7);

  const qc::QuantumComputation c1{
      "./circuits/grover-noancilla_nativegates_ibm_qiskit_opt0_3.qasm"};
  const qc::QuantumComputation c2{
      "./circuits/grover-noancilla_indep_qiskit_3.qasm"};

  // 3 measured qubits and 3 data qubits, full equivalence
  EXPECT_TRUE(dd->partialEquivalenceCheck(
      buildFunctionality(&c1, *dd, false, false),
      buildFunctionality(&c2, *dd, false, false), 3, 3));
}

TEST(PartialEquivalenceTest, MQTBenchGrover7Qubits) {
  const auto dd = std::make_unique<dd::Package<>>(7);

  const qc::QuantumComputation c1{
      "./circuits/grover-noancilla_nativegates_ibm_qiskit_opt0_7.qasm"};
  const qc::QuantumComputation c2{
      "./circuits/grover-noancilla_nativegates_ibm_qiskit_opt1_7.qasm"};

  // 7 measured qubits and 7 data qubits, full equivalence
  EXPECT_TRUE(dd->partialEquivalenceCheck(
      buildFunctionality(&c1, *dd, false, false),
      buildFunctionality(&c2, *dd, false, false), 7, 7));
}

TEST(PartialEquivalenceTest, SliQECGrover22Qubits) {
  const auto dd = std::make_unique<dd::Package<>>(22);

  const qc::QuantumComputation c1{
      "./circuits/Grover_1.qasm"}; // 11 qubits, 11 data qubits
  const qc::QuantumComputation c2{
      "./circuits/Grover_2.qasm"}; // 12 qubits, 11 data qubits

  // 11 measured qubits and 11 data qubits
  const auto c1Dd = buildFunctionality(&c1, *dd, false, false);
  const auto c2Dd = buildFunctionality(&c2, *dd, false, false);
  // adds 10 ancillary qubits -> total number of qubits is 22
  EXPECT_TRUE(dd->partialEquivalenceCheck(c1Dd, c2Dd, 11, 11));
}

TEST(PartialEquivalenceTest, SliQECAdd19Qubits) {
  const auto dd = std::make_unique<dd::Package<>>(20);

  // full equivalence, 19 qubits
  // but this test uses algorithm for partial equivalence, not the "zero
  // ancillae" version
  const qc::QuantumComputation c1{"./circuits/add6_196_1.qasm"};
  const qc::QuantumComputation c2{"./circuits/add6_196_2.qasm"};

  // just for benchmarking reasons, we only measure 8 qubits
  const auto c1Dd = buildFunctionality(&c1, *dd, false, false);
  const auto c2Dd = buildFunctionality(&c2, *dd, false, false);
  // doesn't add ancillary qubits -> total number of qubits is 19
  EXPECT_TRUE(dd->partialEquivalenceCheck(c1Dd, c2Dd, 8, 8));
}

TEST(PartialEquivalenceTest, ExamplePaperDifferentQubitOrder) {
  const auto nqubits = 3U;
  auto dd = std::make_unique<dd::Package<>>(nqubits);

  qc::QuantumComputation c1{3, 1};
  c1.cswap(1, 2, 0);
  c1.h(2);
  c1.z(0);
  c1.cswap(1, 2, 0);

  qc::QuantumComputation c2{3, 1};
  c2.x(1);
  c2.ch(1, 2);

  c1.setLogicalQubitGarbage(1);
  c1.setLogicalQubitGarbage(0);

  c2.setLogicalQubitGarbage(1);
  c2.setLogicalQubitGarbage(0);
  EXPECT_TRUE(dd::partialEquivalenceCheck(c1, c2, dd));
}

TEST(PartialEquivalenceTest, ExamplePaperDifferentQubitOrderAndNumber) {
  const auto nqubits = 4U;
  auto dd = std::make_unique<dd::Package<>>(nqubits);

  qc::QuantumComputation c1{4, 1};
  c1.cswap(1, 2, 0);
  c1.h(2);
  c1.z(0);
  c1.cswap(1, 2, 0);

  qc::QuantumComputation c2{3, 1};
  c2.x(1);
  c2.ch(1, 2);

  c1.setLogicalQubitGarbage(1);
  c1.setLogicalQubitGarbage(0);
  c1.setLogicalQubitGarbage(3);
  c1.setLogicalQubitAncillary(3);

  c2.setLogicalQubitGarbage(1);
  c2.setLogicalQubitGarbage(0);
  EXPECT_TRUE(dd::partialEquivalenceCheck(c1, c2, dd));
  EXPECT_TRUE(dd::partialEquivalenceCheck(c2, c1, dd));
}

TEST(PartialEquivalenceTest, ZeroAncillaSliQEC19Qubits) {
  auto dd = std::make_unique<dd::Package<>>(20);

  // full equivalence, 10 qubits
  const qc::QuantumComputation c1{"./circuits/entanglement_1.qasm"};
  const qc::QuantumComputation c2{"./circuits/entanglement_2.qasm"};

  // calls zeroAncillaePartialEquivalenceCheck
  EXPECT_TRUE(dd::partialEquivalenceCheck(c1, c2, dd));

  // full equivalence, 19 qubits
  const qc::QuantumComputation c3{"./circuits/add6_196_1.qasm"};
  const qc::QuantumComputation c4{"./circuits/add6_196_2.qasm"};

  // calls zeroAncillaePartialEquivalenceCheck
  EXPECT_TRUE(dd::partialEquivalenceCheck(c3, c4, dd));

  // full equivalence, 10 qubits
  const qc::QuantumComputation c5{"./circuits/bv_1.qasm"};
  const qc::QuantumComputation c6{"./circuits/bv_2.qasm"};

  // calls zeroAncillaePartialEquivalenceCheck
  EXPECT_TRUE(dd::partialEquivalenceCheck(c5, c6, dd));
}

TEST(PartialEquivalenceTest, ZeroAncillaSliQECRandomCircuit) {
  auto dd = std::make_unique<dd::Package<>>(20);
  // full equivalence, 10 qubits
  const qc::QuantumComputation c1{"./circuits/random_1.qasm"};
  const qc::QuantumComputation c2{"./circuits/random_2.qasm"};

  // calls buildFunctionality for c2^(-1) concatenated with c1
  EXPECT_TRUE(dd::partialEquivalenceCheck(c1, c2, dd));
}

TEST(PartialEquivalenceTest, SliQECPeriodFinding8Qubits) {
  auto dd = std::make_unique<dd::Package<>>(20);
  // 8 qubits, 3 data qubits
  qc::QuantumComputation c1{"./circuits/period_finding_1.qasm"};
  // 8 qubits, 3 data qubits
  qc::QuantumComputation c2{"./circuits/period_finding_2.qasm"};

  // 3 measured qubits and 3 data qubits

  c2.setLogicalQubitAncillary(7);
  c2.setLogicalQubitGarbage(7);
  c2.setLogicalQubitAncillary(6);
  c2.setLogicalQubitGarbage(6);
  c2.setLogicalQubitAncillary(5);
  c2.setLogicalQubitGarbage(5);
  c2.setLogicalQubitAncillary(3);
  c2.setLogicalQubitGarbage(3);
  c2.setLogicalQubitAncillary(4);
  c2.setLogicalQubitGarbage(4);

  c1.setLogicalQubitAncillary(7);
  c1.setLogicalQubitGarbage(7);
  c1.setLogicalQubitAncillary(6);
  c1.setLogicalQubitGarbage(6);
  c1.setLogicalQubitAncillary(5);
  c1.setLogicalQubitGarbage(5);
  c1.setLogicalQubitAncillary(3);
  c1.setLogicalQubitGarbage(3);
  c1.setLogicalQubitAncillary(4);
  c1.setLogicalQubitGarbage(4);
  EXPECT_TRUE(dd::partialEquivalenceCheck(c1, c2, dd));
}

void partialEquivalencCheckingBenchmarks(
    const std::unique_ptr<dd::Package<dd::DDPackageConfig>>& dd,
    const size_t minN, const size_t maxN, const size_t reps,
    const bool addAncilla) {
  for (size_t n = minN; n < maxN; n++) {
    std::chrono::microseconds totalTime{0};
    std::uint16_t totalGates{0};
    for (size_t k = 0; k < reps; k++) {
      dd::Qubit d{0};
      if (addAncilla) {
        d = static_cast<dd::Qubit>(rand()) % static_cast<dd::Qubit>(n - 1) + 1;
      } else {
        d = static_cast<dd::Qubit>(n);
      }
      dd::Qubit m{0};
      if (d == 1) {
        m = 1;
      } else {
        m = static_cast<dd::Qubit>(rand()) % static_cast<dd::Qubit>(d - 1) + 1;
      }
      const auto [c1, c2] = dd::generateRandomBenchmark(n, d, m);

      const auto start = std::chrono::high_resolution_clock::now();
      const bool result = dd::partialEquivalenceCheck(c1, c2, dd);
      // Get ending timepoint
      const auto stop = std::chrono::high_resolution_clock::now();
      const auto duration =
          std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

      EXPECT_TRUE(result);

      // std::cout << "\nnumber of qubits = " << n << "; data qubits = " << d
      //           << "; measured qubits = " << m
      //           << "; number of gates = " << c2.size() << "\n";
      // std::cout << "time: " << static_cast<double>(duration.count()) /
      // 1000000.
      //           << " seconds\n";
      totalTime += duration;
      totalGates += static_cast<std::uint16_t>(c2.size());
    }
    std::cout << "\nnumber of qubits = " << n << "; number of reps = " << reps
              << "; average time = "
              << (static_cast<double>(totalTime.count()) /
                  static_cast<double>(reps) / 1000000.)
              << " seconds; average number of gates = "
              << (static_cast<double>(totalGates) / static_cast<double>(reps))
              << "\n";
  }
}

TEST(PartialEquivalenceTest, Benchmark) {
  const auto dd = std::make_unique<dd::Package<>>(20);
  srand(55);
  const size_t minN = 2;
  const size_t maxN = 8;
  const size_t reps = 10;
  std::cout << "Partial equivalence check\n";
  partialEquivalencCheckingBenchmarks(dd, minN, maxN, reps, true);
}

TEST(PartialEquivalenceTest, ZeroAncillaBenchmark) {
  const auto dd = std::make_unique<dd::Package<>>(30);
  const size_t minN = 3;
  const size_t maxN = 15;
  const size_t reps = 10;
  std::cout << "Zero-ancilla partial equivalence check\n";
  partialEquivalencCheckingBenchmarks(dd, minN, maxN, reps, false);
}

TEST(PartialEquivalenceTest, InvalidInput) {
  const auto dd = std::make_unique<dd::Package<>>(30);

  // the circuits don't have the same number of measured qubits
  qc::QuantumComputation c1{4, 1};
  c1.x(1);

  qc::QuantumComputation c2{4, 1};
  c2.x(1);

  c1.setLogicalQubitGarbage(1);
  c1.setLogicalQubitGarbage(0);
  c1.setLogicalQubitGarbage(3);

  c2.setLogicalQubitGarbage(1);
  c2.setLogicalQubitGarbage(0);

  EXPECT_FALSE(dd::partialEquivalenceCheck(c1, c2, dd));

  // now they have the same number of measured qubits but a different
  // permutation of garbage qubits
  c2.setLogicalQubitGarbage(2);
  EXPECT_THROW(dd::partialEquivalenceCheck(c1, c2, dd), std::invalid_argument);
}
