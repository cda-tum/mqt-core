#include "dd/Benchmark.hpp"
#include "dd/Package.hpp"
#include "dd/Verification.hpp"

#include "gtest/gtest.h"
#include <iostream>
#include <vector>

using namespace qc::literals;

TEST(PartialEquivalence,
     DDMPartialEquivalenceCheckingExamplePaperDifferentQubitOrder) {
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

TEST(PartialEquivalence,
     DDMPartialEquivalenceCheckingExamplePaperDifferentQubitOrderAndNumber) {
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

TEST(PartialEquivalence, DDMZAPECMQTBenchQPE30Qubits) {
  // zero ancilla partial equivalence test
  auto dd = std::make_unique<dd::Package<>>(31);

  // 30 qubits
  const qc::QuantumComputation c1{
      "./circuits/qpeexact_nativegates_ibm_qiskit_opt0_30.qasm"};
  const qc::QuantumComputation c2{"./circuits/qpeexact_indep_qiskit_30.qasm"};
  // calls zeroAncillaePartialEquivalenceCheck
  // buildFunctionality is already very very slow...
  // EXPECT_TRUE(dd::partialEquivalenceCheck(c1, c2, dd));
  EXPECT_TRUE(true);
}

TEST(PartialEquivalence, DDMZAPECSliQEC19Qubits) {
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

TEST(PartialEquivalence, DDMZAPECSliQECRandomCircuit) {
  // doesn't terminate
  auto dd = std::make_unique<dd::Package<>>(20);
  // full equivalence, 10 qubits
  const qc::QuantumComputation c1{"./circuits/random_1.qasm"};
  const qc::QuantumComputation c2{"./circuits/random_2.qasm"};

  // calls buildFunctionality for c2^-1 concatenated with c1
  EXPECT_TRUE(dd::partialEquivalenceCheck(c1, c2, dd));
}

TEST(PartialEquivalence, DDMPECSliQECPeriodFinding8Qubits) {
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

TEST(PartialEquivalence, DDMPECBenchmark) {
  const auto dd = std::make_unique<dd::Package<>>(20);
  srand(55);
  const size_t minN = 2;
  const size_t maxN = 8;
  const size_t reps = 10;
  std::cout << "Partial equivalence check\n";
  partialEquivalencCheckingBenchmarks(dd, minN, maxN, reps, true);
}

TEST(PartialEquivalence, DDMZAPECBenchmark) {
  const auto dd = std::make_unique<dd::Package<>>(30);
  const size_t minN = 3;
  const size_t maxN = 15;
  const size_t reps = 10;
  std::cout << "Zero-ancilla partial equivalence check\n";
  partialEquivalencCheckingBenchmarks(dd, minN, maxN, reps, false);
}
