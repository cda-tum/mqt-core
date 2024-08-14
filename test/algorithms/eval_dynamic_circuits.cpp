#include "Definitions.hpp"
#include "algorithms/BernsteinVazirani.hpp"
#include "algorithms/QFT.hpp"
#include "algorithms/QPE.hpp"
#include "circuit_optimizer/CircuitOptimizer.hpp"
#include "dd/DDDefinitions.hpp"
#include "dd/Operations.hpp"
#include "dd/Package.hpp"
#include "dd/Simulation.hpp"
#include "ir/QuantumComputation.hpp"

#include <bitset>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

class DynamicCircuitEvalExactQPE : public testing::TestWithParam<std::size_t> {
protected:
  std::size_t precision{};
  qc::fp theta{};
  std::size_t expectedResult{};
  std::string expectedResultRepresentation;
  std::unique_ptr<qc::QuantumComputation> qpe;
  std::unique_ptr<qc::QuantumComputation> iqpe;
  std::size_t qpeNgates{};
  std::size_t iqpeNgates{};
  std::unique_ptr<dd::Package<>> dd;
  std::ofstream ofs;

  void TearDown() override {}
  void SetUp() override {
    precision = GetParam();

    dd = std::make_unique<dd::Package<>>(precision + 1);

    qpe = std::make_unique<qc::QPE>(precision);
    // remove final measurements so that the functionality is unitary
    qc::CircuitOptimizer::removeFinalMeasurements(*qpe);
    qpeNgates = qpe->getNindividualOps();

    const auto lambda = dynamic_cast<qc::QPE*>(qpe.get())->lambda;
    iqpe = std::make_unique<qc::QPE>(lambda, precision, true);
    iqpeNgates = iqpe->getNindividualOps();

    std::cout << "Estimating lambda = " << lambda << "π up to " << precision
              << "-bit precision.\n";

    theta = lambda / 2;

    std::cout << "Expected theta=" << theta << "\n";
    std::bitset<64> binaryExpansion{};
    dd::fp expansion = theta * 2;
    std::size_t index = 0;
    while (std::abs(expansion) > 1e-8) {
      if (expansion >= 1.) {
        binaryExpansion.set(index);
        expansion -= 1.0;
      }
      index++;
      expansion *= 2;
    }

    expectedResult = 0ULL;
    for (std::size_t i = 0; i < precision; ++i) {
      if (binaryExpansion.test(i)) {
        expectedResult |= (1ULL << (precision - 1 - i));
      }
    }
    std::stringstream ss{};
    for (auto i = static_cast<int>(precision - 1); i >= 0; --i) {
      if ((expectedResult & (1ULL << static_cast<std::size_t>(i))) != 0) {
        ss << 1;
      } else {
        ss << 0;
      }
    }
    expectedResultRepresentation = ss.str();

    std::cout << "Theta is exactly representable using " << precision
              << " bits.\n";
    std::cout << "The expected output state is |"
              << expectedResultRepresentation << ">.\n";
  }
};

INSTANTIATE_TEST_SUITE_P(
    Eval, DynamicCircuitEvalExactQPE, testing::Range<std::size_t>(1U, 64U, 5U),
    [](const testing::TestParamInfo<DynamicCircuitEvalExactQPE::ParamType>&
           inf) {
      const auto nqubits = inf.param;
      std::stringstream ss{};
      ss << nqubits;
      if (nqubits == 1) {
        ss << "_qubit";
      } else {
        ss << "_qubits";
      }
      return ss.str();
    });

TEST_P(DynamicCircuitEvalExactQPE, UnitaryTransformation) {
  qpe->reorderOperations();
  const auto start = std::chrono::steady_clock::now();
  // transform dynamic circuit to unitary circuit by first eliminating reset
  // operations and afterwards deferring measurements to the end of the circuit
  qc::CircuitOptimizer::eliminateResets(*iqpe);
  qc::CircuitOptimizer::deferMeasurements(*iqpe);

  // remove final measurements in order to just obtain the unitary functionality
  qc::CircuitOptimizer::removeFinalMeasurements(*iqpe);
  iqpe->reorderOperations();
  const auto finishedTransformation = std::chrono::steady_clock::now();

  qc::MatrixDD e = dd->makeIdent();
  dd->incRef(e);

  auto leftIt = qpe->begin();
  auto rightIt = iqpe->begin();

  while (leftIt != qpe->end() && rightIt != iqpe->end()) {
    auto multLeft = dd->multiply(getDD((*leftIt).get(), *dd), e);
    auto multRight =
        dd->multiply(multLeft, getInverseDD((*rightIt).get(), *dd));
    dd->incRef(multRight);
    dd->decRef(e);
    e = multRight;

    dd->garbageCollect();

    ++leftIt;
    ++rightIt;
  }

  while (leftIt != qpe->end()) {
    auto multLeft = dd->multiply(getDD((*leftIt).get(), *dd), e);
    dd->incRef(multLeft);
    dd->decRef(e);
    e = multLeft;

    dd->garbageCollect();

    ++leftIt;
  }

  while (rightIt != iqpe->end()) {
    auto multRight = dd->multiply(e, getInverseDD((*rightIt).get(), *dd));
    dd->incRef(multRight);
    dd->decRef(e);
    e = multRight;

    dd->garbageCollect();

    ++rightIt;
  }
  const auto finishedEC = std::chrono::steady_clock::now();

  const auto preprocessing =
      std::chrono::duration<double>(finishedTransformation - start).count();
  const auto verification =
      std::chrono::duration<double>(finishedEC - finishedTransformation)
          .count();

  std::stringstream ss{};
  ss << "qpe_exact,transformation," << qpe->getNqubits() << "," << qpeNgates
     << ",2," << iqpeNgates << "," << preprocessing << "," << verification;
  std::cout << ss.str() << "\n";

  EXPECT_TRUE(e.isIdentity());
}

TEST_P(DynamicCircuitEvalExactQPE, ProbabilityExtraction) {
  // generate DD of QPE circuit via simulation
  const auto start = std::chrono::steady_clock::now();
  auto e = simulate(qpe.get(), dd->makeZeroState(qpe->getNqubits()), *dd);
  const auto simulationEnd = std::chrono::steady_clock::now();

  // extract measurement probabilities from IQPE simulations
  dd::SparsePVec probs{};
  extractProbabilityVector(iqpe.get(), dd->makeZeroState(iqpe->getNqubits()),
                           probs, *dd);
  const auto extractionEnd = std::chrono::steady_clock::now();

  // compare outcomes
  auto fidelity =
      dd->fidelityOfMeasurementOutcomes(e, probs, qpe->outputPermutation);
  const auto comparisonEnd = std::chrono::steady_clock::now();

  const auto simulation =
      std::chrono::duration<double>(simulationEnd - start).count();
  const auto extraction =
      std::chrono::duration<double>(extractionEnd - simulationEnd).count();
  const auto comparison =
      std::chrono::duration<double>(comparisonEnd - extractionEnd).count();
  const auto total =
      std::chrono::duration<double>(comparisonEnd - start).count();

  std::stringstream ss{};
  ss << "qpe_exact,extraction," << qpe->getNqubits() << "," << qpeNgates
     << ",2," << iqpeNgates << "," << simulation << "," << extraction << ","
     << comparison << "," << total;
  std::cout << ss.str() << "\n";

  EXPECT_NEAR(fidelity, 1.0, 1e-4);
}

class DynamicCircuitEvalInexactQPE
    : public testing::TestWithParam<std::size_t> {
protected:
  std::size_t precision{};
  dd::fp theta{};
  std::size_t expectedResult{};
  std::string expectedResultRepresentation;
  std::size_t secondExpectedResult{};
  std::string secondExpectedResultRepresentation;
  std::unique_ptr<qc::QuantumComputation> qpe;
  std::unique_ptr<qc::QuantumComputation> iqpe;
  std::size_t qpeNgates{};
  std::size_t iqpeNgates{};
  std::unique_ptr<dd::Package<>> dd;
  std::ofstream ofs;

  void TearDown() override {}
  void SetUp() override {
    precision = GetParam();

    dd = std::make_unique<dd::Package<>>(precision + 1);

    qpe = std::make_unique<qc::QPE>(precision, false);
    // remove final measurements so that the functionality is unitary
    qc::CircuitOptimizer::removeFinalMeasurements(*qpe);
    qpeNgates = qpe->getNindividualOps();

    const auto lambda = dynamic_cast<qc::QPE*>(qpe.get())->lambda;
    iqpe = std::make_unique<qc::QPE>(lambda, precision, true);
    iqpeNgates = iqpe->getNindividualOps();

    std::cout << "Estimating lambda = " << lambda << "π up to " << precision
              << "-bit precision.\n";

    theta = lambda / 2;

    std::cout << "Expected theta=" << theta << "\n";
    std::bitset<64> binaryExpansion{};
    dd::fp expansion = theta * 2;
    std::size_t index = 0;
    while (std::abs(expansion) > 1e-8) {
      if (expansion >= 1.) {
        binaryExpansion.set(index);
        expansion -= 1.0;
      }
      index++;
      expansion *= 2;
    }

    expectedResult = 0ULL;
    for (std::size_t i = 0; i < precision; ++i) {
      if (binaryExpansion.test(i)) {
        expectedResult |= (1ULL << (precision - 1 - i));
      }
    }
    std::stringstream ss{};
    for (auto i = precision; i > 0; --i) {
      if ((expectedResult & (1ULL << (i - 1))) != 0) {
        ss << 1;
      } else {
        ss << 0;
      }
    }
    expectedResultRepresentation = ss.str();

    secondExpectedResult = expectedResult + 1;
    ss.str("");
    for (auto i = precision; i > 0; --i) {
      if ((secondExpectedResult & (1ULL << (i - 1))) != 0) {
        ss << 1;
      } else {
        ss << 0;
      }
    }
    secondExpectedResultRepresentation = ss.str();

    std::cout << "Theta is not exactly representable using " << precision
              << " bits.\n";
    std::cout << "Most probable output states are |"
              << expectedResultRepresentation << "> and |"
              << secondExpectedResultRepresentation << ">.\n";
  }
};

INSTANTIATE_TEST_SUITE_P(Eval, DynamicCircuitEvalInexactQPE,
                         testing::Range<std::size_t>(1U, 15U, 3U),
                         [](const testing::TestParamInfo<
                             DynamicCircuitEvalInexactQPE::ParamType>& inf) {
                           const auto nqubits = inf.param;
                           std::stringstream ss{};
                           ss << nqubits;
                           if (nqubits == 1) {
                             ss << "_qubit";
                           } else {
                             ss << "_qubits";
                           }
                           return ss.str();
                         });

TEST_P(DynamicCircuitEvalInexactQPE, UnitaryTransformation) {
  qpe->reorderOperations();
  const auto start = std::chrono::steady_clock::now();
  // transform dynamic circuit to unitary circuit by first eliminating reset
  // operations and afterwards deferring measurements to the end of the circuit
  qc::CircuitOptimizer::eliminateResets(*iqpe);
  qc::CircuitOptimizer::deferMeasurements(*iqpe);

  // remove final measurements in order to just obtain the unitary functionality
  qc::CircuitOptimizer::removeFinalMeasurements(*iqpe);
  iqpe->reorderOperations();
  const auto finishedTransformation = std::chrono::steady_clock::now();

  qc::MatrixDD e = dd->makeIdent();
  dd->incRef(e);

  auto leftIt = qpe->begin();
  auto rightIt = iqpe->begin();

  while (leftIt != qpe->end() && rightIt != iqpe->end()) {
    auto multLeft = dd->multiply(getDD((*leftIt).get(), *dd), e);
    auto multRight =
        dd->multiply(multLeft, getInverseDD((*rightIt).get(), *dd));
    dd->incRef(multRight);
    dd->decRef(e);
    e = multRight;

    dd->garbageCollect();

    ++leftIt;
    ++rightIt;
  }

  while (leftIt != qpe->end()) {
    auto multLeft = dd->multiply(getDD((*leftIt).get(), *dd), e);
    dd->incRef(multLeft);
    dd->decRef(e);
    e = multLeft;

    dd->garbageCollect();

    ++leftIt;
  }

  while (rightIt != iqpe->end()) {
    auto multRight = dd->multiply(e, getInverseDD((*rightIt).get(), *dd));
    dd->incRef(multRight);
    dd->decRef(e);
    e = multRight;

    dd->garbageCollect();

    ++rightIt;
  }
  const auto finishedEC = std::chrono::steady_clock::now();

  const auto preprocessing =
      std::chrono::duration<double>(finishedTransformation - start).count();
  const auto verification =
      std::chrono::duration<double>(finishedEC - finishedTransformation)
          .count();

  std::stringstream ss{};
  ss << "qpe_inexact,transformation," << qpe->getNqubits() << "," << qpeNgates
     << ",2," << iqpeNgates << "," << preprocessing << "," << verification;
  std::cout << ss.str() << "\n";

  EXPECT_TRUE(e.isIdentity());
}

TEST_P(DynamicCircuitEvalInexactQPE, ProbabilityExtraction) {
  const auto start = std::chrono::steady_clock::now();
  // extract measurement probabilities from IQPE simulations
  dd::SparsePVec probs{};
  extractProbabilityVector(iqpe.get(), dd->makeZeroState(iqpe->getNqubits()),
                           probs, *dd);
  const auto extractionEnd = std::chrono::steady_clock::now();
  std::cout << "---- extraction done ----\n";

  // generate DD of QPE circuit via simulation
  auto e = simulate(qpe.get(), dd->makeZeroState(qpe->getNqubits()), *dd);
  const auto simulationEnd = std::chrono::steady_clock::now();
  std::cout << "---- sim done ----\n";

  // compare outcomes
  auto fidelity =
      dd->fidelityOfMeasurementOutcomes(e, probs, qpe->outputPermutation);
  const auto comparisonEnd = std::chrono::steady_clock::now();

  const auto extraction =
      std::chrono::duration<double>(extractionEnd - start).count();
  const auto simulation =
      std::chrono::duration<double>(simulationEnd - extractionEnd).count();
  const auto comparison =
      std::chrono::duration<double>(comparisonEnd - simulationEnd).count();
  const auto total =
      std::chrono::duration<double>(comparisonEnd - start).count();

  std::stringstream ss{};
  ss << "qpe_inexact,extraction," << qpe->getNqubits() << "," << qpeNgates
     << ",2," << iqpeNgates << "," << simulation << "," << extraction << ","
     << comparison << "," << total;
  std::cout << ss.str() << "\n";

  EXPECT_NEAR(fidelity, 1.0, 1e-4);
}

class DynamicCircuitEvalBV : public testing::TestWithParam<std::size_t> {
protected:
  std::size_t bitwidth{};
  std::unique_ptr<qc::QuantumComputation> bv;
  std::unique_ptr<qc::QuantumComputation> dbv;
  std::size_t bvNgates{};
  std::size_t dbvNgates{};
  std::unique_ptr<dd::Package<>> dd;
  std::ofstream ofs;

  void TearDown() override {}
  void SetUp() override {
    bitwidth = GetParam();

    dd = std::make_unique<dd::Package<>>(bitwidth + 1);

    bv = std::make_unique<qc::BernsteinVazirani>(bitwidth);
    // remove final measurements so that the functionality is unitary
    qc::CircuitOptimizer::removeFinalMeasurements(*bv);
    bvNgates = bv->getNindividualOps();

    const auto s = dynamic_cast<qc::BernsteinVazirani*>(bv.get())->s;
    dbv = std::make_unique<qc::BernsteinVazirani>(s, bitwidth, true);
    dbvNgates = dbv->getNindividualOps();

    const auto expected =
        dynamic_cast<qc::BernsteinVazirani*>(bv.get())->expected;
    std::cout << "Hidden bitstring: " << expected << " (" << bitwidth
              << " qubits)\n";
  }
};

INSTANTIATE_TEST_SUITE_P(
    Eval, DynamicCircuitEvalBV, testing::Range<std::size_t>(1U, 64U, 5U),
    [](const testing::TestParamInfo<DynamicCircuitEvalExactQPE::ParamType>&
           inf) {
      const auto nqubits = inf.param;
      std::stringstream ss{};
      ss << nqubits;
      if (nqubits == 1) {
        ss << "_qubit";
      } else {
        ss << "_qubits";
      }
      return ss.str();
    });

TEST_P(DynamicCircuitEvalBV, UnitaryTransformation) {
  bv->reorderOperations();
  const auto start = std::chrono::steady_clock::now();
  // transform dynamic circuit to unitary circuit by first eliminating reset
  // operations and afterwards deferring measurements to the end of the circuit
  qc::CircuitOptimizer::eliminateResets(*dbv);
  qc::CircuitOptimizer::deferMeasurements(*dbv);

  // remove final measurements in order to just obtain the unitary functionality
  qc::CircuitOptimizer::removeFinalMeasurements(*dbv);
  dbv->reorderOperations();
  const auto finishedTransformation = std::chrono::steady_clock::now();

  qc::MatrixDD e = dd->makeIdent();
  dd->incRef(e);

  auto leftIt = bv->begin();
  auto rightIt = dbv->begin();

  while (leftIt != bv->end() && rightIt != dbv->end()) {
    auto multLeft = dd->multiply(getDD((*leftIt).get(), *dd), e);
    auto multRight =
        dd->multiply(multLeft, getInverseDD((*rightIt).get(), *dd));
    dd->incRef(multRight);
    dd->decRef(e);
    e = multRight;

    dd->garbageCollect();

    ++leftIt;
    ++rightIt;
  }

  while (leftIt != bv->end()) {
    auto multLeft = dd->multiply(getDD((*leftIt).get(), *dd), e);
    dd->incRef(multLeft);
    dd->decRef(e);
    e = multLeft;

    dd->garbageCollect();

    ++leftIt;
  }

  while (rightIt != dbv->end()) {
    auto multRight = dd->multiply(e, getInverseDD((*rightIt).get(), *dd));
    dd->incRef(multRight);
    dd->decRef(e);
    e = multRight;

    dd->garbageCollect();

    ++rightIt;
  }
  const auto finishedEC = std::chrono::steady_clock::now();

  const auto preprocessing =
      std::chrono::duration<double>(finishedTransformation - start).count();
  const auto verification =
      std::chrono::duration<double>(finishedEC - finishedTransformation)
          .count();

  std::stringstream ss{};
  ss << "bv,transformation," << bv->getNqubits() << "," << bvNgates << ",2,"
     << dbvNgates << "," << preprocessing << "," << verification;
  std::cout << ss.str() << "\n";

  EXPECT_TRUE(e.isIdentity());
}

TEST_P(DynamicCircuitEvalBV, ProbabilityExtraction) {
  // generate DD of QPE circuit via simulation
  const auto start = std::chrono::steady_clock::now();
  auto e = simulate(bv.get(), dd->makeZeroState(bv->getNqubits()), *dd);
  const auto simulationEnd = std::chrono::steady_clock::now();

  // extract measurement probabilities from IQPE simulations
  dd::SparsePVec probs{};
  extractProbabilityVector(dbv.get(), dd->makeZeroState(dbv->getNqubits()),
                           probs, *dd);
  const auto extractionEnd = std::chrono::steady_clock::now();

  // compare outcomes
  auto fidelity =
      dd->fidelityOfMeasurementOutcomes(e, probs, bv->outputPermutation);
  const auto comparisonEnd = std::chrono::steady_clock::now();

  const auto simulation =
      std::chrono::duration<double>(simulationEnd - start).count();
  const auto extraction =
      std::chrono::duration<double>(extractionEnd - simulationEnd).count();
  const auto comparison =
      std::chrono::duration<double>(comparisonEnd - extractionEnd).count();
  const auto total =
      std::chrono::duration<double>(comparisonEnd - start).count();

  std::stringstream ss{};
  ss << "bv,extraction," << bv->getNqubits() << "," << bvNgates << ",2,"
     << dbvNgates << "," << simulation << "," << extraction << "," << comparison
     << "," << total;
  std::cout << ss.str() << "\n";

  EXPECT_NEAR(fidelity, 1.0, 1e-4);
}

class DynamicCircuitEvalQFT : public testing::TestWithParam<std::size_t> {
protected:
  std::size_t precision{};
  std::unique_ptr<qc::QuantumComputation> qft;
  std::unique_ptr<qc::QuantumComputation> dqft;
  std::size_t qftNgates{};
  std::size_t dqftNgates{};
  std::unique_ptr<dd::Package<>> dd;
  std::ofstream ofs;

  void TearDown() override {}
  void SetUp() override {
    precision = GetParam();

    dd = std::make_unique<dd::Package<>>(precision);

    qft = std::make_unique<qc::QFT>(precision);
    // remove final measurements so that the functionality is unitary
    qc::CircuitOptimizer::removeFinalMeasurements(*qft);
    qftNgates = qft->getNindividualOps();

    dqft = std::make_unique<qc::QFT>(precision, true, true);
    dqftNgates = dqft->getNindividualOps();
  }
};

INSTANTIATE_TEST_SUITE_P(
    Eval, DynamicCircuitEvalQFT, testing::Range<std::size_t>(1U, 65U, 5U),
    [](const testing::TestParamInfo<DynamicCircuitEvalExactQPE::ParamType>&
           inf) {
      const auto nqubits = inf.param;
      std::stringstream ss{};
      ss << nqubits;
      if (nqubits == 1) {
        ss << "_qubit";
      } else {
        ss << "_qubits";
      }
      return ss.str();
    });

TEST_P(DynamicCircuitEvalQFT, UnitaryTransformation) {
  qft->reorderOperations();
  const auto start = std::chrono::steady_clock::now();
  // transform dynamic circuit to unitary circuit by first eliminating reset
  // operations and afterwards deferring measurements to the end of the circuit
  qc::CircuitOptimizer::eliminateResets(*dqft);
  qc::CircuitOptimizer::deferMeasurements(*dqft);

  // remove final measurements in order to just obtain the unitary functionality
  qc::CircuitOptimizer::removeFinalMeasurements(*dqft);
  dqft->reorderOperations();
  const auto finishedTransformation = std::chrono::steady_clock::now();

  qc::MatrixDD e = dd->makeIdent();
  dd->incRef(e);

  auto leftIt = qft->begin();
  auto rightIt = dqft->begin();

  while (leftIt != qft->end() && rightIt != dqft->end()) {
    auto multLeft = dd->multiply(getDD((*leftIt).get(), *dd), e);
    auto multRight =
        dd->multiply(multLeft, getInverseDD((*rightIt).get(), *dd));
    dd->incRef(multRight);
    dd->decRef(e);
    e = multRight;

    dd->garbageCollect();

    ++leftIt;
    ++rightIt;
  }

  while (leftIt != qft->end()) {
    auto multLeft = dd->multiply(getDD((*leftIt).get(), *dd), e);
    dd->incRef(multLeft);
    dd->decRef(e);
    e = multLeft;

    dd->garbageCollect();

    ++leftIt;
  }

  while (rightIt != dqft->end()) {
    auto multRight = dd->multiply(e, getInverseDD((*rightIt).get(), *dd));
    dd->incRef(multRight);
    dd->decRef(e);
    e = multRight;

    dd->garbageCollect();

    ++rightIt;
  }
  const auto finishedEC = std::chrono::steady_clock::now();

  const auto preprocessing =
      std::chrono::duration<double>(finishedTransformation - start).count();
  const auto verification =
      std::chrono::duration<double>(finishedEC - finishedTransformation)
          .count();

  std::stringstream ss{};
  ss << "qft,transformation," << qft->getNqubits() << "," << qftNgates << ",1,"
     << dqftNgates << "," << preprocessing << "," << verification;
  std::cout << ss.str() << "\n";

  EXPECT_TRUE(e.isIdentity());
}

TEST_P(DynamicCircuitEvalQFT, ProbabilityExtraction) {
  // generate DD of QPE circuit via simulation
  const auto start = std::chrono::steady_clock::now();
  auto e = simulate(qft.get(), dd->makeZeroState(qft->getNqubits()), *dd);
  const auto simulationEnd = std::chrono::steady_clock::now();
  const auto simulation =
      std::chrono::duration<double>(simulationEnd - start).count();

  std::stringstream ss{};
  // extract measurement probabilities from IQPE simulations
  if (qft->getNqubits() <= 15) {
    dd::SparsePVec probs{};
    extractProbabilityVector(dqft.get(), dd->makeZeroState(dqft->getNqubits()),
                             probs, *dd);
    const auto extractionEnd = std::chrono::steady_clock::now();

    // compare outcomes
    auto fidelity =
        dd->fidelityOfMeasurementOutcomes(e, probs, qft->outputPermutation);
    const auto comparisonEnd = std::chrono::steady_clock::now();
    const auto extraction =
        std::chrono::duration<double>(extractionEnd - simulationEnd).count();
    const auto comparison =
        std::chrono::duration<double>(comparisonEnd - extractionEnd).count();
    const auto total =
        std::chrono::duration<double>(comparisonEnd - start).count();
    EXPECT_NEAR(fidelity, 1.0, 1e-4);
    ss << "qft,extraction," << qft->getNqubits() << "," << qftNgates << ",1,"
       << dqftNgates << "," << extraction << "," << simulation << ","
       << comparison << "," << total;
    std::cout << ss.str() << "\n";

  } else {
    ss << "qft,extraction," << qft->getNqubits() << "," << qftNgates << ",1,"
       << dqftNgates << ",," << simulation << ",,,";
    std::cout << ss.str() << "\n";
  }
}
