/*
 * This file is part of MQT QFR library which is released under the MIT license.
 * See file README.md or go to https://www.cda.cit.tum.de/research/quantum/ for
 * more information.
 */

#include "CircuitOptimizer.hpp"
#include "algorithms/QPE.hpp"
#include "dd/FunctionalityConstruction.hpp"
#include "dd/Simulation.hpp"

#include "gtest/gtest.h"
#include <bitset>
#include <iomanip>
#include <string>
#include <utility>

class QPE : public testing::TestWithParam<std::pair<qc::fp, std::size_t>> {
protected:
  qc::fp lambda{};
  std::uint64_t precision{};
  qc::fp theta{};
  bool exactlyRepresentable{};
  std::uint64_t expectedResult{};
  std::string expectedResultRepresentation{};
  std::uint64_t secondExpectedResult{};
  std::string secondExpectedResultRepresentation{};

  void TearDown() override {}
  void SetUp() override {
    lambda = GetParam().first;
    precision = GetParam().second;

    std::cout << "Estimating lambda = " << lambda << "π up to " << precision
              << "-bit precision." << std::endl;

    theta = lambda / 2;

    std::cout << "Expected theta=" << theta << std::endl;
    std::bitset<64> binaryExpansion{};
    auto expansion = theta * 2;
    std::size_t index = 0;
    while (std::abs(expansion) > 1e-8) {
      if (expansion >= 1.) {
        binaryExpansion.set(index);
        expansion -= 1.0;
      }
      index++;
      expansion *= 2;
    }

    exactlyRepresentable = true;
    for (std::uint64_t i = precision; i < binaryExpansion.size(); ++i) {
      if (binaryExpansion.test(i)) {
        exactlyRepresentable = false;
        break;
      }
    }

    expectedResult = 0U;
    for (std::uint64_t i = 0; i < precision; ++i) {
      if (binaryExpansion.test(i)) {
        expectedResult |= (1ULL << (precision - 1 - i));
      }
    }
    std::stringstream ss{};
    for (auto i = static_cast<std::int64_t>(precision - 1); i >= 0; --i) {
      if ((expectedResult & (1ULL << i)) != 0) {
        ss << 1;
      } else {
        ss << 0;
      }
    }
    expectedResultRepresentation = ss.str();

    if (exactlyRepresentable) {
      std::cout << "Theta is exactly representable using " << precision
                << " bits." << std::endl;
      std::cout << "The expected output state is |"
                << expectedResultRepresentation << ">." << std::endl;
    } else {
      secondExpectedResult = expectedResult + 1;
      ss.str("");
      for (auto i = static_cast<std::int64_t>(precision - 1); i >= 0; --i) {
        if ((secondExpectedResult & (1ULL << i)) != 0) {
          ss << 1;
        } else {
          ss << 0;
        }
      }
      secondExpectedResultRepresentation = ss.str();

      std::cout << "Theta is not exactly representable using " << precision
                << " bits." << std::endl;
      std::cout << "Most probable output states are |"
                << expectedResultRepresentation << "> and |"
                << secondExpectedResultRepresentation << ">." << std::endl;
    }
  }
};

INSTANTIATE_TEST_SUITE_P(
    QPE, QPE,
    testing::Values(std::pair{1., 1U}, std::pair{0.5, 2U}, std::pair{0.25, 3U},
                    std::pair{3. / 8, 3U}, std::pair{3. / 8, 4U},
                    std::pair{3. / 32, 5U}, std::pair{3. / 32, 6U}),
    [](const testing::TestParamInfo<QPE::ParamType>& inf) {
      // Generate names for test cases
      const auto lambda = inf.param.first;
      const auto precision = inf.param.second;
      std::stringstream ss{};
      ss << static_cast<std::size_t>(lambda * 100) << "_pi_" << precision;
      return ss.str();
    });

TEST_P(QPE, QPETest) {
  auto dd = std::make_unique<dd::Package<>>(precision + 1);
  std::unique_ptr<qc::QPE> qc;
  qc::VectorDD e{};

  ASSERT_NO_THROW({ qc = std::make_unique<qc::QPE>(lambda, precision); });

  ASSERT_EQ(qc->getNqubits(), precision + 1);

  ASSERT_NO_THROW({ qc::CircuitOptimizer::removeFinalMeasurements(*qc); });

  ASSERT_NO_THROW({
    e = simulate(
        qc.get(),
        dd->makeZeroState(static_cast<dd::QubitCount>(qc->getNqubits())), dd);
  });

  // account for the eigenstate qubit in the expected result by shifting and
  // adding 1
  auto amplitude = dd->getValueByPath(e, (expectedResult << 1) + 1);
  auto probability = amplitude.r * amplitude.r + amplitude.i * amplitude.i;
  std::cout << "Obtained probability for |" << expectedResultRepresentation
            << ">: " << probability << std::endl;

  if (exactlyRepresentable) {
    EXPECT_NEAR(probability, 1.0, 1e-8);
  } else {
    const auto threshold = 4. / (qc::PI * qc::PI);
    // account for the eigenstate qubit in the expected result by shifting and
    // adding 1
    auto secondAmplitude =
        dd->getValueByPath(e, (secondExpectedResult << 1) + 1);
    auto secondProbability = secondAmplitude.r * secondAmplitude.r +
                             secondAmplitude.i * secondAmplitude.i;
    std::cout << "Obtained probability for |"
              << secondExpectedResultRepresentation
              << ">: " << secondProbability << std::endl;

    EXPECT_GT(probability, threshold);
    EXPECT_GT(secondProbability, threshold);
  }
}

TEST_P(QPE, IQPETest) {
  auto dd = std::make_unique<dd::Package<>>(precision + 1);
  std::unique_ptr<qc::QPE> qc;

  ASSERT_NO_THROW({ qc = std::make_unique<qc::QPE>(lambda, precision, true); });

  ASSERT_EQ(qc->getNqubits(), 2U);

  constexpr auto shots = 8192U;
  auto measurements =
      simulate(qc.get(),
               dd->makeZeroState(static_cast<dd::QubitCount>(qc->getNqubits())),
               dd, shots);

  // sort the measurements
  using Measurement = std::pair<std::string, std::size_t>;
  auto comp = [](const Measurement& a, const Measurement& b) -> bool {
    if (a.second != b.second) {
      return a.second > b.second;
    }
    return a.first > b.first;
  };
  const std::set<Measurement, decltype(comp)> ordered(measurements.begin(),
                                                      measurements.end(), comp);

  std::cout << "Obtained measurements: " << std::endl;
  for (const auto& measurement : ordered) {
    std::cout << "\t" << measurement.first << ": " << measurement.second << " ("
              << (measurement.second * 100) / shots << "%)" << std::endl;
  }

  const auto& mostLikely = *ordered.begin();
  if (exactlyRepresentable) {
    EXPECT_EQ(mostLikely.first, expectedResultRepresentation);
    EXPECT_EQ(mostLikely.second, shots);
  } else {
    auto it = ordered.begin();
    std::advance(it, 1);
    const auto& secondMostLikely = *(it);
    EXPECT_TRUE(
        (mostLikely.first == expectedResultRepresentation &&
         secondMostLikely.first == secondExpectedResultRepresentation) ||
        (mostLikely.first == secondExpectedResultRepresentation &&
         secondMostLikely.first == expectedResultRepresentation));
    const auto threshold = 4. / (qc::PI * qc::PI);
    EXPECT_NEAR(static_cast<double>(mostLikely.second) / shots, threshold,
                0.02);
    EXPECT_NEAR(static_cast<double>(secondMostLikely.second) / shots, threshold,
                0.02);
  }
}

TEST_P(QPE, DynamicEquivalenceSimulation) {
  auto dd = std::make_unique<dd::Package<>>(precision + 1);

  // create standard QPE circuit
  auto qpe = std::make_unique<qc::QPE>(lambda, precision);

  // remove final measurements to obtain statevector
  qc::CircuitOptimizer::removeFinalMeasurements(*qpe);

  // simulate circuit
  auto e = simulate(
      qpe.get(),
      dd->makeZeroState(static_cast<dd::QubitCount>(qpe->getNqubits())), dd);

  // create standard IQPE circuit
  auto iqpe = std::make_unique<qc::QPE>(lambda, precision, true);

  // transform dynamic circuits by first eliminating reset operations and
  // afterwards deferring measurements
  qc::CircuitOptimizer::eliminateResets(*iqpe);

  qc::CircuitOptimizer::deferMeasurements(*iqpe);

  // remove final measurements to obtain statevector
  qc::CircuitOptimizer::removeFinalMeasurements(*iqpe);

  // simulate circuit
  auto f = simulate(
      iqpe.get(),
      dd->makeZeroState(static_cast<dd::QubitCount>(iqpe->getNqubits())), dd);

  // calculate fidelity between both results
  auto fidelity = dd->fidelity(e, f);
  std::cout << "Fidelity of both circuits: " << fidelity << std::endl;

  EXPECT_NEAR(fidelity, 1.0, 1e-4);
}

TEST_P(QPE, DynamicEquivalenceFunctionality) {
  auto dd = std::make_unique<dd::Package<>>(precision + 1);

  // create standard QPE circuit
  auto qpe = std::make_unique<qc::QPE>(lambda, precision);

  // remove final measurements to obtain statevector
  qc::CircuitOptimizer::removeFinalMeasurements(*qpe);

  // simulate circuit
  auto e = buildFunctionality(qpe.get(), dd);

  // create standard IQPE circuit
  auto iqpe = std::make_unique<qc::QPE>(lambda, precision, true);

  // transform dynamic circuits by first eliminating reset operations and
  // afterwards deferring measurements
  qc::CircuitOptimizer::eliminateResets(*iqpe);
  qc::CircuitOptimizer::deferMeasurements(*iqpe);

  // remove final measurements to obtain statevector
  qc::CircuitOptimizer::removeFinalMeasurements(*iqpe);

  // simulate circuit
  auto f = buildFunctionality(iqpe.get(), dd);

  EXPECT_EQ(e, f);
}

TEST_P(QPE, ProbabilityExtraction) {
  auto dd = std::make_unique<dd::Package<>>(precision + 1);

  // create standard QPE circuit
  auto iqpe = std::make_unique<qc::QPE>(lambda, precision, true);

  std::cout << *iqpe << std::endl;
  dd::ProbabilityVector probs{};
  extractProbabilityVector(
      iqpe.get(),
      dd->makeZeroState(static_cast<dd::QubitCount>(iqpe->getNqubits())), probs,
      dd);

  for (const auto& [state, prob] : probs) {
    std::stringstream ss{};
    qc::QuantumComputation::printBin(state, ss);
    std::cout << ss.str() << ": " << prob << std::endl;
  }

  if (exactlyRepresentable) {
    EXPECT_NEAR(probs.at(expectedResult), 1.0, 1e-6);
  } else {
    const auto threshold = 4. / (qc::PI * qc::PI);
    EXPECT_NEAR(probs.at(expectedResult), threshold, 0.02);
    EXPECT_NEAR(probs.at(secondExpectedResult), threshold, 0.02);
  }
}

TEST_P(QPE, DynamicEquivalenceSimulationProbabilityExtraction) {
  auto dd = std::make_unique<dd::Package<>>(precision + 1);

  // create standard QPE circuit
  auto qpe = std::make_unique<qc::QPE>(lambda, precision);

  // remove final measurements to obtain statevector
  qc::CircuitOptimizer::removeFinalMeasurements(*qpe);

  // simulate circuit
  auto e = simulate(
      qpe.get(),
      dd->makeZeroState(static_cast<dd::QubitCount>(qpe->getNqubits())), dd);
  const auto vec = dd->getVector(e);
  std::cout << "QPE: " << std::endl;
  for (const auto& amp : vec) {
    std::cout << std::norm(amp) << std::endl;
  }

  // create standard IQPE circuit
  auto iqpe = std::make_unique<qc::QPE>(lambda, precision, true);

  // extract measurement probabilities from IQPE simulations
  dd::ProbabilityVector probs{};
  extractProbabilityVector(
      iqpe.get(),
      dd->makeZeroState(static_cast<dd::QubitCount>(iqpe->getNqubits())), probs,
      dd);

  // extend to account for 0 qubit
  auto stub = dd::ProbabilityVector{};
  stub.reserve(probs.size());
  for (const auto& [state, prob] : probs) {
    stub[2 * state + 1] = prob;
  }

  std::cout << "IQPE: " << std::endl;
  for (const auto& [state, prob] : stub) {
    std::stringstream ss{};
    qc::QuantumComputation::printBin(state, ss);
    std::cout << ss.str() << ": " << prob << std::endl;
  }

  // calculate fidelity between both results
  auto fidelity = dd->fidelityOfMeasurementOutcomes(e, stub);
  std::cout << "Fidelity of both circuits' measurement outcomes: " << fidelity
            << std::endl;

  EXPECT_NEAR(fidelity, 1.0, 1e-4);
}
