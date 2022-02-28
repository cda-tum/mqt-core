/*
 * This file is part of MQT QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#include "CircuitOptimizer.hpp"
#include "algorithms/QPE.hpp"

#include "gtest/gtest.h"
#include <bitset>
#include <iomanip>
#include <string>
#include <utility>

class QPE: public testing::TestWithParam<std::pair<dd::fp, dd::QubitCount>> {
protected:
    dd::fp         lambda{};
    dd::QubitCount precision{};
    dd::fp         theta{};
    bool           exactlyRepresentable{};
    std::size_t    expectedResult{};
    std::string    expectedResultRepresentation{};
    std::size_t    secondExpectedResult{};
    std::string    secondExpectedResultRepresentation{};

    void TearDown() override {}
    void SetUp() override {
        lambda    = GetParam().first;
        precision = GetParam().second;

        std::cout << "Estimating lambda = " << lambda << "π up to " << static_cast<std::size_t>(precision) << "-bit precision." << std::endl;

        theta = lambda / 2;

        std::cout << "Expected theta=" << theta << std::endl;
        std::bitset<64> binaryExpansion{};
        dd::fp          expansion = theta * 2;
        std::size_t     index     = 0;
        while (std::abs(expansion) > 1e-8) {
            if (expansion >= 1.) {
                binaryExpansion.set(index);
                expansion -= 1.0;
            }
            index++;
            expansion *= 2;
        }

        exactlyRepresentable = true;
        for (std::size_t i = precision; i < binaryExpansion.size(); ++i) {
            if (binaryExpansion.test(i)) {
                exactlyRepresentable = false;
                break;
            }
        }

        expectedResult = 0U;
        for (std::size_t i = 0; i < precision; ++i) {
            if (binaryExpansion.test(i)) {
                expectedResult |= (1U << (precision - 1 - i));
            }
        }
        std::stringstream ss{};
        for (auto i = static_cast<int>(precision - 1); i >= 0; --i) {
            if (expectedResult & (1U << i)) {
                ss << 1;
            } else {
                ss << 0;
            }
        }
        expectedResultRepresentation = ss.str();

        if (exactlyRepresentable) {
            std::cout << "Theta is exactly representable using " << static_cast<std::size_t>(precision) << " bits." << std::endl;
            std::cout << "The expected output state is |" << expectedResultRepresentation << ">." << std::endl;
        } else {
            secondExpectedResult = expectedResult + 1;
            ss.str("");
            for (auto i = static_cast<int>(precision - 1); i >= 0; --i) {
                if (secondExpectedResult & (1U << i)) {
                    ss << 1;
                } else {
                    ss << 0;
                }
            }
            secondExpectedResultRepresentation = ss.str();

            std::cout << "Theta is not exactly representable using " << static_cast<std::size_t>(precision) << " bits." << std::endl;
            std::cout << "Most probable output states are |" << expectedResultRepresentation << "> and |" << secondExpectedResultRepresentation << ">." << std::endl;
        }
    }
};

INSTANTIATE_TEST_SUITE_P(QPE, QPE,
                         testing::Values(
                                 std::pair{1., static_cast<dd::QubitCount>(1)},
                                 std::pair{0.5, static_cast<dd::QubitCount>(2)},
                                 std::pair{0.25, static_cast<dd::QubitCount>(3)},
                                 std::pair{3. / 8, static_cast<dd::QubitCount>(3)},
                                 std::pair{3. / 8, static_cast<dd::QubitCount>(4)},
                                 std::pair{3. / 32, static_cast<dd::QubitCount>(5)},
                                 std::pair{3. / 32, static_cast<dd::QubitCount>(6)}),
                         [](const testing::TestParamInfo<QPE::ParamType>& info) {
                             // Generate names for test cases
                             dd::fp            lambda    = info.param.first;
                             dd::QubitCount    precision = info.param.second;
                             std::stringstream ss{};
                             ss << static_cast<std::size_t>(lambda * 100) << "_pi_" << static_cast<std::size_t>(precision);
                             return ss.str();
                         });

TEST_P(QPE, QPETest) {
    auto                     dd = std::make_unique<dd::Package>(precision + 1);
    std::unique_ptr<qc::QPE> qc;
    qc::VectorDD             e{};

    ASSERT_NO_THROW({ qc = std::make_unique<qc::QPE>(lambda, precision); });

    //std::cout << *qc<<std::endl;

    ASSERT_EQ(static_cast<std::size_t>(qc->getNqubits()), precision + 1);

    ASSERT_NO_THROW({ qc::CircuitOptimizer::removeFinalMeasurements(*qc); });

    ASSERT_NO_THROW({ e = qc->simulate(dd->makeZeroState(qc->getNqubits()), dd); });

    // account for the eigenstate qubit in the expected result by shifting and adding 1
    auto amplitude   = dd->getValueByPath(e, (expectedResult << 1) + 1);
    auto probability = amplitude.r * amplitude.r + amplitude.i * amplitude.i;
    std::cout << "Obtained probability for |" << expectedResultRepresentation << ">: " << probability << std::endl;

    if (exactlyRepresentable) {
        EXPECT_NEAR(probability, 1.0, 1e-8);
    } else {
        auto threshold = 4. / (dd::PI * dd::PI);
        // account for the eigenstate qubit in the expected result by shifting and adding 1
        auto secondAmplitude   = dd->getValueByPath(e, (secondExpectedResult << 1) + 1);
        auto secondProbability = secondAmplitude.r * secondAmplitude.r + secondAmplitude.i * secondAmplitude.i;
        std::cout << "Obtained probability for |" << secondExpectedResultRepresentation << ">: " << secondProbability << std::endl;

        EXPECT_GT(probability, threshold);
        EXPECT_GT(secondProbability, threshold);
    }
}

TEST_P(QPE, IQPETest) {
    auto                     dd = std::make_unique<dd::Package>(precision + 1);
    std::unique_ptr<qc::QPE> qc;

    ASSERT_NO_THROW({ qc = std::make_unique<qc::QPE>(lambda, precision, true); });

    ASSERT_EQ(static_cast<std::size_t>(qc->getNqubits()), 2U);

    //    std::cout << *qc << std::endl;

    constexpr auto shots        = 8192U;
    auto           measurements = qc->simulate(dd->makeZeroState(qc->getNqubits()), dd, shots);

    // sort the measurements
    using Measurement = std::pair<std::string, std::size_t>;
    auto comp         = [](const Measurement& a, const Measurement& b) -> bool {
        if (a.second != b.second) {
            return a.second > b.second;
        }
        return a.first > b.first;
    };
    std::set<Measurement, decltype(comp)> ordered(measurements.begin(), measurements.end(), comp);

    std::cout << "Obtained measurements: " << std::endl;
    for (const auto& measurement: ordered) {
        std::cout << "\t" << measurement.first << ": " << measurement.second << " (" << (measurement.second * 100) / shots << "%)" << std::endl;
    }

    const auto& mostLikely = *ordered.begin();
    if (exactlyRepresentable) {
        EXPECT_EQ(mostLikely.first, expectedResultRepresentation);
        EXPECT_EQ(mostLikely.second, shots);
    } else {
        auto it = ordered.begin();
        std::advance(it, 1);
        const auto& secondMostLikely = *(it);
        EXPECT_TRUE((mostLikely.first == expectedResultRepresentation && secondMostLikely.first == secondExpectedResultRepresentation) ||
                    (mostLikely.first == secondExpectedResultRepresentation && secondMostLikely.first == expectedResultRepresentation));
        auto threshold = 4. / (dd::PI * dd::PI);
        EXPECT_NEAR(static_cast<double>(mostLikely.second) / shots, threshold, 0.02);
        EXPECT_NEAR(static_cast<double>(secondMostLikely.second) / shots, threshold, 0.02);
    }
}

TEST_P(QPE, DynamicEquivalenceSimulation) {
    auto dd = std::make_unique<dd::Package>(precision + 1);

    // create standard QPE circuit
    auto qpe = std::make_unique<qc::QPE>(lambda, precision);

    // remove final measurements to obtain statevector
    qc::CircuitOptimizer::removeFinalMeasurements(*qpe);

    // simulate circuit
    auto e = qpe->simulate(dd->makeZeroState(qpe->getNqubits()), dd);

    // create standard IQPE circuit
    auto iqpe = std::make_unique<qc::QPE>(lambda, precision, true);

    // transform dynamic circuits by first eliminating reset operations and afterwards deferring measurements
    qc::CircuitOptimizer::eliminateResets(*iqpe);

    qc::CircuitOptimizer::deferMeasurements(*iqpe);

    // remove final measurements to obtain statevector
    qc::CircuitOptimizer::removeFinalMeasurements(*iqpe);

    // simulate circuit
    auto f = iqpe->simulate(dd->makeZeroState(iqpe->getNqubits()), dd);

    // calculate fidelity between both results
    auto fidelity = dd->fidelity(e, f);
    std::cout << "Fidelity of both circuits: " << fidelity << std::endl;

    EXPECT_NEAR(fidelity, 1.0, 1e-4);
}

TEST_P(QPE, DynamicEquivalenceFunctionality) {
    auto dd = std::make_unique<dd::Package>(precision + 1);

    // create standard QPE circuit
    auto qpe = std::make_unique<qc::QPE>(lambda, precision);

    // remove final measurements to obtain statevector
    qc::CircuitOptimizer::removeFinalMeasurements(*qpe);

    // simulate circuit
    auto e = qpe->buildFunctionality(dd);

    // create standard IQPE circuit
    auto iqpe = std::make_unique<qc::QPE>(lambda, precision, true);

    // transform dynamic circuits by first eliminating reset operations and afterwards deferring measurements
    qc::CircuitOptimizer::eliminateResets(*iqpe);
    qc::CircuitOptimizer::deferMeasurements(*iqpe);

    // remove final measurements to obtain statevector
    qc::CircuitOptimizer::removeFinalMeasurements(*iqpe);

    // simulate circuit
    auto f = iqpe->buildFunctionality(dd);

    EXPECT_EQ(e, f);
}

TEST_P(QPE, ProbabilityExtraction) {
    auto dd = std::make_unique<dd::Package>(precision + 1);

    // create standard QPE circuit
    auto iqpe = std::make_unique<qc::QPE>(lambda, precision, true);

    std::cout << *iqpe << std::endl;
    dd::ProbabilityVector probs{};
    iqpe->extractProbabilityVector(dd->makeZeroState(iqpe->getNqubits()), probs, dd);

    for (const auto& [state, prob]: probs) {
        std::stringstream ss{};
        qc::QuantumComputation::printBin(state, ss);
        std::cout << ss.str() << ": " << prob << std::endl;
    }

    if (exactlyRepresentable) {
        EXPECT_NEAR(probs.at(expectedResult), 1.0, 1e-6);
    } else {
        auto threshold = 4. / (dd::PI * dd::PI);
        EXPECT_NEAR(probs.at(expectedResult), threshold, 0.02);
        EXPECT_NEAR(probs.at(secondExpectedResult), threshold, 0.02);
    }
}

TEST_P(QPE, DynamicEquivalenceSimulationProbabilityExtraction) {
    auto dd = std::make_unique<dd::Package>(precision + 1);

    // create standard QPE circuit
    auto qpe = std::make_unique<qc::QPE>(lambda, precision);

    // remove final measurements to obtain statevector
    qc::CircuitOptimizer::removeFinalMeasurements(*qpe);

    // simulate circuit
    auto       e   = qpe->simulate(dd->makeZeroState(qpe->getNqubits()), dd);
    const auto vec = dd->getVector(e);
    std::cout << "QPE: " << std::endl;
    for (const auto& amp: vec) {
        std::cout << std::norm(amp) << std::endl;
    }

    // create standard IQPE circuit
    auto iqpe = std::make_unique<qc::QPE>(lambda, precision, true);

    // extract measurement probabilities from IQPE simulations
    dd::ProbabilityVector probs{};
    iqpe->extractProbabilityVector(dd->makeZeroState(iqpe->getNqubits()), probs, dd);

    // extend to account for 0 qubit
    auto stub = dd::ProbabilityVector{};
    stub.reserve(probs.size());
    for (const auto& [state, prob]: probs) {
        stub[2 * state + 1] = prob;
    }

    std::cout << "IQPE: " << std::endl;
    for (const auto& [state, prob]: stub) {
        std::stringstream ss{};
        qc::QuantumComputation::printBin(state, ss);
        std::cout << ss.str() << ": " << prob << std::endl;
    }

    // calculate fidelity between both results
    auto fidelity = dd->fidelityOfMeasurementOutcomes(e, stub);
    std::cout << "Fidelity of both circuits' measurement outcomes: " << fidelity << std::endl;

    EXPECT_NEAR(fidelity, 1.0, 1e-4);
}
