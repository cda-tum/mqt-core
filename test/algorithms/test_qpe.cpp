/*
 * This file is part of JKQ QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#include "CircuitOptimizer.hpp"
#include "algorithms/IQPE.hpp"
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

        std::cout << "Estimating lambda = " << lambda << " up to " << static_cast<std::size_t>(precision) << "-bit precision." << std::endl;

        theta = lambda / (2 * dd::PI);

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
                                 std::pair{dd::PI, static_cast<dd::QubitCount>(1)},
                                 std::pair{dd::PI_2, static_cast<dd::QubitCount>(2)},
                                 std::pair{dd::PI_4, static_cast<dd::QubitCount>(3)},
                                 std::pair{3 * dd::PI / 8, static_cast<dd::QubitCount>(3)},
                                 std::pair{3 * dd::PI / 8, static_cast<dd::QubitCount>(4)}),
                         [](const testing::TestParamInfo<QPE::ParamType>& info) {
                             // Generate names for test cases
                             dd::fp            lambda    = info.param.first;
                             dd::QubitCount    precision = info.param.second;
                             std::stringstream ss{};
                             ss << static_cast<std::size_t>(lambda * 100) << "_" << static_cast<std::size_t>(precision);
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
    auto                      dd = std::make_unique<dd::Package>(precision + 1);
    std::unique_ptr<qc::IQPE> qc;
    qc::VectorDD              e{};

    ASSERT_NO_THROW({ qc = std::make_unique<qc::IQPE>(lambda, precision); });

    ASSERT_EQ(static_cast<std::size_t>(qc->getNqubits()), 2U);

    /// TODO: at the moment no further checks are here due to the QFR simulator not supporting measurements
}
