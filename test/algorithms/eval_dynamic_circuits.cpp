/*
* This file is part of JKQ QFR library which is released under the MIT license.
* See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
*/

#include "CircuitOptimizer.hpp"
#include "algorithms/IQPE.hpp"
#include "algorithms/QPE.hpp"

#include "gtest/gtest.h"
#include <bitset>
#include <chrono>
#include <iomanip>
#include <string>
#include <utility>

class DynamicCircuitEvalExactQPE: public testing::TestWithParam<std::size_t> {
protected:
    dd::QubitCount                          precision{};
    dd::fp                                  theta{};
    std::size_t                             expectedResult{};
    std::string                             expectedResultRepresentation{};
    std::unique_ptr<qc::QuantumComputation> qpe;
    std::unique_ptr<qc::QuantumComputation> iqpe;
    std::size_t                             qpeNgates{};
    std::size_t                             iqpeNgates{};
    std::unique_ptr<dd::Package>            dd;
    std::ofstream                           ofs;

    void TearDown() override {}
    void SetUp() override {
        precision = GetParam();

        dd = std::make_unique<dd::Package>(precision + 1);

        qpe = std::make_unique<qc::QPE>(precision);
        // remove final measurements so that the functionality is unitary
        qc::CircuitOptimizer::removeFinalMeasurements(*qpe);
        qpeNgates = qpe->getNindividualOps();

        const auto lambda = dynamic_cast<qc::QPE*>(qpe.get())->lambda;
        iqpe              = std::make_unique<qc::IQPE>(lambda, precision);
        iqpeNgates        = iqpe->getNindividualOps();

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

        std::cout << "Theta is exactly representable using " << static_cast<std::size_t>(precision) << " bits." << std::endl;
        std::cout << "The expected output state is |" << expectedResultRepresentation << ">." << std::endl;
    }
};

INSTANTIATE_TEST_SUITE_P(Eval, DynamicCircuitEvalExactQPE,
                         testing::Range<std::size_t>(1U, 10U),
                         [](const testing::TestParamInfo<DynamicCircuitEvalExactQPE::ParamType>& info) {
                             dd::QubitCount nqubits = info.param;
                             std::stringstream ss{};
                             ss << static_cast<std::size_t>(nqubits);
                             if (nqubits == 1) {
                                 ss << "_qubit";
                             } else {
                                 ss << "_qubits";
                             }
                             return ss.str(); });

TEST_P(DynamicCircuitEvalExactQPE, UnitaryTransformation) {
    const auto start = std::chrono::steady_clock::now();
    // transform dynamic circuit to unitary circuit by first eliminating reset operations and afterwards deferring measurements to the end of the circuit
    qc::CircuitOptimizer::eliminateResets(*iqpe);
    qc::CircuitOptimizer::deferMeasurements(*iqpe);

    // remove final measurements in order to just obtain the unitary functionality
    qc::CircuitOptimizer::removeFinalMeasurements(*iqpe);
    const auto finishedTransformation = std::chrono::steady_clock::now();

    qc::MatrixDD e = dd->makeIdent(precision + 1);
    dd->incRef(e);

    auto leftIt  = qpe->begin();
    auto rightIt = iqpe->begin();

    while (leftIt != qpe->end() && rightIt != iqpe->end()) {
        auto multLeft  = dd->multiply((*leftIt)->getDD(dd), e);
        auto multRight = dd->multiply(multLeft, (*rightIt)->getInverseDD(dd));
        dd->incRef(multRight);
        dd->decRef(e);
        e = multRight;

        dd->garbageCollect();

        ++leftIt;
        ++rightIt;
    }

    while (leftIt != qpe->end()) {
        auto multLeft = dd->multiply((*leftIt)->getDD(dd), e);
        dd->incRef(multLeft);
        dd->decRef(e);
        e = multLeft;

        dd->garbageCollect();

        ++leftIt;
    }

    while (rightIt != iqpe->end()) {
        auto multRight = dd->multiply(e, (*rightIt)->getInverseDD(dd));
        dd->incRef(multRight);
        dd->decRef(e);
        e = multRight;

        dd->garbageCollect();

        ++rightIt;
    }
    const auto finishedEC = std::chrono::steady_clock::now();

    const auto preprocessing = std::chrono::duration<double>(finishedTransformation - start).count();
    const auto verification  = std::chrono::duration<double>(finishedEC - finishedTransformation).count();

    std::stringstream ss{};
    ss << "qpe_exact,transformation," << static_cast<std::size_t>(qpe->getNqubits()) << "," << qpeNgates << ",2," << iqpeNgates << "," << preprocessing << "," << verification;
    std::cout << ss.str() << std::endl;
    ofs.open("results_exact.csv", std::ios_base::app);
    ofs << ss.str() << std::endl;

    EXPECT_TRUE(e.p->ident);
}

TEST_P(DynamicCircuitEvalExactQPE, ProbabilityExtraction) {
    // generate DD of QPE circuit via simulation
    const auto start          = std::chrono::steady_clock::now();
    auto       e              = qpe->simulate(dd->makeZeroState(qpe->getNqubits()), dd);
    const auto simulation_end = std::chrono::steady_clock::now();

    // extract measurement probabilities from IQPE simulations
    std::vector<dd::fp> probs{};
    iqpe->extractProbabilityVector(dd->makeZeroState(iqpe->getNqubits()), probs, dd);
    const auto extraction_end = std::chrono::steady_clock::now();

    // interleave with zeros to account for 0 qubit
    auto stub = std::vector<dd::fp>(1 << (qpe->getNqubits()));
    for (std::size_t i = 1; i < stub.size(); i += 2) {
        stub.at(i) = probs.at((i - 1) / 2);
    }
    // compare outcomes
    auto       fidelity       = dd->fidelityOfMeasurementOutcomes(e, stub);
    const auto comparison_end = std::chrono::steady_clock::now();

    const auto simulation = std::chrono::duration<double>(simulation_end - start).count();
    const auto extraction = std::chrono::duration<double>(extraction_end - simulation_end).count();
    const auto comparison = std::chrono::duration<double>(comparison_end - extraction_end).count();
    const auto total      = std::chrono::duration<double>(comparison_end - start).count();

    std::stringstream ss{};
    ss << "qpe_exact,extraction," << static_cast<std::size_t>(qpe->getNqubits()) << "," << qpeNgates << ",2," << iqpeNgates << "," << simulation << "," << extraction << "," << comparison << "," << total;
    std::cout << ss.str() << std::endl;
    ofs.open("results_exact_prob.csv", std::ios_base::app);
    ofs << ss.str() << std::endl;

    EXPECT_NEAR(fidelity, 1.0, 1e-4);
}

class DynamicCircuitEvalInexactQPE: public testing::TestWithParam<std::size_t> {
protected:
    dd::QubitCount                          precision{};
    dd::fp                                  theta{};
    std::size_t                             expectedResult{};
    std::string                             expectedResultRepresentation{};
    std::size_t                             secondExpectedResult{};
    std::string                             secondExpectedResultRepresentation{};
    std::unique_ptr<qc::QuantumComputation> qpe;
    std::unique_ptr<qc::QuantumComputation> iqpe;
    std::size_t                             qpeNgates{};
    std::size_t                             iqpeNgates{};
    std::unique_ptr<dd::Package>            dd;
    std::ofstream                           ofs;

    void TearDown() override {}
    void SetUp() override {
        precision = GetParam();

        dd = std::make_unique<dd::Package>(precision + 1);

        qpe = std::make_unique<qc::QPE>(precision, false);
        // remove final measurements so that the functionality is unitary
        qc::CircuitOptimizer::removeFinalMeasurements(*qpe);
        qpeNgates = qpe->getNindividualOps();

        const auto lambda = dynamic_cast<qc::QPE*>(qpe.get())->lambda;
        iqpe              = std::make_unique<qc::IQPE>(lambda, precision);
        iqpeNgates        = iqpe->getNindividualOps();

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
};

INSTANTIATE_TEST_SUITE_P(Eval, DynamicCircuitEvalInexactQPE,
                         testing::Range<std::size_t>(1U, 10U),
                         [](const testing::TestParamInfo<DynamicCircuitEvalInexactQPE::ParamType>& info) {
            dd::QubitCount nqubits = info.param;
            std::stringstream ss{};
            ss << static_cast<std::size_t>(nqubits);
            if (nqubits == 1) {
                ss << "_qubit";
            } else {
                ss << "_qubits";
            }
            return ss.str(); });

TEST_P(DynamicCircuitEvalInexactQPE, UnitaryTransformation) {
    const auto start = std::chrono::steady_clock::now();
    // transform dynamic circuit to unitary circuit by first eliminating reset operations and afterwards deferring measurements to the end of the circuit
    qc::CircuitOptimizer::eliminateResets(*iqpe);
    qc::CircuitOptimizer::deferMeasurements(*iqpe);

    // remove final measurements in order to just obtain the unitary functionality
    qc::CircuitOptimizer::removeFinalMeasurements(*iqpe);
    const auto finishedTransformation = std::chrono::steady_clock::now();

    qc::MatrixDD e = dd->makeIdent(precision + 1);
    dd->incRef(e);

    auto leftIt  = qpe->begin();
    auto rightIt = iqpe->begin();

    while (leftIt != qpe->end() && rightIt != iqpe->end()) {
        auto multLeft  = dd->multiply((*leftIt)->getDD(dd), e);
        auto multRight = dd->multiply(multLeft, (*rightIt)->getInverseDD(dd));
        dd->incRef(multRight);
        dd->decRef(e);
        e = multRight;

        dd->garbageCollect();

        ++leftIt;
        ++rightIt;
    }

    while (leftIt != qpe->end()) {
        auto multLeft = dd->multiply((*leftIt)->getDD(dd), e);
        dd->incRef(multLeft);
        dd->decRef(e);
        e = multLeft;

        dd->garbageCollect();

        ++leftIt;
    }

    while (rightIt != iqpe->end()) {
        auto multRight = dd->multiply(e, (*rightIt)->getInverseDD(dd));
        dd->incRef(multRight);
        dd->decRef(e);
        e = multRight;

        dd->garbageCollect();

        ++rightIt;
    }
    const auto finishedEC = std::chrono::steady_clock::now();

    const auto preprocessing = std::chrono::duration<double>(finishedTransformation - start).count();
    const auto verification  = std::chrono::duration<double>(finishedEC - finishedTransformation).count();

    std::stringstream ss{};
    ss << "qpe_inexact,transformation," << static_cast<std::size_t>(qpe->getNqubits()) << "," << qpeNgates << ",2," << iqpeNgates << "," << preprocessing << "," << verification;
    std::cout << ss.str() << std::endl;
    ofs.open("results_inexact.csv", std::ios_base::app);
    ofs << ss.str() << std::endl;

    EXPECT_TRUE(e.p->ident);
}

TEST_P(DynamicCircuitEvalInexactQPE, ProbabilityExtraction) {
    const auto start = std::chrono::steady_clock::now();
    // extract measurement probabilities from IQPE simulations
    std::vector<dd::fp> probs{};
    iqpe->extractProbabilityVector(dd->makeZeroState(iqpe->getNqubits()), probs, dd);
    const auto extraction_end = std::chrono::steady_clock::now();
    std::cout << "---- extraction done ----" << std::endl;

    // generate DD of QPE circuit via simulation
    auto       e              = qpe->simulate(dd->makeZeroState(qpe->getNqubits()), dd);
    const auto simulation_end = std::chrono::steady_clock::now();
    std::cout << "---- sim done ----" << std::endl;

    // interleave with zeros to account for 0 qubit
    auto stub = std::vector<dd::fp>(1 << (qpe->getNqubits()));
    for (std::size_t i = 1; i < stub.size(); i += 2) {
        stub.at(i) = probs.at((i - 1) / 2);
    }
    // compare outcomes
    auto       fidelity       = dd->fidelityOfMeasurementOutcomes(e, stub);
    const auto comparison_end = std::chrono::steady_clock::now();

    const auto extraction = std::chrono::duration<double>(extraction_end - start).count();
    const auto simulation = std::chrono::duration<double>(simulation_end - extraction_end).count();
    const auto comparison = std::chrono::duration<double>(comparison_end - simulation_end).count();
    const auto total      = std::chrono::duration<double>(comparison_end - start).count();

    std::stringstream ss{};
    ss << "qpe_inexact,extraction," << static_cast<std::size_t>(qpe->getNqubits()) << "," << qpeNgates << ",2," << iqpeNgates << "," << simulation << "," << extraction << "," << comparison << "," << total;
    std::cout << ss.str() << std::endl;
    ofs.open("results_inexact_prob.csv", std::ios_base::app);
    ofs << ss.str() << std::endl;

    EXPECT_NEAR(fidelity, 1.0, 1e-4);
}
