/*
* This file is part of MQT QFR library which is released under the MIT license.
* See file README.md or go to https://www.cda.cit.tum.de/research/quantum/ for more information.
*/

#include "gtest/gtest.h"
#include <dd/Simulation.hpp>
#include <eccs/Ecc.hpp>
#include <eccs/IdEcc.hpp>
#include <eccs/Q18SurfaceEcc.hpp>
#include <eccs/Q3ShorEcc.hpp>
#include <eccs/Q5LaflammeEcc.hpp>
#include <eccs/Q7SteaneEcc.hpp>
#include <eccs/Q9ShorEcc.hpp>
#include <eccs/Q9SurfaceEcc.hpp>
#include <random>

using namespace qc;
using namespace dd;

typedef std::vector<std::function<std::shared_ptr<qc::QuantumComputation>()>> circuitFunctions;

class DDECCFunctionalityTest: public ::testing::Test {
protected:
    void SetUp() override {}

    void TearDown() override {}

    static bool verifyExecution(const std::shared_ptr<qc::QuantumComputation>& qcOriginal, const std::shared_ptr<qc::QuantumComputation>& qcECC,
                                bool                          simulateWithErrors     = false,
                                const std::vector<dd::Qubit>& dataQubits             = {},
                                int                           insertErrorAfterNGates = 0) {
        auto   shots             = 80;
        double tolerance         = 0.8;
        size_t seed              = 1;
        auto   toleranceAbsolute = (shots / 100.0) * (tolerance * 100.0);

        std::unique_ptr<dd::Package<>> ddOriginal       = std::make_unique<dd::Package<>>(qcOriginal->getNqubits());
        auto                           originalRootEdge = ddOriginal->makeZeroState(qcOriginal->getNqubits());
        ddOriginal->incRef(originalRootEdge);

        std::unique_ptr<dd::Package<>> ddEcc       = std::make_unique<dd::Package<>>(qcECC->getNqubits());
        auto                           eccRootEdge = ddEcc->makeZeroState(qcECC->getNqubits());
        ddEcc->incRef(eccRootEdge);

        std::map<std::string, size_t> measurementsOriginal = simulate(qcOriginal.get(), originalRootEdge, ddOriginal, shots, seed);

        if (!simulateWithErrors) {
            std::map<std::string, size_t> measurementsProtected = simulate(qcECC.get(), eccRootEdge, ddEcc, shots, seed);
            for (auto const& [cBitsOriginal, cHitsOriginal]: measurementsOriginal) {
                // Count the cHitsOriginal in the register with error correction
                size_t cHitsProtected = 0;
                for (auto const& [cBitsProtected, cHitsProtectedTemp]: measurementsProtected) {
                    if (0 == cBitsProtected.compare(cBitsProtected.length() - cBitsOriginal.length(), cBitsOriginal.length(), cBitsOriginal)) cHitsProtected += cHitsProtectedTemp;
                }
                auto difference = std::max(cHitsProtected, cHitsOriginal) - std::min(cHitsProtected, cHitsOriginal);
                if (static_cast<double>(difference) > toleranceAbsolute) {
                    return false;
                }
            }
        } else {
            for (auto const& qubit: dataQubits) {
                std::map<std::string, size_t> measurementsProtected = simulate(qcECC.get(), eccRootEdge, ddEcc, shots, seed, qubit, insertErrorAfterNGates, true);
                for (auto const& [classicalBit, hits]: measurementsOriginal) {
                    // Since the result is stored as one bit string. I have to count the relevant classical bits.
                    size_t eccHits = 0;
                    for (auto const& [eccMeasure, tempHits]: measurementsProtected) {
                        if (0 == eccMeasure.compare(eccMeasure.length() - classicalBit.length(), classicalBit.length(), classicalBit)) eccHits += tempHits;
                    }
                    auto difference = std::max(eccHits, hits) - std::min(eccHits, hits);
                    if (static_cast<double>(difference) > toleranceAbsolute) {
                        std::cout << "Simulation failed when applying error to qubit " << static_cast<unsigned>(qubit) << " after " << insertErrorAfterNGates << " gates.\n";
                        std::cout << "Error in bit " << classicalBit << " original register: " << hits << " ecc register: " << eccHits << std::endl;
                        return false;
                    } else {
                        std::cout << "Diff/tolerance " << difference << "/" << toleranceAbsolute << " Original register: " << hits << " ecc register: " << eccHits;
                        std::cout << " Error at qubit " << static_cast<unsigned>(qubit) << " after " << insertErrorAfterNGates << " gates." << std::endl;
                    }
                }
            }
        }
        return true;
    }

    static std::shared_ptr<qc::QuantumComputation> createIdentityCircuit() {
        auto qc = std::make_shared<qc::QuantumComputation>();
        qc->addQubitRegister(1U);
        qc->addClassicalRegister(1U, "resultReg");
        qc->x(0);
        qc->x(0);
        qc->measure(0, {"resultReg", 0});
        return qc;
    }

    static std::shared_ptr<qc::QuantumComputation> createXCircuit() {
        auto qc = std::make_shared<qc::QuantumComputation>();
        qc->addQubitRegister(1U);
        qc->addClassicalRegister(1U, "resultReg");
        qc->x(0);
        qc->measure(0, {"resultReg", 0});
        return qc;
    }

    static std::shared_ptr<qc::QuantumComputation> createYCircuit() {
        auto qc = std::make_shared<qc::QuantumComputation>();
        qc->addQubitRegister(1U);
        qc->addClassicalRegister(1U, "resultReg");
        qc->h(0);
        qc->y(0);
        qc->h(0);
        qc->measure(0, {"resultReg", 0});
        return qc;
    }

    static std::shared_ptr<qc::QuantumComputation> createHCircuit() {
        auto qc = std::make_shared<qc::QuantumComputation>();
        qc->addQubitRegister(1U);
        qc->addClassicalRegister(1U, "resultReg");
        qc->h(0);
        qc->measure(0, {"resultReg", 0});
        return qc;
    }

    static std::shared_ptr<qc::QuantumComputation> createHTCircuit() {
        auto qc = std::make_shared<qc::QuantumComputation>();
        qc->addQubitRegister(1U);
        qc->addClassicalRegister(1U, "resultReg");
        qc->h(0);
        qc->t(0);
        qc->tdag(0);
        qc->h(0);
        qc->measure(0, {"resultReg", 0});
        return qc;
    }

    static std::shared_ptr<qc::QuantumComputation> createHZCircuit() {
        auto qc = std::make_shared<qc::QuantumComputation>();
        qc->addQubitRegister(1U);
        qc->addClassicalRegister(1U, "resultReg");
        qc->h(0);
        qc->z(0);
        qc->h(0);
        qc->measure(0, {"resultReg", 0});
        return qc;
    }

    static std::shared_ptr<qc::QuantumComputation> createCXCircuit() {
        auto qc = std::make_shared<qc::QuantumComputation>();
        qc->addQubitRegister(2U);
        qc->addClassicalRegister(2U, "resultReg");
        qc->x(0);
        qc->x(1, 0_pc);
        qc->measure(0, {"resultReg", 0});
        qc->measure(1, {"resultReg", 1});
        return qc;
    }

    static std::shared_ptr<qc::QuantumComputation> createCZCircuit() {
        auto qc = std::make_shared<qc::QuantumComputation>();
        qc->addQubitRegister(2U);
        qc->addClassicalRegister(2U, "resultReg");
        qc->x(0);
        qc->h(1);
        qc->z(1, 0_pc);
        qc->h(1);
        qc->measure(0, {"resultReg", 0});
        qc->measure(1, {"resultReg", 1});
        return qc;
    }

    static std::shared_ptr<qc::QuantumComputation> createCYCircuit() {
        auto qc = std::make_shared<qc::QuantumComputation>();
        qc->addQubitRegister(2U);
        qc->addClassicalRegister(2U, "resultReg");
        qc->x(0);
        qc->y(1, 0_pc);
        qc->measure(0, {"resultReg", 0});
        qc->measure(1, {"resultReg", 1});
        return qc;
    }

    template<class eccType>
    static bool testCircuits(const circuitFunctions&       circuitsExpectToPass,
                             int                           measureFrequency       = 0,
                             const std::vector<dd::Qubit>& dataQubits             = {},
                             int                           insertErrorAfterNGates = 0,
                             bool                          simulateNoise          = false) {
        int circuitCounter = 0;
        for (auto& circuit: circuitsExpectToPass) {
            circuitCounter++;
            std::cout << "Testing circuit " << circuitCounter << std::endl;
            auto qcOriginal = circuit();
            auto mapper     = std::make_unique<eccType>(qcOriginal, measureFrequency);
            auto qcECC      = mapper->apply();
            bool success    = verifyExecution(qcOriginal, qcECC, simulateNoise, dataQubits, insertErrorAfterNGates);
            if (!success) {
                return false;
            }
        }
        return true;
    }
};

TEST_F(DDECCFunctionalityTest, testIdEcc) {
    int measureFrequency = 0;

    circuitFunctions circuitsExpectToPass;
    circuitsExpectToPass.emplace_back(createIdentityCircuit);
    circuitsExpectToPass.emplace_back(createXCircuit);
    circuitsExpectToPass.emplace_back(createYCircuit);
    circuitsExpectToPass.emplace_back(createHCircuit);
    circuitsExpectToPass.emplace_back(createHTCircuit);
    circuitsExpectToPass.emplace_back(createHZCircuit);
    circuitsExpectToPass.emplace_back(createCXCircuit);
    circuitsExpectToPass.emplace_back(createCZCircuit);
    circuitsExpectToPass.emplace_back(createCYCircuit);

    EXPECT_TRUE(testCircuits<IdEcc>(circuitsExpectToPass, measureFrequency, {}, 0, false));
}

TEST_F(DDECCFunctionalityTest, testQ3Shor) {
    int measureFrequency = 0;

    circuitFunctions circuitsExpectToPass;
    circuitsExpectToPass.emplace_back(createIdentityCircuit);
    circuitsExpectToPass.emplace_back(createXCircuit);
    circuitsExpectToPass.emplace_back(createCXCircuit);
    circuitsExpectToPass.emplace_back(createCYCircuit);

    circuitFunctions circuitsExpectToFail;
    circuitsExpectToFail.emplace_back(createYCircuit);
    circuitsExpectToFail.emplace_back(createHCircuit);
    circuitsExpectToFail.emplace_back(createHTCircuit);
    circuitsExpectToFail.emplace_back(createCZCircuit);
    circuitsExpectToFail.emplace_back(createHZCircuit);

    std::vector<dd::Qubit> dataQubits             = {0, 1, 2};
    int                    insertErrorAfterNGates = 1;

    EXPECT_TRUE(testCircuits<Q3ShorEcc>(circuitsExpectToPass, measureFrequency, dataQubits, insertErrorAfterNGates, true));
    EXPECT_ANY_THROW(testCircuits<Q3ShorEcc>(circuitsExpectToFail, measureFrequency, dataQubits, insertErrorAfterNGates, true));
}

TEST_F(DDECCFunctionalityTest, testQ5LaflammeEcc) {
    int measureFrequency = 0;

    circuitFunctions circuitsExpectToPass;
    circuitsExpectToPass.emplace_back(createIdentityCircuit);
    circuitsExpectToPass.emplace_back(createXCircuit);

    circuitFunctions circuitsExpectToFail;
    circuitsExpectToFail.emplace_back(createYCircuit);
    circuitsExpectToFail.emplace_back(createHCircuit);
    circuitsExpectToFail.emplace_back(createHTCircuit);
    circuitsExpectToFail.emplace_back(createHZCircuit);
    circuitsExpectToFail.emplace_back(createCXCircuit);
    circuitsExpectToFail.emplace_back(createCZCircuit);
    circuitsExpectToFail.emplace_back(createCYCircuit);

    std::vector<dd::Qubit> dataQubits = {0, 1, 2, 3, 4};

    int insertErrorAfterNGates = 61;
    EXPECT_TRUE(testCircuits<Q5LaflammeEcc>(circuitsExpectToPass, measureFrequency, dataQubits, insertErrorAfterNGates, true));
    EXPECT_ANY_THROW(testCircuits<Q5LaflammeEcc>(circuitsExpectToFail, measureFrequency, dataQubits, insertErrorAfterNGates, true));
}

TEST_F(DDECCFunctionalityTest, testQ7Steane) {
    int measureFrequency = 0;

    circuitFunctions circuitsExpectToPass;
    circuitsExpectToPass.emplace_back(createIdentityCircuit);
    circuitsExpectToPass.emplace_back(createXCircuit);
    circuitsExpectToPass.emplace_back(createYCircuit);
    circuitsExpectToPass.emplace_back(createHCircuit);
    circuitsExpectToPass.emplace_back(createHTCircuit);
    circuitsExpectToPass.emplace_back(createHZCircuit);
    circuitsExpectToPass.emplace_back(createCXCircuit);
    circuitsExpectToPass.emplace_back(createCZCircuit);
    circuitsExpectToPass.emplace_back(createCYCircuit);

    std::vector<dd::Qubit> dataQubits = {0, 1, 2, 3, 4, 5, 6};

    int insertErrorAfterNGates = 57;
    EXPECT_TRUE(testCircuits<Q7SteaneEcc>(circuitsExpectToPass, measureFrequency, dataQubits, insertErrorAfterNGates, true));
}

TEST_F(DDECCFunctionalityTest, testQ9ShorEcc) {
    int measureFrequency = 0;

    circuitFunctions circuitsExpectToPass;
    circuitsExpectToPass.emplace_back(createIdentityCircuit);
    circuitsExpectToPass.emplace_back(createXCircuit);
    circuitsExpectToPass.emplace_back(createCXCircuit);
    circuitsExpectToPass.emplace_back(createCYCircuit);

    circuitFunctions circuitsExpectToFail;
    circuitsExpectToFail.emplace_back(createYCircuit);
    circuitsExpectToFail.emplace_back(createHCircuit);
    circuitsExpectToFail.emplace_back(createHTCircuit);
    circuitsExpectToFail.emplace_back(createHZCircuit);
    circuitsExpectToFail.emplace_back(createCZCircuit);

    std::vector<dd::Qubit> dataQubits = {0, 1, 2, 4, 5, 6, 7, 8};

    int insertErrorAfterNGates = 1;
    EXPECT_TRUE(testCircuits<Q9ShorEcc>(circuitsExpectToPass, measureFrequency, dataQubits, insertErrorAfterNGates, true));
    EXPECT_ANY_THROW(testCircuits<Q9ShorEcc>(circuitsExpectToFail, measureFrequency, dataQubits, insertErrorAfterNGates, true));
}

TEST_F(DDECCFunctionalityTest, testQ9SurfaceEcc) {
    int measureFrequency = 0;

    circuitFunctions circuitsExpectToPass;
    circuitsExpectToPass.emplace_back(createIdentityCircuit);
    circuitsExpectToPass.emplace_back(createXCircuit);
    circuitsExpectToPass.emplace_back(createCXCircuit);
    circuitsExpectToPass.emplace_back(createYCircuit);
    circuitsExpectToPass.emplace_back(createHCircuit);
    circuitsExpectToPass.emplace_back(createHZCircuit);
    circuitsExpectToPass.emplace_back(createCZCircuit);
    circuitsExpectToPass.emplace_back(createCYCircuit);

    circuitFunctions circuitsExpectToFail;
    circuitsExpectToFail.emplace_back(createHTCircuit);

    std::vector<dd::Qubit> dataQubits = {0, 1, 2, 4, 5, 6, 7, 8};

    int insertErrorAfterNGates = 55;
    EXPECT_TRUE(testCircuits<Q9SurfaceEcc>(circuitsExpectToPass, measureFrequency, dataQubits, insertErrorAfterNGates, true));
    EXPECT_ANY_THROW(testCircuits<Q9SurfaceEcc>(circuitsExpectToFail, measureFrequency, dataQubits, insertErrorAfterNGates, true));
}

TEST_F(DDECCFunctionalityTest, testQ18SurfaceEcc) {
    int measureFrequency = 0;

    circuitFunctions circuitsExpectToPass;
    circuitsExpectToPass.emplace_back(createIdentityCircuit);
    circuitsExpectToPass.emplace_back(createXCircuit);
    circuitsExpectToPass.emplace_back(createYCircuit);
    circuitsExpectToPass.emplace_back(createHCircuit);
    circuitsExpectToPass.emplace_back(createHZCircuit);

    circuitFunctions circuitsExpectToFail;
    circuitsExpectToFail.emplace_back(createHTCircuit);
    circuitsExpectToFail.emplace_back(createCXCircuit);
    circuitsExpectToFail.emplace_back(createCZCircuit);
    circuitsExpectToFail.emplace_back(createCYCircuit);

    std::vector<dd::Qubit> dataQubits(Q18SurfaceEcc::dataQubits.begin(), Q18SurfaceEcc::dataQubits.end());

    int insertErrorAfterNGates = 115;
    EXPECT_TRUE(testCircuits<Q18SurfaceEcc>(circuitsExpectToPass, measureFrequency, dataQubits, insertErrorAfterNGates, true));
    EXPECT_ANY_THROW(testCircuits<Q18SurfaceEcc>(circuitsExpectToFail, measureFrequency, dataQubits, insertErrorAfterNGates, true));
}
