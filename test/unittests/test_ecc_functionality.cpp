/*
* This file is part of MQT QFR library which is released under the MIT license.
* See file README.md or go to https://www.cda.cit.tum.de/research/quantum/ for more information.
*/

#include "gtest/gtest.h"
#include <ecc/Ecc.hpp>
#include <ecc/IdEcc.hpp>
#include <ecc/Q18SurfaceEcc.hpp>
#include <ecc/Q3ShorEcc.hpp>
#include <ecc/Q5LaflammeEcc.hpp>
#include <ecc/Q7SteaneEcc.hpp>
#include <ecc/Q9ShorEcc.hpp>
#include <ecc/Q9SurfaceEcc.hpp>
#include <random>

using namespace qc;
using namespace dd;

using circuitFunctions = std::vector<std::function<std::shared_ptr<qc::QuantumComputation>()>>;

class DDECCFunctionalityTest: public ::testing::Test {
protected:
    void SetUp() override {}

    void TearDown() override {}

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
            mapper->apply();
            bool success = mapper->verifyExecution(simulateNoise, dataQubits, insertErrorAfterNGates);
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

    circuitsExpectToPass.emplace_back(createYCircuit);
    circuitsExpectToPass.emplace_back(createHCircuit);
    circuitsExpectToPass.emplace_back(createHZCircuit);

    circuitFunctions circuitsExpectToFail;
    circuitsExpectToFail.emplace_back(createHTCircuit);
    circuitsExpectToFail.emplace_back(createCXCircuit);
    circuitsExpectToFail.emplace_back(createCZCircuit);
    circuitsExpectToFail.emplace_back(createCYCircuit);

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
