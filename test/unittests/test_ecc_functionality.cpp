/*
* This file is part of MQT QFR library which is released under the MIT license.
* See file README.md or go to https://www.cda.cit.tum.de/research/quantum/ for more information.
*/

#include "gtest/gtest.h"
#include <ecc/Ecc.hpp>
#include <ecc/Id.hpp>
#include <ecc/Q18Surface.hpp>
#include <ecc/Q3Shor.hpp>
#include <ecc/Q5Laflamme.hpp>
#include <ecc/Q7Steane.hpp>
#include <ecc/Q9Shor.hpp>
#include <ecc/Q9Surface.hpp>
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
    static bool testCircuits(const circuitFunctions& circuitsExpectToPass, bool simulateNoise = false) {
        size_t circuitCounter = 0;
        for (const auto& circuit: circuitsExpectToPass) {
            auto qcOriginal = circuit();
            auto mapper     = std::make_unique<eccType>(qcOriginal, 0);
            mapper->apply();
            circuitCounter++;
            std::cout << "Testing circuit " << circuitCounter << std::endl;
            bool const success = mapper->verifyExecution(simulateNoise);
            if (!success) {
                return false;
            }
        }
        return true;
    }
};

TEST_F(DDECCFunctionalityTest, testIdEcc) {
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

    EXPECT_TRUE(testCircuits<Id>(circuitsExpectToPass, false));
}

TEST_F(DDECCFunctionalityTest, testQ3Shor) {
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

    EXPECT_TRUE(testCircuits<Q3Shor>(circuitsExpectToPass, true));
    EXPECT_ANY_THROW(testCircuits<Q3Shor>(circuitsExpectToFail, true));
}

TEST_F(DDECCFunctionalityTest, testQ5LaflammeEcc) {
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

    EXPECT_TRUE(testCircuits<Q5Laflamme>(circuitsExpectToPass, true));
    EXPECT_ANY_THROW(testCircuits<Q5Laflamme>(circuitsExpectToFail, true));
}

TEST_F(DDECCFunctionalityTest, testQ7Steane) {
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

    EXPECT_TRUE(testCircuits<Q7Steane>(circuitsExpectToPass, true));
}

TEST_F(DDECCFunctionalityTest, testQ9ShorEcc) {
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

    EXPECT_TRUE(testCircuits<Q9Shor>(circuitsExpectToPass, true));
    EXPECT_ANY_THROW(testCircuits<Q9Shor>(circuitsExpectToFail, true));
}

TEST_F(DDECCFunctionalityTest, testQ9SurfaceEcc) {
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

    EXPECT_TRUE(testCircuits<Q9Surface>(circuitsExpectToPass, true));
    EXPECT_ANY_THROW(testCircuits<Q9Surface>(circuitsExpectToFail, true));
}

TEST_F(DDECCFunctionalityTest, testQ18SurfaceEcc) {
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

    EXPECT_TRUE(testCircuits<Q18Surface>(circuitsExpectToPass, true));
    EXPECT_ANY_THROW(testCircuits<Q18Surface>(circuitsExpectToFail, true));
}
