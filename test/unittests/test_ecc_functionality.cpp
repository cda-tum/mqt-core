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

using circuitFunctions = std::function<std::shared_ptr<qc::QuantumComputation>()>;

struct testCase {
    circuitFunctions       circuit;
    bool                   testNoise;
    size_t                 insertNoiseAfterNQubits;
    std::vector<dd::Qubit> dataQubits;
};

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
    static bool testCircuits(const std::vector<testCase>& circuitsExpectToPass) {
        size_t circuitCounter = 0;
        for (const auto& testParameter: circuitsExpectToPass) {
            auto qcOriginal = testParameter.circuit();
            auto mapper     = std::make_unique<eccType>(qcOriginal, 0);
            mapper->apply();
            circuitCounter++;
            std::cout << "Testing circuit " << circuitCounter << std::endl;
            bool const success = mapper->verifyExecution(testParameter.testNoise, testParameter.dataQubits, testParameter.insertNoiseAfterNQubits);
            if (!success) {
                return false;
            }
        }
        return true;
    }
};

TEST_F(DDECCFunctionalityTest, testIdEcc) {
    size_t                 insertNoiseAfterNQubits = 1;
    std::vector<dd::Qubit> oneQubitDataQubits      = {};

    std::vector<testCase> circuitsExpectToPass = {
            {createIdentityCircuit, false, insertNoiseAfterNQubits, oneQubitDataQubits},
            {createXCircuit, false, insertNoiseAfterNQubits, oneQubitDataQubits},
            {createYCircuit, false, insertNoiseAfterNQubits, oneQubitDataQubits},
            {createHCircuit, false, insertNoiseAfterNQubits, oneQubitDataQubits},
            {createHTCircuit, false, insertNoiseAfterNQubits, oneQubitDataQubits},
            {createHZCircuit, false, insertNoiseAfterNQubits, oneQubitDataQubits},
            {createCXCircuit, false, insertNoiseAfterNQubits, oneQubitDataQubits},
            {createCZCircuit, false, insertNoiseAfterNQubits, oneQubitDataQubits},
            {createCYCircuit, false, insertNoiseAfterNQubits, oneQubitDataQubits},
    };
    EXPECT_TRUE(testCircuits<ecc::Id>(circuitsExpectToPass));
}

TEST_F(DDECCFunctionalityTest, testQ3Shor) {
    size_t                 insertNoiseAfterNQubits = 4;
    std::vector<dd::Qubit> oneQubitDataQubits      = {0, 1, 2};
    std::vector<dd::Qubit> twoQubitDataQubits      = {0, 1, 2, 3, 4, 5};

    std::vector<testCase> circuitsExpectToPass = {
            {createIdentityCircuit, true, insertNoiseAfterNQubits, oneQubitDataQubits},
            {createXCircuit, true, insertNoiseAfterNQubits, oneQubitDataQubits},
            {createCXCircuit, true, insertNoiseAfterNQubits * 2, twoQubitDataQubits},
            {createCYCircuit, true, insertNoiseAfterNQubits * 2, twoQubitDataQubits},
    };
    std::vector<testCase> circuitsExpectToFail = {
            {createYCircuit, true, insertNoiseAfterNQubits, oneQubitDataQubits},
            {createHCircuit, true, insertNoiseAfterNQubits, oneQubitDataQubits},
            {createHTCircuit, true, insertNoiseAfterNQubits, oneQubitDataQubits},
            {createCZCircuit, true, insertNoiseAfterNQubits, oneQubitDataQubits},
            {createHZCircuit, true, insertNoiseAfterNQubits, oneQubitDataQubits},
    };
    EXPECT_TRUE(testCircuits<ecc::Q3Shor>(circuitsExpectToPass));
    EXPECT_ANY_THROW(testCircuits<ecc::Q3Shor>(circuitsExpectToFail));
}

TEST_F(DDECCFunctionalityTest, testQ5LaflammeEcc) {
    size_t                 insertNoiseAfterNQubits = 61;
    std::vector<dd::Qubit> oneQubitDataQubits      = {0, 1, 2, 3, 4};

    std::vector<testCase> circuitsExpectToPass = {
            {createIdentityCircuit, true, insertNoiseAfterNQubits, oneQubitDataQubits},
            {createXCircuit, true, insertNoiseAfterNQubits, oneQubitDataQubits},
    };
    std::vector<testCase> circuitsExpectToFail = {
            {createYCircuit, true, insertNoiseAfterNQubits, oneQubitDataQubits},
            {createHCircuit, true, insertNoiseAfterNQubits, oneQubitDataQubits},
            {createHTCircuit, true, insertNoiseAfterNQubits, oneQubitDataQubits},
            {createHZCircuit, true, insertNoiseAfterNQubits, oneQubitDataQubits},
            {createCXCircuit, true, insertNoiseAfterNQubits, oneQubitDataQubits},
            {createCZCircuit, true, insertNoiseAfterNQubits, oneQubitDataQubits},
            {createCYCircuit, true, insertNoiseAfterNQubits, oneQubitDataQubits},
    };
    EXPECT_TRUE(testCircuits<ecc::Q5Laflamme>(circuitsExpectToPass));
    EXPECT_ANY_THROW(testCircuits<ecc::Q5Laflamme>(circuitsExpectToFail));
}

TEST_F(DDECCFunctionalityTest, testQ7Steane) {
    size_t                 insertNoiseAfterNQubits = 57;
    std::vector<dd::Qubit> oneQubitDataQubits      = {0, 1, 2, 3, 4, 5, 6};
    std::vector<dd::Qubit> twoQubitDataQubits      = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};

    std::vector<testCase> circuitsExpectToPass = {
            {createIdentityCircuit, true, insertNoiseAfterNQubits, oneQubitDataQubits},
            {createXCircuit, true, insertNoiseAfterNQubits, oneQubitDataQubits},
            {createYCircuit, true, insertNoiseAfterNQubits, oneQubitDataQubits},
            {createHCircuit, true, insertNoiseAfterNQubits, oneQubitDataQubits},
            {createHTCircuit, true, insertNoiseAfterNQubits, oneQubitDataQubits},
            {createHZCircuit, true, insertNoiseAfterNQubits, oneQubitDataQubits},
            {createCXCircuit, true, insertNoiseAfterNQubits * 2, twoQubitDataQubits},
            {createCZCircuit, true, insertNoiseAfterNQubits * 2, twoQubitDataQubits},
            {createCYCircuit, true, insertNoiseAfterNQubits * 2, twoQubitDataQubits},
    };
    EXPECT_TRUE(testCircuits<ecc::Q7Steane>(circuitsExpectToPass));
}

TEST_F(DDECCFunctionalityTest, testQ9ShorEcc) {
    size_t                 insertNoiseAfterNQubits = 7;
    std::vector<dd::Qubit> oneQubitDataQubits      = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<dd::Qubit> twoQubitDataQubits      = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17};

    std::vector<testCase> circuitsExpectToPass = {
            {createIdentityCircuit, true, insertNoiseAfterNQubits, oneQubitDataQubits},
            {createXCircuit, true, insertNoiseAfterNQubits, oneQubitDataQubits},
            {createCXCircuit, true, insertNoiseAfterNQubits * 2, twoQubitDataQubits},
            {createCYCircuit, true, insertNoiseAfterNQubits * 2, twoQubitDataQubits},
    };

    std::vector<testCase> circuitsExpectToFail = {
            {createYCircuit, true, insertNoiseAfterNQubits, oneQubitDataQubits},
            {createHCircuit, true, insertNoiseAfterNQubits, oneQubitDataQubits},
            {createHTCircuit, true, insertNoiseAfterNQubits, oneQubitDataQubits},
            {createHZCircuit, true, insertNoiseAfterNQubits, oneQubitDataQubits},
            {createCZCircuit, true, insertNoiseAfterNQubits, oneQubitDataQubits},
    };

    EXPECT_TRUE(testCircuits<ecc::Q9Shor>(circuitsExpectToPass));
    EXPECT_ANY_THROW(testCircuits<ecc::Q9Shor>(circuitsExpectToFail));
}

TEST_F(DDECCFunctionalityTest, testQ9SurfaceEcc) {
    size_t                 insertNoiseAfterNQubits = 55;
    std::vector<dd::Qubit> oneQubitDataQubits      = {0, 1, 2, 3, 4, 5, 6, 7, 8};

    std::vector<testCase> circuitsExpectToPass = {
            {createIdentityCircuit, true, insertNoiseAfterNQubits, oneQubitDataQubits},
            {createXCircuit, true, insertNoiseAfterNQubits, oneQubitDataQubits},
            {createYCircuit, true, insertNoiseAfterNQubits, oneQubitDataQubits},
            {createHCircuit, true, insertNoiseAfterNQubits, oneQubitDataQubits},
            {createHZCircuit, true, insertNoiseAfterNQubits, oneQubitDataQubits},
    };

    std::vector<testCase> circuitsExpectToFail = {
            {createHTCircuit, true, insertNoiseAfterNQubits, oneQubitDataQubits},
            {createCXCircuit, true, insertNoiseAfterNQubits, oneQubitDataQubits},
            {createCZCircuit, true, insertNoiseAfterNQubits, oneQubitDataQubits},
            {createCYCircuit, true, insertNoiseAfterNQubits, oneQubitDataQubits},
    };

    EXPECT_TRUE(testCircuits<ecc::Q9Surface>(circuitsExpectToPass));
    EXPECT_ANY_THROW(testCircuits<ecc::Q9Surface>(circuitsExpectToFail));
}

TEST_F(DDECCFunctionalityTest, testQ18SurfaceEcc) {
    size_t                 insertNoiseAfterNQubits = 115;
    std::vector<dd::Qubit> oneQubitDataQubits      = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17};

    std::vector<testCase> circuitsExpectToPass{
            {createIdentityCircuit, true, insertNoiseAfterNQubits, oneQubitDataQubits},
            {createXCircuit, true, insertNoiseAfterNQubits, oneQubitDataQubits},
            {createYCircuit, true, insertNoiseAfterNQubits, oneQubitDataQubits},
            {createHCircuit, true, insertNoiseAfterNQubits, oneQubitDataQubits},
            {createHZCircuit, true, insertNoiseAfterNQubits, oneQubitDataQubits},
    };

    std::vector<testCase> circuitsExpectToFail = {
            {createHTCircuit, true, insertNoiseAfterNQubits, oneQubitDataQubits},
            {createCXCircuit, true, insertNoiseAfterNQubits, oneQubitDataQubits},
            {createCZCircuit, true, insertNoiseAfterNQubits, oneQubitDataQubits},
            {createCYCircuit, true, insertNoiseAfterNQubits, oneQubitDataQubits},
    };

    EXPECT_TRUE(testCircuits<ecc::Q18Surface>(circuitsExpectToPass));
    EXPECT_ANY_THROW(testCircuits<ecc::Q18Surface>(circuitsExpectToFail));
}
