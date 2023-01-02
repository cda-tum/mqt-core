/*
* This file is part of MQT QFR library which is released under the MIT license.
* See file README.md or go to https://www.cda.cit.tum.de/research/quantum/ for more information.
*/

#include "gtest/gtest.h"
#include "dd/Simulation.hpp"
#include "ecc/Ecc.hpp"
#include "ecc/Id.hpp"
#include "ecc/Q18Surface.hpp"
#include "ecc/Q3Shor.hpp"
#include "ecc/Q5Laflamme.hpp"
#include "ecc/Q7Steane.hpp"
#include "ecc/Q9Shor.hpp"
#include "ecc/Q9Surface.hpp"
#include <random>

using namespace qc;

using circuitFunctions = std::function<std::shared_ptr<qc::QuantumComputation>()>;

struct testCase {
    circuitFunctions circuit;
    bool             testNoise;
    size_t           insertNoiseAfterNGates;
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
    bool testCircuits(const std::vector<testCase>& circuitsExpectToPass) {
        size_t circuitCounter = 0;
        for (const auto& testParameter: circuitsExpectToPass) {
            auto qcOriginal = testParameter.circuit();
            auto mapper     = std::make_unique<eccType>(qcOriginal, 0);
            mapper->apply();
            circuitCounter++;
            std::cout << "Testing circuit " << circuitCounter << std::endl;
            bool const success = testErrorCorrectionCircuit(mapper->getOriginalCircuit(), mapper->getMappedCircuit(), testParameter.testNoise, mapper->getDataQubits(), testParameter.insertNoiseAfterNGates);
            if (!success) {
                return false;
            }
        }
        return true;
    }

    static bool testErrorCorrectionCircuit(const std::shared_ptr<qc::QuantumComputation>& qcOriginal, const std::shared_ptr<qc::QuantumComputation>& qcMapped, bool simulateWithErrors, const std::vector<Qubit>& dataQubits = {}, std::size_t insertErrorAfterNGates = 0) {
        if (!simulateWithErrors) {
            return simulateAndVerifyResults(qcOriginal, qcMapped);
        } else {
            for (auto target: dataQubits) {
                auto initialQuit = qcMapped->begin();
                qcMapped->insert(initialQuit + static_cast<long>(insertErrorAfterNGates), std::make_unique<NonUnitaryOperation>(qcMapped->getNqubits(), std::vector<Qubit>{target}, qc::Reset));
                auto result = simulateAndVerifyResults(qcOriginal, qcMapped, simulateWithErrors, insertErrorAfterNGates, target);
                qcMapped->erase(initialQuit + static_cast<long>(insertErrorAfterNGates));
                if (!result) {
                    return false;
                }
            }
        }
        return true;
    }

    static bool simulateAndVerifyResults(const std::shared_ptr<qc::QuantumComputation>& qcOriginal,
                                         const std::shared_ptr<qc::QuantumComputation>& qcMapped,
                                         bool                                           simulateWithErrors     = false,
                                         size_t                                         insertErrorAfterNGates = 0,
                                         unsigned int                                   target                 = 0,
                                         double                                         tolerance              = 0.25,
                                         int                                            shots                  = 50,
                                         int                                            seed                   = 1) {
        auto toleranceAbsolute = (static_cast<double>(shots) / 100.0) * (tolerance * 100.0);

        auto ddOriginal       = std::make_unique<dd::Package<>>(qcOriginal->getNqubits());
        auto originalRootEdge = ddOriginal->makeZeroState(qcOriginal->getNqubits());
        ddOriginal->incRef(originalRootEdge);

        auto measurementsOriginal = simulate(qcOriginal.get(), originalRootEdge, ddOriginal, shots);

        auto ddEcc       = std::make_unique<dd::Package<>>(qcMapped->getNqubits());
        auto eccRootEdge = ddEcc->makeZeroState(qcMapped->getNqubits());
        ddEcc->incRef(eccRootEdge);

        auto measurementsProtected = simulate(qcMapped.get(), eccRootEdge, ddEcc, shots, seed);
        for (auto const& [classicalBit, hits]: measurementsOriginal) {
            // Since the result is stored as one bit string. I have to count the relevant classical bits.
            size_t eccHits = 0;
            for (auto const& [eccMeasure, tempHits]: measurementsProtected) {
                if (0 == eccMeasure.compare(eccMeasure.length() - classicalBit.length(), classicalBit.length(), classicalBit)) {
                    eccHits += tempHits;
                }
            }
            auto difference = std::max(eccHits, hits) - std::min(eccHits, hits);
            std::cout << "Diff/tolerance " << difference << "/" << toleranceAbsolute << " Original register: " << hits << " ecc register: " << eccHits;
            if (simulateWithErrors) {
                std::cout << " Simulating an error in qubit " << static_cast<unsigned>(target) << " after " << insertErrorAfterNGates << " gates." << std::endl;
            }
            if (static_cast<double>(difference) > toleranceAbsolute) {
                std::cout << "Error is too large!" << std::endl;
                return false;
            }
        }
        return true;
    }
};

TEST_F(DDECCFunctionalityTest, testIdEcc) {
    size_t insertNoiseAfterNQubits = 1;

    std::vector<testCase> const circuitsExpectToPass = {
            {createIdentityCircuit, false, insertNoiseAfterNQubits},
            {createXCircuit, false, insertNoiseAfterNQubits},
            {createYCircuit, false, insertNoiseAfterNQubits},
            {createHCircuit, false, insertNoiseAfterNQubits},
            {createHTCircuit, false, insertNoiseAfterNQubits},
            {createHZCircuit, false, insertNoiseAfterNQubits},
            {createCXCircuit, false, insertNoiseAfterNQubits},
            {createCZCircuit, false, insertNoiseAfterNQubits},
            {createCYCircuit, false, insertNoiseAfterNQubits},
    };
    EXPECT_TRUE(testCircuits<ecc::Id>(circuitsExpectToPass));
}

TEST_F(DDECCFunctionalityTest, testQ3Shor) {
    size_t insertNoiseAfterNQubits = 4;

    std::vector<testCase> const circuitsExpectToPass = {
            {createIdentityCircuit, true, insertNoiseAfterNQubits},
            {createXCircuit, true, insertNoiseAfterNQubits},
            {createCXCircuit, true, insertNoiseAfterNQubits * 2},
            {createCYCircuit, true, insertNoiseAfterNQubits * 2},
    };
    std::vector<testCase> const circuitsExpectToFail = {
            {createYCircuit, true, insertNoiseAfterNQubits},
            {createHCircuit, true, insertNoiseAfterNQubits},
            {createHTCircuit, true, insertNoiseAfterNQubits},
            {createCZCircuit, true, insertNoiseAfterNQubits},
            {createHZCircuit, true, insertNoiseAfterNQubits},
    };
    EXPECT_TRUE(testCircuits<ecc::Q3Shor>(circuitsExpectToPass));
    EXPECT_ANY_THROW(testCircuits<ecc::Q3Shor>(circuitsExpectToFail));
}

TEST_F(DDECCFunctionalityTest, testQ5LaflammeEcc) {
    size_t insertNoiseAfterNQubits = 61;

    std::vector<testCase> const circuitsExpectToPass = {
            {createIdentityCircuit, true, insertNoiseAfterNQubits},
            {createXCircuit, true, insertNoiseAfterNQubits},
    };
    std::vector<testCase> const circuitsExpectToFail = {
            {createYCircuit, true, insertNoiseAfterNQubits},
            {createHCircuit, true, insertNoiseAfterNQubits},
            {createHTCircuit, true, insertNoiseAfterNQubits},
            {createHZCircuit, true, insertNoiseAfterNQubits},
            {createCXCircuit, true, insertNoiseAfterNQubits},
            {createCZCircuit, true, insertNoiseAfterNQubits},
            {createCYCircuit, true, insertNoiseAfterNQubits},
    };
    EXPECT_TRUE(testCircuits<ecc::Q5Laflamme>(circuitsExpectToPass));
    EXPECT_ANY_THROW(testCircuits<ecc::Q5Laflamme>(circuitsExpectToFail));
}

TEST_F(DDECCFunctionalityTest, testQ7Steane) {
    size_t insertNoiseAfterNQubits = 57;

    std::vector<testCase> const circuitsExpectToPass = {
            {createIdentityCircuit, true, insertNoiseAfterNQubits},
            {createXCircuit, true, insertNoiseAfterNQubits},
            {createYCircuit, true, insertNoiseAfterNQubits},
            {createHCircuit, true, insertNoiseAfterNQubits},
            {createHTCircuit, true, insertNoiseAfterNQubits},
            {createHZCircuit, true, insertNoiseAfterNQubits},
            {createCXCircuit, true, insertNoiseAfterNQubits * 2},
            {createCZCircuit, true, insertNoiseAfterNQubits * 2},
            {createCYCircuit, true, insertNoiseAfterNQubits * 2},
    };
    EXPECT_TRUE(testCircuits<ecc::Q7Steane>(circuitsExpectToPass));
}

TEST_F(DDECCFunctionalityTest, testQ9ShorEcc) {
    size_t insertNoiseAfterNQubits = 7;

    std::vector<testCase> const circuitsExpectToPass = {
            {createIdentityCircuit, true, insertNoiseAfterNQubits},
            {createXCircuit, true, insertNoiseAfterNQubits},
            {createCXCircuit, true, insertNoiseAfterNQubits * 2},
            {createCYCircuit, true, insertNoiseAfterNQubits * 2},
    };

    std::vector<testCase> const circuitsExpectToFail = {
            {createYCircuit, true, insertNoiseAfterNQubits},
            {createHCircuit, true, insertNoiseAfterNQubits},
            {createHTCircuit, true, insertNoiseAfterNQubits},
            {createHZCircuit, true, insertNoiseAfterNQubits},
            {createCZCircuit, true, insertNoiseAfterNQubits},
    };

    EXPECT_TRUE(testCircuits<ecc::Q9Shor>(circuitsExpectToPass));
    EXPECT_ANY_THROW(testCircuits<ecc::Q9Shor>(circuitsExpectToFail));
}

TEST_F(DDECCFunctionalityTest, testQ9SurfaceEcc) {
    size_t insertNoiseAfterNQubits = 55;

    std::vector<testCase> const circuitsExpectToPass = {
            {createIdentityCircuit, true, insertNoiseAfterNQubits},
            {createXCircuit, true, insertNoiseAfterNQubits},
            {createYCircuit, true, insertNoiseAfterNQubits},
            {createHCircuit, true, insertNoiseAfterNQubits},
            {createHZCircuit, true, insertNoiseAfterNQubits},
    };

    std::vector<testCase> const circuitsExpectToFail = {
            {createHTCircuit, true, insertNoiseAfterNQubits},
            {createCXCircuit, true, insertNoiseAfterNQubits},
            {createCZCircuit, true, insertNoiseAfterNQubits},
            {createCYCircuit, true, insertNoiseAfterNQubits},
    };

    EXPECT_TRUE(testCircuits<ecc::Q9Surface>(circuitsExpectToPass));
    EXPECT_ANY_THROW(testCircuits<ecc::Q9Surface>(circuitsExpectToFail));
}

TEST_F(DDECCFunctionalityTest, testQ18SurfaceEcc) {
    size_t insertNoiseAfterNQubits = 115;

    std::vector<testCase> const circuitsExpectToPass{
            {createIdentityCircuit, false, insertNoiseAfterNQubits},
            {createXCircuit, false, insertNoiseAfterNQubits},
            {createYCircuit, false, insertNoiseAfterNQubits},
            {createHCircuit, false, insertNoiseAfterNQubits},
            {createHZCircuit, false, insertNoiseAfterNQubits},
    };

    std::vector<testCase> const circuitsExpectToFail = {
            {createHTCircuit, true, insertNoiseAfterNQubits},
            {createCXCircuit, true, insertNoiseAfterNQubits},
            {createCZCircuit, true, insertNoiseAfterNQubits},
            {createCYCircuit, true, insertNoiseAfterNQubits},
    };

    EXPECT_TRUE(testCircuits<ecc::Q18Surface>(circuitsExpectToPass));
    EXPECT_ANY_THROW(testCircuits<ecc::Q18Surface>(circuitsExpectToFail));
}
