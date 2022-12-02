/*
* This file is part of MQT QFR library which is released under the MIT license.
* See file README.md or go to https://www.cda.cit.tum.de/research/quantum/ for more information.
*/

#include "gtest/gtest.h"
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

class DDECCFunctionalityTest: public ::testing::Test {
protected:
    void SetUp() override {}

    void TearDown() override {}

    static void simulateCircuit(std::shared_ptr<qc::QuantumComputation>& qc,
                                std::map<std::size_t,
                                         double>&                        finalClassicValues,
                                std::mt19937_64                          mt,
                                int                                      nrShots                    = 5000,
                                bool                                     simulateNoise              = false,
                                Qubit                                    targetForNoise             = 0,
                                std::uint_fast32_t                       applyNoiseAfterNOperations = 0) {
        std::uint_fast32_t operationCounter = 0;
        for (int sample = 0; sample < nrShots; sample++) {
            std::map<std::size_t, bool> classicValuesECC{};
            auto                        dd       = std::make_unique<dd::Package<>>(qc->getNqubits());
            vEdge                       rootEdge = dd->makeZeroState(qc->getNqubits());
            for (auto const& op: *qc) {
                if (op->getType() == qc::Measure) {
                    auto* nu_op   = dynamic_cast<qc::NonUnitaryOperation*>(op.get());
                    auto  quantum = nu_op->getTargets();
                    auto  classic = nu_op->getClassics();
                    for (unsigned int i = 0; i < quantum.size(); ++i) {
                        dd->incRef(rootEdge);
                        auto result = dd->measureOneCollapsing(rootEdge, quantum.at(i), false, mt);
                        assert(result == '0' || result == '1');
                        classicValuesECC[classic.at(i)] = (result == '1');
                    }
                } else if (op->getType() == qc::Reset) {
                    auto*       nu_op   = dynamic_cast<qc::NonUnitaryOperation*>(op.get());
                    auto const& quantum = nu_op->getTargets();
                    for (signed char qubit: quantum) {
                        dd->incRef(rootEdge);
                        if (auto result = dd->measureOneCollapsing(rootEdge, qubit, false, mt); result == '1') {
                            auto flipOperation = StandardOperation(nu_op->getNqubits(), qubit, qc::X);
                            auto operation     = dd::getDD(&flipOperation, dd);
                            rootEdge           = dd->multiply(operation, rootEdge);
                        }
                        assert(dd->measureOneCollapsing(rootEdge, qubit, false, mt) == '0');
                    }
                } else {
                    if (op->getType() == qc::ClassicControlled) {
                        auto*              cc_op          = dynamic_cast<qc::ClassicControlledOperation*>(op.get());
                        const auto         start_index    = static_cast<unsigned short>(cc_op->getParameter().at(0));
                        const auto         length         = static_cast<unsigned short>(cc_op->getParameter().at(1));
                        const unsigned int expected_value = cc_op->getExpectedValue();
                        unsigned int       actual_value   = 0;
                        for (unsigned int i = 0; i < length; i++) {
                            actual_value |= (classicValuesECC[start_index + i] ? 1u : 0u) << i;
                        }
                        if (actual_value != expected_value) {
                            continue;
                        }
                    }
                    auto operation = dd::getDD(op.get(), dd);
                    rootEdge       = dd->multiply(operation, rootEdge);
                }
                if (simulateNoise && operationCounter == applyNoiseAfterNOperations) {
                    dd->incRef(rootEdge);
                    auto result = dd->measureOneCollapsing(rootEdge, targetForNoise, false, mt);
                    if (result == '1') {
                        auto flipOperation = StandardOperation(qc->getNqubits(), targetForNoise, qc::X);
                        auto operation     = dd::getDD(&flipOperation, dd);
                        rootEdge           = dd->multiply(operation, rootEdge);
                    }
                }
                operationCounter++;
            }
            for (std::size_t i = 0; i < classicValuesECC.size(); i++) {
                // Counting the final hit in resultReg after each shot
                auto regName = qc->returnClassicalRegisterName(i);
                if (regName == "resultReg") {
                    if (finalClassicValues.count(i) == 0) {
                        finalClassicValues[i] = 0;
                    }
                    if (classicValuesECC[i]) {
                        finalClassicValues[i] += 1.0 / nrShots;
                    }
                }
            }
        }
    }

    static bool verifyExecution(std::shared_ptr<qc::QuantumComputation> qcOriginal, std::shared_ptr<qc::QuantumComputation>& qcECC,
                                bool                          simulateWithErrors     = false,
                                const std::vector<dd::Qubit>& dataQubits             = {},
                                int                           insertErrorAfterNGates = 0) {
        double                        tolerance = 0.15;
        double                        aboutOne  = 1.00000001;
        std::map<std::size_t, double> finalClassicValuesOriginal{};
        bool                          testingSuccessful = true;
        auto                          shots             = 50;

        std::mt19937_64 mt(1);

        std::cout.precision(2);

        simulateCircuit(qcOriginal, finalClassicValuesOriginal, mt);

        if (!simulateWithErrors) {
            std::map<std::size_t, double> finalClassicValuesECC{};
            simulateCircuit(qcECC, finalClassicValuesECC, mt, shots);
            for (auto const& [classicalBit, probability]: finalClassicValuesOriginal) {
                assert(probability <= aboutOne && finalClassicValuesECC[classicalBit] <= aboutOne);
                if (std::abs(probability - finalClassicValuesECC[classicalBit]) > tolerance) {
                    testingSuccessful = false;
                }
            }
        } else {
            for (auto const& qubit: dataQubits) {
                std::map<std::size_t, double> finalClassicValuesECC{};
                simulateCircuit(qcECC, finalClassicValuesECC, mt, shots, true, qubit, insertErrorAfterNGates);
                for (auto const& [classicalBit, probability]: finalClassicValuesOriginal) {
                    assert(probability <= aboutOne && finalClassicValuesECC[classicalBit] <= aboutOne);
                    if (std::abs(probability - finalClassicValuesECC[classicalBit]) > tolerance) {
                        std::cout << "Simulation failed when applying error to qubit " << static_cast<unsigned>(qubit) << " after " << insertErrorAfterNGates << " gates.\n";
                        std::cout << "Error in bit " << classicalBit << " original register: " << probability << " ecc register: " << finalClassicValuesECC[classicalBit] << std::endl;
                        testingSuccessful = false;
                    } else {
                        std::cout << "Diff/tolerance " << std::abs(probability - finalClassicValuesECC[classicalBit]) << "/" << tolerance << " Original register: " << probability << " ecc register: " << finalClassicValuesECC[classicalBit];
                        std::cout << " Error at qubit " << static_cast<unsigned>(qubit) << " after " << insertErrorAfterNGates << " gates." << std::endl;
                    }
                }
            }
        }
        return testingSuccessful;
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
};

TEST_F(DDECCFunctionalityTest, testIdEcc) {
    int measureFrequency = 0;

    std::vector<std::shared_ptr<qc::QuantumComputation> (*)()> circuitsExpectToPass;
    circuitsExpectToPass.push_back(createIdentityCircuit);
    circuitsExpectToPass.push_back(createIdentityCircuit);
    circuitsExpectToPass.push_back(createXCircuit);
    circuitsExpectToPass.push_back(createYCircuit);
    circuitsExpectToPass.push_back(createHCircuit);
    circuitsExpectToPass.push_back(createHTCircuit);
    circuitsExpectToPass.push_back(createHZCircuit);
    circuitsExpectToPass.push_back(createCXCircuit);
    circuitsExpectToPass.push_back(createCZCircuit);
    circuitsExpectToPass.push_back(createCYCircuit);

    int circuitCounter = 0;
    for (auto& circuit: circuitsExpectToPass) {
        circuitCounter++;
        std::cout << "Testing circuit " << circuitCounter << std::endl;
        auto qcOriginal = circuit();
        auto mapper     = std::make_unique<IdEcc>(qcOriginal, measureFrequency);
        auto qcECC      = mapper->apply();
        EXPECT_TRUE(verifyExecution(qcOriginal, qcECC));
    }
}

TEST_F(DDECCFunctionalityTest, testQ3Shor) {
    int measureFrequency = 0;

    std::vector<std::shared_ptr<qc::QuantumComputation> (*)()> circuitsExpectToPass;
    circuitsExpectToPass.push_back(createIdentityCircuit);
    circuitsExpectToPass.push_back(createXCircuit);
    circuitsExpectToPass.push_back(createCXCircuit);
    circuitsExpectToPass.push_back(createCYCircuit);

    std::vector<std::shared_ptr<qc::QuantumComputation> (*)()> circuitsExpectToFail;
    circuitsExpectToFail.push_back(createYCircuit);
    circuitsExpectToFail.push_back(createHCircuit);
    circuitsExpectToFail.push_back(createHTCircuit);
    circuitsExpectToFail.push_back(createCZCircuit);
    circuitsExpectToFail.push_back(createHZCircuit);

    int                    circuitCounter = 0;
    std::vector<dd::Qubit> dataQubits     = {0, 1, 2};

    int insertErrorAfterNGates = 1;
    for (auto& circuit: circuitsExpectToPass) {
        circuitCounter++;
        std::cout << "Testing circuit " << circuitCounter << std::endl;
        auto qcOriginal = circuit();
        auto mapper     = std::make_unique<Q3ShorEcc>(qcOriginal, measureFrequency);
        auto qcECC      = mapper->apply();
        EXPECT_TRUE(verifyExecution(qcOriginal, qcECC, true, dataQubits, insertErrorAfterNGates));
    }

    for (auto& circuit: circuitsExpectToFail) {
        auto qcOriginal = circuit();
        auto mapper     = std::make_unique<Q3ShorEcc>(qcOriginal, measureFrequency);
        EXPECT_ANY_THROW(mapper->apply());
    }
}

TEST_F(DDECCFunctionalityTest, testQ5LaflammeEcc) {
    int measureFrequency = 0;

    std::vector<std::shared_ptr<qc::QuantumComputation> (*)()> circuitsExpectToPass;
    circuitsExpectToPass.push_back(createIdentityCircuit);
    circuitsExpectToPass.push_back(createXCircuit);

    std::vector<std::shared_ptr<qc::QuantumComputation> (*)()> circuitsExpectToFail;
    circuitsExpectToFail.push_back(createYCircuit);
    circuitsExpectToFail.push_back(createHCircuit);
    circuitsExpectToFail.push_back(createHTCircuit);
    circuitsExpectToFail.push_back(createHZCircuit);
    circuitsExpectToFail.push_back(createCXCircuit);
    circuitsExpectToFail.push_back(createCZCircuit);
    circuitsExpectToFail.push_back(createCYCircuit);

    std::vector<dd::Qubit> dataQubits = {0, 1, 2, 3, 4};

    int insertErrorAfterNGates = 30;
    for (auto& circuit: circuitsExpectToPass) {
        auto qcOriginal = circuit();
        auto mapper     = std::make_unique<Q5LaflammeEcc>(qcOriginal, measureFrequency);
        auto qcECC      = mapper->apply();
        EXPECT_TRUE(verifyExecution(qcOriginal, qcECC, true, {}, insertErrorAfterNGates));
    }

    for (auto& circuit: circuitsExpectToFail) {
        auto qcOriginal = circuit();
        auto mapper     = std::make_unique<Q5LaflammeEcc>(qcOriginal, measureFrequency);
        EXPECT_ANY_THROW(mapper->apply());
    }
}

TEST_F(DDECCFunctionalityTest, testQ7Steane) {
    int measureFrequency = 0;

    std::vector<std::shared_ptr<qc::QuantumComputation> (*)()> circuitsExpectToPass;
    circuitsExpectToPass.push_back(createIdentityCircuit);
    circuitsExpectToPass.push_back(createXCircuit);
    circuitsExpectToPass.push_back(createYCircuit);
    circuitsExpectToPass.push_back(createHCircuit);
    circuitsExpectToPass.push_back(createHTCircuit);
    circuitsExpectToPass.push_back(createHZCircuit);
    circuitsExpectToPass.push_back(createCXCircuit);
    circuitsExpectToPass.push_back(createCZCircuit);
    circuitsExpectToPass.push_back(createCYCircuit);

    int                    circuitCounter = 0;
    std::vector<dd::Qubit> dataQubits     = {0, 1, 2, 3, 4, 5, 6};

    int insertErrorAfterNGates = 31;
    for (auto& circuit: circuitsExpectToPass) {
        circuitCounter++;
        std::cout << "Testing circuit " << circuitCounter << std::endl;
        auto qcOriginal = circuit();
        auto mapper     = std::make_unique<Q7SteaneEcc>(qcOriginal, measureFrequency);
        auto qcECC      = mapper->apply();
        EXPECT_TRUE(verifyExecution(qcOriginal, qcECC, true, dataQubits, insertErrorAfterNGates));
    }
}

TEST_F(DDECCFunctionalityTest, testQ9ShorEcc) {
    int measureFrequency = 0;

    std::vector<std::shared_ptr<qc::QuantumComputation> (*)()> circuitsExpectToPass;
    circuitsExpectToPass.push_back(createIdentityCircuit);
    circuitsExpectToPass.push_back(createXCircuit);
    circuitsExpectToPass.push_back(createCXCircuit);
    circuitsExpectToPass.push_back(createCYCircuit);

    std::vector<std::shared_ptr<qc::QuantumComputation> (*)()> circuitsExpectToFail;
    circuitsExpectToFail.push_back(createYCircuit);
    circuitsExpectToFail.push_back(createHCircuit);
    circuitsExpectToFail.push_back(createHTCircuit);
    circuitsExpectToFail.push_back(createHZCircuit);
    circuitsExpectToFail.push_back(createCZCircuit);

    int                    circuitCounter = 0;
    std::vector<dd::Qubit> dataQubits     = {0, 1, 2, 4, 5, 6, 7, 8};

    int insertErrorAfterNGates = 1;
    for (auto& circuit: circuitsExpectToPass) {
        circuitCounter++;
        std::cout << "Testing circuit " << circuitCounter << std::endl;
        auto qcOriginal = circuit();
        auto mapper     = std::make_unique<Q9ShorEcc>(qcOriginal, measureFrequency);
        auto qcECC      = mapper->apply();
        EXPECT_TRUE(verifyExecution(qcOriginal, qcECC, true, dataQubits, insertErrorAfterNGates));
    }

    for (auto& circuit: circuitsExpectToFail) {
        auto qcOriginal = circuit();
        auto mapper     = std::make_unique<Q9ShorEcc>(qcOriginal, measureFrequency);
        EXPECT_ANY_THROW(mapper->apply());
    }
}

TEST_F(DDECCFunctionalityTest, testQ9SurfaceEcc) {
    int measureFrequency = 0;

    std::vector<std::shared_ptr<qc::QuantumComputation> (*)()> circuitsExpectToPass;
    circuitsExpectToPass.push_back(createIdentityCircuit);
    circuitsExpectToPass.push_back(createXCircuit);
    circuitsExpectToPass.push_back(createCXCircuit);
    circuitsExpectToPass.push_back(createYCircuit);
    circuitsExpectToPass.push_back(createHCircuit);
    circuitsExpectToPass.push_back(createHZCircuit);
    circuitsExpectToPass.push_back(createCZCircuit);
    circuitsExpectToPass.push_back(createCYCircuit);

    std::vector<std::shared_ptr<qc::QuantumComputation> (*)()> circuitsExpectToFail;
    circuitsExpectToFail.push_back(createHTCircuit);

    int                    circuitCounter = 0;
    std::vector<dd::Qubit> dataQubits     = {0, 1, 2, 4, 5, 6, 7, 8};

    int insertErrorAfterNGates = 55;
    for (auto& circuit: circuitsExpectToPass) {
        circuitCounter++;
        std::cout << "Testing circuit " << circuitCounter << std::endl;
        auto qcOriginal = circuit();
        auto mapper     = std::make_unique<Q9SurfaceEcc>(qcOriginal, measureFrequency);
        auto qcECC      = mapper->apply();
        EXPECT_TRUE(verifyExecution(qcOriginal, qcECC, true, dataQubits, insertErrorAfterNGates));
    }

    for (auto& circuit: circuitsExpectToFail) {
        auto qcOriginal = circuit();
        auto mapper     = std::make_unique<Q9SurfaceEcc>(qcOriginal, measureFrequency);
        EXPECT_ANY_THROW(mapper->apply());
    }
}

TEST_F(DDECCFunctionalityTest, testQ18SurfaceEcc) {
    int measureFrequency = 0;

    std::vector<std::shared_ptr<qc::QuantumComputation> (*)()> circuitsExpectToPass;
    circuitsExpectToPass.push_back(createIdentityCircuit);
    circuitsExpectToPass.push_back(createXCircuit);
    circuitsExpectToPass.push_back(createYCircuit);
    circuitsExpectToPass.push_back(createHCircuit);
    circuitsExpectToPass.push_back(createHZCircuit);

    std::vector<std::shared_ptr<qc::QuantumComputation> (*)()> circuitsExpectToFail;
    circuitsExpectToFail.push_back(createHTCircuit);
    circuitsExpectToFail.push_back(createCXCircuit);
    circuitsExpectToFail.push_back(createCZCircuit);
    circuitsExpectToFail.push_back(createCYCircuit);

    int                    circuitCounter = 0;
    std::vector<dd::Qubit> dataQubits     = {0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17};

    int insertErrorAfterNGates = 127;
    for (auto& circuit: circuitsExpectToPass) {
        circuitCounter++;
        std::cout << "Testing circuit " << circuitCounter << std::endl;
        auto qcOriginal = circuit();
        auto mapper     = std::make_unique<Q18SurfaceEcc>(qcOriginal, measureFrequency);
        auto qcECC      = mapper->apply();
        EXPECT_TRUE(verifyExecution(qcOriginal, qcECC, true, dataQubits, insertErrorAfterNGates));
    }

    for (auto& circuit: circuitsExpectToFail) {
        auto qcOriginal = circuit();
        auto mapper     = std::make_unique<Q18SurfaceEcc>(qcOriginal, measureFrequency);
        EXPECT_ANY_THROW(mapper->apply());
    }
}
