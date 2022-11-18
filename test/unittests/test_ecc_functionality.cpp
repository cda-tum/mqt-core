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
    void SetUp() override {
    }

    void TearDown() override {
    }

    static void simulateCircuit(qc::QuantumComputation& qc, std::map<std::size_t, int>& finalClassicValues, bool simulateNoise = false, Qubit targetForNoise = 0, std::uint_fast32_t applyNoiseAfterNOperations = 0) {
        std::mt19937_64 mt;
        mt.seed(7);

        std::size_t opCounter = 0;

        std::uint_fast32_t operationCounter = 0;
        for (int sample = 0; sample < 50; sample++) {
            std::map<std::size_t, bool> classicValuesECC{};
            auto                        dd       = std::make_unique<dd::Package<>>(qc.getNqubits());
            vEdge                       rootEdge = dd->makeZeroState(qc.getNqubits());
            for (auto const& op: qc) {
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
                    auto* nu_op   = dynamic_cast<qc::NonUnitaryOperation*>(op.get());
                    auto  quantum = nu_op->getTargets();
                    for (signed char qubit: quantum) {
                        dd->incRef(rootEdge);
                        auto result = dd->measureOneCollapsing(rootEdge, qubit, false, mt);
                        if (result == '1') {
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
                        auto flipOperation = StandardOperation(qc.getNqubits(), targetForNoise, qc::X);
                        auto operation     = dd::getDD(&flipOperation, dd);
                        rootEdge           = dd->multiply(operation, rootEdge);
                    }
                }
                operationCounter++;
            }
            // Counting the final results
            for (std::size_t i = 0; i < classicValuesECC.size(); i++) {
                auto regName = qc.returnClassicalRegisterName(i);
                if (regName == "resultReg") {
                    if (finalClassicValues.count(i) == 0) {
                        finalClassicValues[i] = 0;
                    }
                    if (finalClassicValues[i]) {
                        finalClassicValues[i]++;
                    }
                }
            }
        }
    }

    static bool verifyExecution(qc::QuantumComputation& qcOriginal, qc::QuantumComputation& qcECC,
                                bool                          simulateWithErrors     = false,
                                const std::vector<dd::Qubit>& dataQubits             = {},
                                int                           insertErrorAfterNGates = 0) {
        std::map<std::size_t, int> finalClassicValuesOriginal{};
        bool                       testingSuccessful = true;

        simulateCircuit(qcOriginal, finalClassicValuesOriginal);

        if (!simulateWithErrors) {
            std::map<std::size_t, int> finalClassicValuesECC{};
            simulateCircuit(qcECC, finalClassicValuesECC);
            for (auto const& x: finalClassicValuesOriginal) {
                if (std::abs(x.second - finalClassicValuesECC[x.first]) > 3) {
                    testingSuccessful = false;
                }
            }
        } else {
            for (auto const& qubit: dataQubits) {
                std::map<std::size_t, int> finalClassicValuesECC{};
                simulateCircuit(qcECC, finalClassicValuesECC, true, qubit, insertErrorAfterNGates);
                for (auto const& x: finalClassicValuesOriginal) {
                    if (std::abs(x.second - finalClassicValuesECC[x.first]) > 3) {
                        std::cout << "Simulation failed when applying error to qubit " << static_cast<unsigned>(qubit) << " after " << insertErrorAfterNGates << " gates." << std::endl;
                        testingSuccessful = false;
                    }
                }
            }
        }
        return testingSuccessful;
    }

    static void createIdentityCircuit(qc::QuantumComputation& qc) {
        qc = {};
        qc.addQubitRegister(1U);
        qc.addClassicalRegister(1U, "resultReg");
        qc.x(0);
        qc.x(0);
        qc.measure(0, {"resultReg", 0});
    }

    static void createXCircuit(qc::QuantumComputation& qc) {
        qc = {};
        qc.addQubitRegister(1U);
        qc.addClassicalRegister(1U, "resultReg");
        qc.x(0);
        qc.measure(0, {"resultReg", 0});
    }

    static void createYCircuit(qc::QuantumComputation& qc) {
        qc = {};
        qc.addQubitRegister(1U);
        qc.addClassicalRegister(1U, "resultReg");
        qc.h(0);
        qc.y(0);
        qc.h(0);
        qc.measure(0, {"resultReg", 0});
    }

    static void createHCircuit(qc::QuantumComputation& qc) {
        qc = {};
        qc.addQubitRegister(1U);
        qc.addClassicalRegister(1U, "resultReg");
        qc.h(0);
        qc.measure(0, {"resultReg", 0});
    }

    static void createHTCircuit(qc::QuantumComputation& qc) {
        qc = {};
        qc.addQubitRegister(1U);
        qc.addClassicalRegister(1U, "resultReg");
        qc.h(0);
        qc.t(0);
        qc.tdag(0);
        qc.h(0);
        qc.measure(0, {"resultReg", 0});
    }

    static void createHZCircuit(qc::QuantumComputation& qc) {
        qc = {};
        qc.addQubitRegister(1U);
        qc.addClassicalRegister(1U, "resultReg");
        qc.h(0);
        qc.z(0);
        qc.h(0);
        qc.measure(0, {"resultReg", 0});
    }

    static void createCXCircuit(qc::QuantumComputation& qc) {
        qc = {};
        qc.addQubitRegister(2U);
        qc.addClassicalRegister(2U, "resultReg");
        qc.x(0);
        qc.x(1, 0_pc);
        qc.measure(0, {"resultReg", 0});
        qc.measure(1, {"resultReg", 1});
    }
};

TEST_F(DDECCFunctionalityTest, testQ3Shor) {
    bool decomposeMC      = false;
    bool cliffOnly        = false;
    int  measureFrequency = 0;

    void (*circuitsExpectToPass[3])(qc::QuantumComputation & qc) = {createIdentityCircuit, createXCircuit, createCXCircuit};
    void (*circuitsExpectToFail[4])(qc::QuantumComputation & qc) = {createYCircuit, createHCircuit, createHTCircuit, createHZCircuit};

    int                    circuitCounter = 0;
    std::vector<dd::Qubit> dataQubits     = {0, 1, 2};

    int insertErrorAfterNGates = 1;
    for (auto& circuit: circuitsExpectToPass) {
        std::cout << "Testing circuit " << ++circuitCounter << std::endl;
        qc::QuantumComputation qcOriginal{};
        circuit(qcOriginal);
        Ecc*                    mapper = new Q3ShorEcc(qcOriginal, measureFrequency, decomposeMC, cliffOnly);
        qc::QuantumComputation& qcECC  = mapper->apply();
        EXPECT_TRUE(verifyExecution(qcOriginal, qcECC, true, dataQubits, insertErrorAfterNGates));
    }

    for (auto& circuit: circuitsExpectToFail) {
        qc::QuantumComputation qcOriginal{};
        circuit(qcOriginal);
        Ecc* mapper = new Q3ShorEcc(qcOriginal, measureFrequency, decomposeMC, cliffOnly);
        EXPECT_ANY_THROW(mapper->apply());
    }
}

TEST_F(DDECCFunctionalityTest, testQ5LaflammeEcc) {
    bool decomposeMC      = false;
    bool cliffOnly        = false;
    int  measureFrequency = 0;

    void (*circuitsExpectToPass[2])(qc::QuantumComputation & qc) = {createIdentityCircuit, createXCircuit};
    void (*circuitsExpectToFail[5])(qc::QuantumComputation & qc) = {createYCircuit, createHCircuit, createHTCircuit, createHZCircuit, createCXCircuit};

    std::vector<dd::Qubit> dataQubits = {0, 1, 2, 3, 4};

    int insertErrorAfterNGates = 30;
    for (auto& circuit: circuitsExpectToPass) {
        qc::QuantumComputation qcOriginal{};
        circuit(qcOriginal);
        Ecc*                    mapper = new Q5LaflammeEcc(qcOriginal, measureFrequency, decomposeMC, cliffOnly);
        qc::QuantumComputation& qcECC  = mapper->apply();

        //        std::stringstream ss{};
        //        qcECC.dumpOpenQASM(ss);
        //        std::cout << ss.str() << std::endl;

        EXPECT_TRUE(verifyExecution(qcOriginal, qcECC, true, {}, insertErrorAfterNGates));
    }

    for (auto& circuit: circuitsExpectToFail) {
        qc::QuantumComputation qcOriginal{};
        circuit(qcOriginal);
        Ecc* mapper = new Q5LaflammeEcc(qcOriginal, measureFrequency, decomposeMC, cliffOnly);
        EXPECT_ANY_THROW(mapper->apply());
    }
}

TEST_F(DDECCFunctionalityTest, testQ7Steane) {
    bool decomposeMC      = false;
    bool cliffOnly        = false;
    int  measureFrequency = 0;

    void (*circuitsExpectToPass[7])(qc::QuantumComputation & qc) = {
            createIdentityCircuit,
            createXCircuit,
            createYCircuit,
            createHCircuit,
            createHTCircuit,
            createHZCircuit,
            createCXCircuit,
    };

    int                    circuitCounter = 0;
    std::vector<dd::Qubit> dataQubits     = {0, 1, 2, 3, 4, 5, 6};

    int insertErrorAfterNGates = 30;
    for (auto& circuit: circuitsExpectToPass) {
        std::cout << "Testing circuit " << ++circuitCounter << std::endl;
        qc::QuantumComputation qcOriginal{};
        circuit(qcOriginal);
        Ecc*                    mapper = new Q7SteaneEcc(qcOriginal, measureFrequency, decomposeMC, cliffOnly);
        qc::QuantumComputation& qcECC  = mapper->apply();
        EXPECT_TRUE(verifyExecution(qcOriginal, qcECC, true, dataQubits, insertErrorAfterNGates));
    }
}

TEST_F(DDECCFunctionalityTest, testQ9ShorEcc) {
    bool decomposeMC      = false;
    bool cliffOnly        = false;
    int  measureFrequency = 0;

    void (*circuitsExpectToPass[3])(qc::QuantumComputation & qc) = {createIdentityCircuit, createXCircuit, createCXCircuit};
    void (*circuitsExpectToFail[4])(qc::QuantumComputation & qc) = {createYCircuit, createHCircuit, createHTCircuit, createHZCircuit};

    int                    circuitCounter = 0;
    std::vector<dd::Qubit> dataQubits     = {0, 1, 2, 4, 5, 6, 7, 8};

    int insertErrorAfterNGates = 1;
    for (auto& circuit: circuitsExpectToPass) {
        std::cout << "Testing circuit " << ++circuitCounter << std::endl;
        qc::QuantumComputation qcOriginal{};
        circuit(qcOriginal);
        Ecc*                    mapper = new Q9ShorEcc(qcOriginal, measureFrequency, decomposeMC, cliffOnly);
        qc::QuantumComputation& qcECC  = mapper->apply();
        EXPECT_TRUE(verifyExecution(qcOriginal, qcECC, true, dataQubits, insertErrorAfterNGates));
    }

    for (auto& circuit: circuitsExpectToFail) {
        qc::QuantumComputation qcOriginal{};
        circuit(qcOriginal);
        Ecc* mapper = new Q9ShorEcc(qcOriginal, measureFrequency, decomposeMC, cliffOnly);
        EXPECT_ANY_THROW(mapper->apply());
    }
}

TEST_F(DDECCFunctionalityTest, testQ9SurfaceEcc) {
    bool decomposeMC      = false;
    bool cliffOnly        = false;
    int  measureFrequency = 0;

    //    qc::QuantumComputation qcOriginal{};
    //    createSpecial(qcOriginal);
    //    EXPECT_TRUE(verifyExecution(qcOriginal, qcOriginal));

    void (*circuitsExpectToPass[2])(qc::QuantumComputation & qc) = {createIdentityCircuit, createXCircuit};
    void (*circuitsExpectToFail[5])(qc::QuantumComputation & qc) = {createYCircuit, createHTCircuit, createHCircuit, createHZCircuit, createCXCircuit}; //todo @Christoph the circuit createCXCircuit should fail according to the github readme

    int                    circuitCounter = 0;
    std::vector<dd::Qubit> dataQubits     = {0, 1, 2, 4, 5, 6, 7, 8};

    int insertErrorAfterNGates = 55;
    for (auto& circuit: circuitsExpectToPass) {
        std::cout << "Testing circuit " << ++circuitCounter << std::endl;
        qc::QuantumComputation qcOriginal{};
        circuit(qcOriginal);
        Ecc*                    mapper = new Q9SurfaceEcc(qcOriginal, measureFrequency, decomposeMC, cliffOnly);
        qc::QuantumComputation& qcECC  = mapper->apply();
        EXPECT_TRUE(verifyExecution(qcOriginal, qcECC, true, dataQubits, insertErrorAfterNGates));
    }

    for (auto& circuit: circuitsExpectToFail) {
        qc::QuantumComputation qcOriginal{};
        circuit(qcOriginal);
        Ecc* mapper = new Q9SurfaceEcc(qcOriginal, measureFrequency, decomposeMC, cliffOnly);
        EXPECT_ANY_THROW(mapper->apply());
    }
}

TEST_F(DDECCFunctionalityTest, testQ18SurfaceEcc) {
    bool decomposeMC      = false;
    bool cliffOnly        = false;
    int  measureFrequency = 0;

    void (*circuitsExpectToPass[5])(qc::QuantumComputation & qc) = {createIdentityCircuit, createXCircuit, createYCircuit, createHCircuit, createHZCircuit};
    void (*circuitsExpectToFail[2])(qc::QuantumComputation & qc) = {createHTCircuit, createCXCircuit};

    int                    circuitCounter = 0;
    std::vector<dd::Qubit> dataQubits     = {0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17};

    int insertErrorAfterNGates = 127;
    for (auto& circuit: circuitsExpectToPass) {
        std::cout << "Testing circuit " << ++circuitCounter << std::endl;
        qc::QuantumComputation qcOriginal{};
        circuit(qcOriginal);
        Ecc*                    mapper = new Q18SurfaceEcc(qcOriginal, measureFrequency, decomposeMC, cliffOnly);
        qc::QuantumComputation& qcECC  = mapper->apply();
        EXPECT_TRUE(verifyExecution(qcOriginal, qcECC, true, dataQubits, insertErrorAfterNGates));
    }

    for (auto& circuit: circuitsExpectToFail) {
        qc::QuantumComputation qcOriginal{};
        circuit(qcOriginal);
        Ecc* mapper = new Q18SurfaceEcc(qcOriginal, measureFrequency, decomposeMC, cliffOnly);
        EXPECT_ANY_THROW(mapper->apply());
    }
}
