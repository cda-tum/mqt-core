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

    static void simulateCircuit(qc::QuantumComputation& qc, std::map<std::size_t, bool>& finalClassicValues, bool simulateNoise = false, Qubit targetForNoise = 0, std::uint_fast32_t applyNoiseAfterNOperations = 0) {
        std::mt19937_64 mt;
        mt.seed(1);

        std::map<std::size_t, bool> classicValuesECC;

        auto ddECC = std::make_unique<dd::Package<>>(qc.getNqubits());

        vEdge rootEdgeECC = ddECC->makeZeroState(qc.getNqubits());

        std::uint_fast32_t operationCounter = 0;

        for (auto const& op: qc) {
            if (op->getType() == qc::Measure) {
                auto* nu_op   = dynamic_cast<qc::NonUnitaryOperation*>(op.get());
                auto  quantum = nu_op->getTargets();
                auto  classic = nu_op->getClassics();
                for (unsigned int i = 0; i < quantum.size(); ++i) {
                    ddECC->incRef(rootEdgeECC);
                    auto result = ddECC->measureOneCollapsing(rootEdgeECC, quantum.at(i), false, mt);
                    assert(result == '0' || result == '1');
                    classicValuesECC[classic.at(i)] = (result == '1');
                    auto regName                    = qc.returnClassicalRegisterName(quantum.at(i));
                    if (regName == "resultReg") {
                        finalClassicValues[classic.at(i)] = (result == '1');
                    }
                }
            } else if (op->getType() == qc::Reset) {
                auto* nu_op   = dynamic_cast<qc::NonUnitaryOperation*>(op.get());
                auto  quantum = nu_op->getTargets();
                for (unsigned int i = 0; i < quantum.size(); ++i) {
                    ddECC->incRef(rootEdgeECC);
                    auto result = ddECC->measureOneCollapsing(rootEdgeECC, quantum.at(i), false, mt);
                    if (result == 1) {
                        auto flipOperation = StandardOperation(nu_op->getNqubits(), i, qc::X);
                        auto operation     = dd::getDD(&flipOperation, ddECC);
                        rootEdgeECC        = ddECC->multiply(operation, rootEdgeECC);
                    }
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

                    //std::clog << "expected " << expected_value << " and actual value was " << actual_value << "\n";

                    if (actual_value != expected_value) {
                        continue;
                    }
                }
                auto operation = dd::getDD(op.get(), ddECC);
                rootEdgeECC    = ddECC->multiply(operation, rootEdgeECC);
            }
            if (simulateNoise && operationCounter == applyNoiseAfterNOperations) {
                auto operation = ddECC->makeGateDD(Xmat, qc.getNqubits(), targetForNoise);
                rootEdgeECC    = ddECC->multiply(operation, rootEdgeECC);
            }
            operationCounter++;
        }
    }

    static bool verifyExecution(qc::QuantumComputation& qcOriginal, qc::QuantumComputation& qcECC, bool simulateWithErrors = true) {
        std::map<std::size_t, bool> finalClassicValuesOriginal{};
        bool                        testingSuccessful = true;

        simulateCircuit(qcOriginal, finalClassicValuesOriginal);

        if (!simulateWithErrors) {
            std::map<std::size_t, bool> finalClassicValuesECC{};
            simulateCircuit(qcECC, finalClassicValuesECC);
            for (auto const& x: finalClassicValuesOriginal) {
                if (x.second != finalClassicValuesECC[x.first]) {
                    testingSuccessful = false;
                }
            }
        } else {
            auto nrQubits    = qcECC.getNqubits();
            auto nrOperation = 50;
            for (auto i = 0; i < nrQubits; i++) {
                std::map<std::size_t, bool> finalClassicValuesECC{};
                simulateCircuit(qcECC, finalClassicValuesECC, true, i, nrOperation);
                for (auto const& x: finalClassicValuesOriginal) {
                    if (x.second != finalClassicValuesECC[x.first]) {
                        std::cout << "Simulation failed when applying error to qubit " << i << " after " << nrOperation << " gates." << std::endl;
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
        qc.addQubitRegister(2U);
        qc.addClassicalRegister(2U, "resultReg");
        qc.h(0);
        qc.y(0);
        qc.h(0);
        qc.y(1);
        qc.measure(0, {"resultReg", 0});
        qc.measure(1, {"resultReg", 1});
    }

    static void createHCircuit(qc::QuantumComputation& qc) {
        qc = {};
        qc.addQubitRegister(1U);
        qc.h(0);
    }

    static void createHTCircuit(qc::QuantumComputation& qc) {
        qc = {};
        qc.addQubitRegister(2U);
        qc.h(0);
        qc.t(0);
        qc.h(1);
        qc.t(1);
        qc.tdag(1);
        qc.h(1);
    }

    static void createHZCircuit(qc::QuantumComputation& qc) {
        qc = {};
        qc.addQubitRegister(2U);
        qc.h(0);
        qc.z(0);
        qc.h(0);
        qc.h(0);
    }

    static void createCXCircuit(qc::QuantumComputation& qc) {
        qc = {};
        qc.addQubitRegister(2U);
        qc.x(0);
        qc.x(1, 0_pc);
    }
};

TEST_F(DDECCFunctionalityTest, testQ3Shor) {
    bool decomposeMC      = false;
    bool cliffOnly        = false;
    int  measureFrequency = 1;

    void (*circuitsExpectToPass[3])(qc::QuantumComputation & qc) = {createIdentityCircuit, createXCircuit, createCXCircuit};
    void (*circuitsExpectToFail[4])(qc::QuantumComputation & qc) = {createYCircuit, createHCircuit, createHTCircuit, createHZCircuit};

    for (auto& circuit: circuitsExpectToPass) {
        qc::QuantumComputation qcOriginal{};
        circuit(qcOriginal);
        Ecc*                    mapper = new Q3ShorEcc(qcOriginal, measureFrequency, decomposeMC, cliffOnly);
        qc::QuantumComputation& qcECC  = mapper->apply();
        EXPECT_TRUE(verifyExecution(qcOriginal, qcECC));
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
    int  measureFrequency = 1;

    void (*circuitsExpectToPass[2])(qc::QuantumComputation & qc) = {createIdentityCircuit, createXCircuit};
    void (*circuitsExpectToFail[5])(qc::QuantumComputation & qc) = {createYCircuit, createHCircuit, createHTCircuit, createHZCircuit, createCXCircuit};

    for (auto& circuit: circuitsExpectToPass) {
        qc::QuantumComputation qcOriginal{};
        circuit(qcOriginal);
        Ecc*                    mapper = new Q5LaflammeEcc(qcOriginal, measureFrequency, decomposeMC, cliffOnly);
        qc::QuantumComputation& qcECC  = mapper->apply();
        EXPECT_TRUE(verifyExecution(qcOriginal, qcECC));
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
    int  measureFrequency = 1;

    void (*circuitsExpectToPass[7])(qc::QuantumComputation & qc) = {
            createIdentityCircuit,
            createXCircuit,
            createYCircuit,
            createHCircuit,
            createHTCircuit,
            createHZCircuit,
            createCXCircuit
    };

    for (auto& circuit: circuitsExpectToPass) {
        qc::QuantumComputation qcOriginal{};
        circuit(qcOriginal);
        Ecc*                    mapper = new Q7SteaneEcc(qcOriginal, measureFrequency, decomposeMC, cliffOnly);
        qc::QuantumComputation& qcECC  = mapper->apply();
        EXPECT_TRUE(verifyExecution(qcOriginal, qcECC));
    }
}

TEST_F(DDECCFunctionalityTest, testQ9ShorEcc) {
    bool decomposeMC      = false;
    bool cliffOnly        = false;
    int  measureFrequency = 1;

    void (*circuitsExpectToPass[3])(qc::QuantumComputation & qc) = {createIdentityCircuit, createXCircuit, createCXCircuit};
    void (*circuitsExpectToFail[4])(qc::QuantumComputation & qc) = {createYCircuit, createHCircuit, createHTCircuit, createHZCircuit};

    for (auto& circuit: circuitsExpectToPass) {
        qc::QuantumComputation qcOriginal{};
        circuit(qcOriginal);
        Ecc*                    mapper = new Q9ShorEcc(qcOriginal, measureFrequency, decomposeMC, cliffOnly);
        qc::QuantumComputation& qcECC  = mapper->apply();
        EXPECT_TRUE(verifyExecution(qcOriginal, qcECC));
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
    int  measureFrequency = 1;

    void (*circuitsExpectToPass[3])(qc::QuantumComputation & qc) = {createIdentityCircuit, createXCircuit, createCXCircuit}; //todo @Christoph the circuit createCXCircuit should fail according to the github readme
    void (*circuitsExpectToFail[4])(qc::QuantumComputation & qc) = {createYCircuit, createHTCircuit, createHCircuit, createHZCircuit};
    for (auto& circuit: circuitsExpectToPass) {
        qc::QuantumComputation qcOriginal{};
        circuit(qcOriginal);
        Ecc*                    mapper = new Q9SurfaceEcc(qcOriginal, measureFrequency, decomposeMC, cliffOnly);
        qc::QuantumComputation& qcECC  = mapper->apply();
        EXPECT_TRUE(verifyExecution(qcOriginal, qcECC));
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
    int  measureFrequency = 1;

    void (*circuitsExpectToPass[5])(qc::QuantumComputation & qc) = {createIdentityCircuit, createXCircuit, createYCircuit, createHCircuit, createHZCircuit};
    void (*circuitsExpectToFail[2])(qc::QuantumComputation & qc) = {createHTCircuit, createCXCircuit};

    for (auto& circuit: circuitsExpectToPass) {
        qc::QuantumComputation qcOriginal{};
        circuit(qcOriginal);
        Ecc*                    mapper = new Q18SurfaceEcc(qcOriginal, measureFrequency, decomposeMC, cliffOnly);
        qc::QuantumComputation& qcECC  = mapper->apply();
        EXPECT_TRUE(verifyExecution(qcOriginal, qcECC));
    }

    for (auto& circuit: circuitsExpectToFail) {
        qc::QuantumComputation qcOriginal{};
        circuit(qcOriginal);
        Ecc* mapper = new Q18SurfaceEcc(qcOriginal, measureFrequency, decomposeMC, cliffOnly);
        EXPECT_ANY_THROW(mapper->apply());
    }
}
