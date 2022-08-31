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

    bool verifyExecution(qc::QuantumComputation& qcOriginal, qc::QuantumComputation& qcECC) {
        std::mt19937_64 mt;
        mt.seed(1);

        std::map<std::size_t, bool> classicValuesECC;
        std::map<std::size_t, bool> classicValuesOriginal;

        auto ddECC      = std::make_unique<dd::Package<>>(qcECC.getNqubits());
        auto ddOriginal = std::make_unique<dd::Package<>>(qcOriginal.getNqubits());

        vEdge rootEdgeECC      = ddECC->makeZeroState(qcECC.getNqubits());
        vEdge rootEdgeOriginal = ddOriginal->makeZeroState(qcOriginal.getNqubits());

        for (auto const& op: qcECC) {
            if (op->getType() == qc::Measure) {
                auto* nu_op   = dynamic_cast<qc::NonUnitaryOperation*>(op.get());
                auto  quantum = nu_op->getTargets();
                auto  classic = nu_op->getClassics();
                for (unsigned int i = 0; i < quantum.size(); ++i) {
                    ddECC->incRef(rootEdgeECC);
                    auto result = ddECC->measureOneCollapsing(rootEdgeECC, quantum.at(i), false, mt);
                    assert(result == '0' || result == '1');
                    classicValuesECC[classic.at(i)] = (result == '1');
                }
            } else if (op->getType() == qc::Reset) {
                auto* nu_op   = dynamic_cast<qc::NonUnitaryOperation*>(op.get());
                auto  quantum = nu_op->getTargets();
                for (unsigned int i = 0; i < quantum.size(); ++i) {
                    ddECC->incRef(rootEdgeECC);
                    auto result = ddECC->measureOneCollapsing(rootEdgeECC, quantum.at(i), false, mt);
                    if (result == 1) {
                        auto flipOperation = StandardOperation(nu_op->getNqubits(), i, qc::X);
                        auto operation = dd::getDD(&flipOperation, ddECC);
                        rootEdgeECC    = ddECC->multiply(operation, rootEdgeECC);
                    }
                }
            } else {
                auto operation = dd::getDD(op.get(), ddECC);
                rootEdgeECC    = ddECC->multiply(operation, rootEdgeECC);
            }
        }

        for (auto const& op: qcOriginal) {
            if (op->getType() == qc::Measure) {
                auto* nu_op   = dynamic_cast<qc::NonUnitaryOperation*>(op.get());
                auto  quantum = nu_op->getTargets();
                auto  classic = nu_op->getClassics();
                for (unsigned int i = 0; i < quantum.size(); ++i) {
                    ddECC->incRef(rootEdgeOriginal);
                    auto result = ddOriginal->measureOneCollapsing(rootEdgeOriginal, quantum.at(i), false, mt);
                    assert(result == '0' || result == '1');
                    classicValuesOriginal[classic.at(i)] = (result == '1');
                }
            } else {
                auto operation   = dd::getDD(op.get(), ddOriginal);
                rootEdgeOriginal = ddOriginal->multiply(operation, rootEdgeOriginal);
            }
        }

        for (auto const& x: classicValuesOriginal) {
            if (x.second != classicValuesECC[x.first]) return false;
            std::cout << "first: " << x.first << " second: " << x.second << std::endl;
            std::cout << "first: "
                      << " second: " << classicValuesECC[x.first] << std::endl;
        }

        return true;
    }

    static void createXCircuit(qc::QuantumComputation& qc) {
        qc = {};
        qc.addQubitRegister(1U);
        qc.addClassicalRegister(1U);
        qc.x(0);
        qc.measure(0, 0);
    }

    void createYCircuit(qc::QuantumComputation &qc) {
        qc = {};
        qc.addQubitRegister(2U);
        qc.addClassicalRegister(2U);
        qc.addClassicalRegister(2U, "finalMeasure");
        qc.addClassicalRegister(2U, "testReg");
        qc.h(0);
        qc.y(0);
        qc.h(0);
        qc.y(1);
        qc.measure(0, {"finalMeasure", 0});
        qc.measure(1, 1);
    }

    //void createHCircuit(qc::QuantumComputation &qc) {
    //    qc = {};
    //    qc.addQubitRegister(1U);
    //    qc.h(0);
    //}
    //
    //void createCXCircuit(qc::QuantumComputation &qc) {
    //    qc = {};
    //    qc.addQubitRegister(2U);
    //    qc.x(0);
    //    qc.x(1, 0_pc);
    //}
    //
    //void createHTCircuit(qc::QuantumComputation &qc) {
    //    qc = {};
    //    qc.addQubitRegister(2U);
    //    qc.h(0);
    //    qc.t(0);
    //    qc.h(1);
    //    qc.t(1);
    //    qc.tdag(1);
    //    qc.h(1);
    //}
    //
    //void createHZCircuit(qc::QuantumComputation &qc) {
    //    qc = {};
    //    qc.addQubitRegister(2U);
    //    qc.h(0);
    //    qc.z(0);
    //    qc.h(0);
    //    qc.h(0);
    //}
};

TEST_F(DDECCFunctionalityTest, StochSimulateAdder4IdentiyError) {
    bool decomposeMC      = false;
    bool cliffOnly        = false;
    int  measureFrequency = 1;

//    {
//        qc::QuantumComputation qcOriginal{};
//        createXCircuit(qcOriginal);
//        Ecc*                    mapper = new Q7SteaneEcc(qcOriginal, measureFrequency, decomposeMC, cliffOnly);
//        qc::QuantumComputation& qcECC  = mapper->apply();
//        EXPECT_TRUE(verifyExecution(qcOriginal, qcECC));
//    }

    {
        qc::QuantumComputation qcOriginal{};
        createYCircuit(qcOriginal);
        Ecc*                    mapper = new Q7SteaneEcc(qcOriginal, measureFrequency, decomposeMC, cliffOnly);
        qc::QuantumComputation& qcECC  = mapper->apply();
        EXPECT_TRUE(verifyExecution(qcOriginal, qcECC));
    }

    printf("\n");
}
