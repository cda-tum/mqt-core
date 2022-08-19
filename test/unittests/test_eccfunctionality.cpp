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

class DDNoiseFunctionalityTest: public ::testing::Test {
protected:
    void SetUp() override {
        // circuit taken from https://github.com/pnnl/qasmbench
        qc.addQubitRegister(4U);
        qc.x(0);
        qc.x(1);
        qc.h(3);
        qc.x(3, 2_pc);
        qc.t(0);
        qc.t(1);
        qc.t(2);
        qc.tdag(3);
        qc.x(1, 0_pc);
        qc.x(3, 2_pc);
        qc.x(0, 3_pc);
        qc.x(2, 1_pc);
        qc.x(1, 0_pc);
        qc.x(3, 2_pc);
        qc.tdag(0);
        qc.tdag(1);
        qc.tdag(2);
        qc.t(3);
        qc.x(1, 0_pc);
        qc.x(3, 2_pc);
        qc.s(3);
        qc.x(0, 3_pc);
        qc.h(3);
    }

    void TearDown() override {
    }

    qc::QuantumComputation qc{};
};

TEST_F(DDNoiseFunctionalityTest, StochSimulateAdder4IdentiyError) {
    Ecc* mapper           = nullptr;
    bool decomposeMC      = false;
    bool cliffOnly        = false;
    int  measureFrequency = 1;

    mapper = new IdEcc(qc, measureFrequency, decomposeMC, cliffOnly);

    mapper->apply();

    for (auto const& op: qc) {
        std::cout << op->getName() << std::endl;
    }
}
