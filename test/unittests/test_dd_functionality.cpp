/*
 * This file is part of MQT QFR library which is released under the MIT license.
 * See file README.md or go to https://www.cda.cit.tum.de/research/quantum/ for more information.
 */

#include "CircuitOptimizer.hpp"
#include "QuantumComputation.hpp"
#include "dd/FunctionalityConstruction.hpp"
#include "dd/Simulation.hpp"

#include "gtest/gtest.h"
#include <random>

using namespace qc;

class DDFunctionality: public testing::TestWithParam<qc::OpType> {
protected:
    void TearDown() override {
        if (!e.isTerminal()) {
            dd->decRef(e);
        }
        dd->garbageCollect(true);

        // number of complex table entries after clean-up should equal initial number of entries
        EXPECT_EQ(dd->cn.complexTable.getCount(), initialComplexCount);
        // number of available cache entries after clean-up should equal initial number of entries
        EXPECT_EQ(dd->cn.complexCache.getCount(), initialCacheCount);
    }

    void SetUp() override {
        // dd
        dd                  = std::make_unique<dd::Package<>>(nqubits);
        initialCacheCount   = dd->cn.complexCache.getCount();
        initialComplexCount = dd->cn.complexTable.getCount();

        // initial state preparation
        e = ident = dd->makeIdent(nqubits);
        dd->incRef(ident);

        std::array<std::mt19937_64::result_type, std::mt19937_64::state_size> randomData{};
        std::random_device                                                    rd;
        std::generate(begin(randomData), end(randomData), [&]() { return rd(); });
        std::seed_seq seeds(begin(randomData), end(randomData));
        mt.seed(seeds);
        dist = std::uniform_real_distribution<dd::fp>(0.0, 2. * dd::PI);
    }

    dd::QubitCount                         nqubits             = 4U;
    std::size_t                            initialCacheCount   = 0U;
    std::size_t                            initialComplexCount = 0U;
    qc::MatrixDD                           e{}, ident{};
    std::unique_ptr<dd::Package<>>         dd;
    std::mt19937_64                        mt;
    std::uniform_real_distribution<dd::fp> dist;
};

INSTANTIATE_TEST_SUITE_P(Parameters,
                         DDFunctionality,
                         testing::Values(qc::I, qc::H, qc::X, qc::Y, qc::Z, qc::S, qc::Sdag, qc::T, qc::Tdag, qc::SX, qc::SXdag, qc::V,
                                         qc::Vdag, qc::U3, qc::U2, qc::Phase, qc::RX, qc::RY, qc::RZ, qc::Peres, qc::Peresdag,
                                         qc::SWAP, qc::iSWAP),
                         [](const testing::TestParamInfo<DDFunctionality::ParamType>& info) {
                             auto gate = (qc::OpType)info.param;
                             switch (gate) {
                                 case qc::I: return "i";
                                 case qc::H: return "h";
                                 case qc::X: return "x";
                                 case qc::Y: return "y";
                                 case qc::Z: return "z";
                                 case qc::S: return "s";
                                 case qc::Sdag: return "sdg";
                                 case qc::T: return "t";
                                 case qc::Tdag: return "tdg";
                                 case qc::SX: return "sx";
                                 case qc::SXdag: return "sxdg";
                                 case qc::V: return "v";
                                 case qc::Vdag: return "vdg";
                                 case qc::U3: return "u3";
                                 case qc::U2: return "u2";
                                 case qc::Phase: return "u1";
                                 case qc::RX: return "rx";
                                 case qc::RY: return "ry";
                                 case qc::RZ: return "rz";
                                 case qc::SWAP: return "swap";
                                 case qc::iSWAP: return "iswap";
                                 case qc::Peres: return "p";
                                 case qc::Peresdag: return "pdag";
                                 default: return "unknownGate";
                             }
                         });

TEST_P(DDFunctionality, standard_op_build_inverse_build) {
    using namespace qc::literals;
    auto gate = static_cast<qc::OpType>(GetParam());

    qc::StandardOperation op;
    switch (gate) {
        case qc::U3:
            op = qc::StandardOperation(nqubits, 0, gate, dist(mt), dist(mt), dist(mt));
            break;
        case qc::U2:
            op = qc::StandardOperation(nqubits, 0, gate, dist(mt), dist(mt));
            break;
        case qc::RX:
        case qc::RY:
        case qc::RZ:
        case qc::Phase:
            op = qc::StandardOperation(nqubits, 0, gate, dist(mt));
            break;

        case qc::SWAP:
        case qc::iSWAP:
            op = qc::StandardOperation(nqubits, Controls{}, 0, 1, gate);
            break;
        case qc::Peres:
        case qc::Peresdag:
            op = qc::StandardOperation(nqubits, {0_pc}, 1, 2, gate);
            break;
        default:
            op = qc::StandardOperation(nqubits, 0, gate);
    }

    ASSERT_NO_THROW({ e = dd->multiply(getDD(&op, dd), getInverseDD(&op, dd)); });
    dd->incRef(e);

    EXPECT_EQ(ident, e);
}

TEST_F(DDFunctionality, build_circuit) {
    qc::QuantumComputation qc(nqubits);

    qc.emplace_back<qc::StandardOperation>(nqubits, 0, qc::X);
    qc.emplace_back<qc::StandardOperation>(nqubits, std::vector<Qubit>{0, 1}, qc::SWAP);
    qc.emplace_back<qc::StandardOperation>(nqubits, 0, qc::H);
    qc.emplace_back<qc::StandardOperation>(nqubits, 3, qc::S);
    qc.emplace_back<qc::StandardOperation>(nqubits, 2, qc::Sdag);
    qc.emplace_back<qc::StandardOperation>(nqubits, 0, qc::V);
    qc.emplace_back<qc::StandardOperation>(nqubits, 1, qc::T);
    qc.emplace_back<qc::StandardOperation>(nqubits, 0_pc, 1, qc::X);
    qc.emplace_back<qc::StandardOperation>(nqubits, 3_pc, 2, qc::X);
    qc.emplace_back<qc::StandardOperation>(nqubits, Controls{2_pc, 3_pc}, 0, qc::X);

    qc.emplace_back<qc::StandardOperation>(nqubits, Controls{2_pc, 3_pc}, 0, qc::X);
    qc.emplace_back<qc::StandardOperation>(nqubits, 3_pc, 2, qc::X);
    qc.emplace_back<qc::StandardOperation>(nqubits, 0_pc, 1, qc::X);
    qc.emplace_back<qc::StandardOperation>(nqubits, 1, qc::Tdag);
    qc.emplace_back<qc::StandardOperation>(nqubits, 0, qc::Vdag);
    qc.emplace_back<qc::StandardOperation>(nqubits, 2, qc::S);
    qc.emplace_back<qc::StandardOperation>(nqubits, 3, qc::Sdag);
    qc.emplace_back<qc::StandardOperation>(nqubits, 0, qc::H);
    qc.emplace_back<qc::StandardOperation>(nqubits, std::vector<Qubit>{0, 1}, qc::SWAP);
    qc.emplace_back<qc::StandardOperation>(nqubits, 0, qc::X);

    e = dd->multiply(buildFunctionality(&qc, dd), e);

    EXPECT_EQ(ident, e);

    qc.emplace_back<qc::StandardOperation>(nqubits, 0, qc::X);
    e = dd->multiply(buildFunctionality(&qc, dd), e);
    dd->incRef(e);

    EXPECT_NE(ident, e);
}

TEST_F(DDFunctionality, non_unitary) {
    const qc::QuantumComputation qc{};
    auto                         dummyMap = Permutation{};
    auto                         op       = qc::NonUnitaryOperation(nqubits, {0, 1, 2, 3}, {0, 1, 2, 3});
    EXPECT_FALSE(op.isUnitary());
    EXPECT_THROW(getDD(&op, dd), qc::QFRException);
    EXPECT_THROW(getInverseDD(&op, dd), qc::QFRException);
    EXPECT_THROW(getDD(&op, dd, dummyMap), qc::QFRException);
    EXPECT_THROW(getInverseDD(&op, dd, dummyMap), qc::QFRException);
    for (Qubit i = 0; i < nqubits; ++i) {
        EXPECT_TRUE(op.actsOn(i));
    }

    for (Qubit i = 0; i < nqubits; ++i) {
        dummyMap[i] = i;
    }
    auto barrier = qc::NonUnitaryOperation(nqubits, {0, 1, 2, 3}, qc::OpType::Barrier);
    EXPECT_EQ(getDD(&barrier, dd), dd->makeIdent(nqubits));
    EXPECT_EQ(getInverseDD(&barrier, dd), dd->makeIdent(nqubits));
    EXPECT_EQ(getDD(&barrier, dd, dummyMap), dd->makeIdent(nqubits));
    EXPECT_EQ(getInverseDD(&barrier, dd, dummyMap), dd->makeIdent(nqubits));
    for (Qubit i = 0; i < nqubits; ++i) {
        EXPECT_FALSE(barrier.actsOn(i));
    }
}

TEST_F(DDFunctionality, CircuitEquivalence) {
    // verify that the IBM decomposition of the H gate into RZ-SX-RZ works as expected (i.e., realizes H up to a global phase)
    qc::QuantumComputation qc1(1);
    qc1.h(0);

    qc::QuantumComputation qc2(1);
    qc2.rz(0, PI_2);
    qc2.sx(0);
    qc2.rz(0, PI_2);

    const qc::MatrixDD dd1 = buildFunctionality(&qc1, dd);
    const qc::MatrixDD dd2 = buildFunctionality(&qc2, dd);

    EXPECT_EQ(dd1.p, dd2.p);
}

TEST_F(DDFunctionality, changePermutation) {
    qc::QuantumComputation qc{};
    std::stringstream      ss{};
    ss << "// o 1 0\n"
       << "OPENQASM 2.0;"
       << "include \"qelib1.inc\";"
       << "qreg q[2];"
       << "x q[0];"
       << std::endl;
    qc.import(ss, qc::Format::OpenQASM);
    auto dd  = std::make_unique<dd::Package<>>();
    auto sim = simulate(&qc, dd->makeZeroState(qc.getNqubits()), dd);
    EXPECT_TRUE(sim.p->e[0].isZeroTerminal());
    EXPECT_TRUE(sim.p->e[1].w.approximatelyOne());
    EXPECT_TRUE(sim.p->e[1].p->e[1].isZeroTerminal());
    EXPECT_TRUE(sim.p->e[1].p->e[0].w.approximatelyOne());
    auto func = buildFunctionality(&qc, dd);
    EXPECT_FALSE(func.p->e[0].isZeroTerminal());
    EXPECT_FALSE(func.p->e[1].isZeroTerminal());
    EXPECT_FALSE(func.p->e[2].isZeroTerminal());
    EXPECT_FALSE(func.p->e[3].isZeroTerminal());
    EXPECT_TRUE(func.p->e[0].p->e[1].w.approximatelyOne());
    EXPECT_TRUE(func.p->e[1].p->e[3].w.approximatelyOne());
    EXPECT_TRUE(func.p->e[2].p->e[0].w.approximatelyOne());
    EXPECT_TRUE(func.p->e[3].p->e[2].w.approximatelyOne());
}

TEST_F(DDFunctionality, basicTensorDumpTest) {
    QuantumComputation qc(2);
    qc.h(1);
    qc.x(0, 1_pc);

    std::stringstream ss{};
    dd::dumpTensorNetwork(ss, qc);

    const std::string reference = "{\"tensors\": [\n"
                                  "[[\"H   \", \"Q1\", \"GATE0\"], [\"q1_0\", \"q1_1\"], [2, 2], [[0.70710678118654757, 0], [0.70710678118654757, 0], [0.70710678118654757, 0], [-0.70710678118654757, 0]]],\n"
                                  "[[\"X   \", \"Q1\", \"Q0\", \"GATE1\"], [\"q1_1\", \"q0_0\", \"q1_2\", \"q0_1\"], [2, 2, 2, 2], [[1, 0], [0, 0], [0, 0], [0, 0], [0, 0], [1, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [1, 0], [0, 0], [0, 0], [1, 0], [0, 0]]]\n"
                                  "]}\n";
    EXPECT_EQ(ss.str(), reference);
}

TEST_F(DDFunctionality, compoundTensorDumpTest) {
    QuantumComputation qc(2);
    QuantumComputation comp(2);
    comp.h(1);
    comp.x(0, 1_pc);
    qc.emplace_back(comp.asOperation());

    std::stringstream ss{};
    dd::dumpTensorNetwork(ss, qc);

    const std::string reference = "{\"tensors\": [\n"
                                  "[[\"H   \", \"Q1\", \"GATE0\"], [\"q1_0\", \"q1_1\"], [2, 2], [[0.70710678118654757, 0], [0.70710678118654757, 0], [0.70710678118654757, 0], [-0.70710678118654757, 0]]],\n"
                                  "[[\"X   \", \"Q1\", \"Q0\", \"GATE1\"], [\"q1_1\", \"q0_0\", \"q1_2\", \"q0_1\"], [2, 2, 2, 2], [[1, 0], [0, 0], [0, 0], [0, 0], [0, 0], [1, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [1, 0], [0, 0], [0, 0], [1, 0], [0, 0]]]\n"
                                  "]}\n";
    EXPECT_EQ(ss.str(), reference);
}

TEST_F(DDFunctionality, errorTensorDumpTest) {
    QuantumComputation qc(2);
    qc.classicControlled(qc::X, 0, {0, 1U}, 1U);

    std::stringstream ss{};
    EXPECT_THROW(dd::dumpTensorNetwork(ss, qc), qc::QFRException);

    ss.str("");
    qc.erase(qc.begin());
    qc.barrier(0);
    qc.measure(0, 0);
    EXPECT_NO_THROW(dd::dumpTensorNetwork(ss, qc));

    ss.str("");
    qc.reset(0);
    EXPECT_THROW(dd::dumpTensorNetwork(ss, qc), qc::QFRException);
}

TEST_F(DDFunctionality, FuseTwoSingleQubitGates) {
    const std::size_t  nqubits = 1;
    QuantumComputation qc(nqubits);
    qc.x(0);
    qc.h(0);

    qc.print(std::cout);
    const auto e = buildFunctionality(&qc, dd);
    CircuitOptimizer::singleQubitGateFusion(qc);
    const auto f = buildFunctionality(&qc, dd);
    std::cout << "-----------------------------" << std::endl;
    qc.print(std::cout);
    EXPECT_EQ(qc.getNops(), 1);
    EXPECT_EQ(e, f);
}

TEST_F(DDFunctionality, FuseThreeSingleQubitGates) {
    const std::size_t  nqubits = 1;
    QuantumComputation qc(nqubits);
    qc.x(0);
    qc.h(0);
    qc.y(0);

    const auto e = buildFunctionality(&qc, dd);
    std::cout << "-----------------------------" << std::endl;
    qc.print(std::cout);
    CircuitOptimizer::singleQubitGateFusion(qc);
    const auto f = buildFunctionality(&qc, dd);
    std::cout << "-----------------------------" << std::endl;
    qc.print(std::cout);
    EXPECT_EQ(qc.getNops(), 1);
    EXPECT_EQ(e, f);
}

TEST_F(DDFunctionality, FuseNoSingleQubitGates) {
    const std::size_t  nqubits = 2;
    QuantumComputation qc(nqubits);
    qc.h(0);
    qc.x(1, 0_pc);
    qc.y(0);
    const auto e = buildFunctionality(&qc, dd);
    std::cout << "-----------------------------" << std::endl;
    qc.print(std::cout);
    CircuitOptimizer::singleQubitGateFusion(qc);
    const auto f = buildFunctionality(&qc, dd);
    std::cout << "-----------------------------" << std::endl;
    qc.print(std::cout);
    EXPECT_EQ(qc.getNops(), 3);
    EXPECT_EQ(e, f);
}

TEST_F(DDFunctionality, FuseSingleQubitGatesAcrossOtherGates) {
    const std::size_t  nqubits = 2;
    QuantumComputation qc(nqubits);
    qc.h(0);
    qc.z(1);
    qc.y(0);
    const auto e = buildFunctionality(&qc, dd);
    std::cout << "-----------------------------" << std::endl;
    qc.print(std::cout);
    CircuitOptimizer::singleQubitGateFusion(qc);
    const auto f = buildFunctionality(&qc, dd);
    std::cout << "-----------------------------" << std::endl;
    qc.print(std::cout);
    EXPECT_EQ(qc.getNops(), 2);
    EXPECT_EQ(e, f);
}
