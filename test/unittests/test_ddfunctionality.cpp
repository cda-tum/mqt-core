/*
 * This file is part of JKQ QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#include "QuantumComputation.hpp"

#include "gtest/gtest.h"
#include <random>

using namespace qc;
using namespace dd;

class DDFunctionality: public testing::TestWithParam<qc::OpType> {
protected:
    void TearDown() override {
        if (!e.isTerminal())
            dd->decRef(e);
        dd->garbageCollect(true);

        // number of complex table entries after clean-up should equal initial number of entries
        EXPECT_EQ(dd->cn.complexTable.getCount(), initialComplexCount);
        // number of available cache entries after clean-up should equal initial number of entries
        EXPECT_EQ(dd->cn.complexCache.getCount(), initialCacheCount);
    }

    void SetUp() override {
        // dd
        dd                  = std::make_unique<dd::Package>(nqubits);
        initialCacheCount   = dd->cn.complexCache.getCount();
        initialComplexCount = dd->cn.complexTable.getCount();

        // initial state preparation
        e = ident = dd->makeIdent(nqubits);
        dd->incRef(ident);

        std::array<std::mt19937_64::result_type, std::mt19937_64::state_size> random_data{};
        std::random_device                                                    rd;
        std::generate(begin(random_data), end(random_data), [&]() { return rd(); });
        std::seed_seq seeds(begin(random_data), end(random_data));
        mt.seed(seeds);
        dist = std::uniform_real_distribution<dd::fp>(0.0, 2 * dd::PI);
    }

    dd::QubitCount                         nqubits             = 4;
    std::size_t                            initialCacheCount   = 0;
    std::size_t                            initialComplexCount = 0;
    qc::MatrixDD                           e{}, ident{};
    std::unique_ptr<dd::Package>           dd;
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
    auto gate = (qc::OpType)GetParam();

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
            op = qc::StandardOperation(nqubits, dd::Controls{}, 0, 1, gate);
            break;
        case qc::Peres:
        case qc::Peresdag:
            op = qc::StandardOperation(nqubits, {0_pc}, 1, 2, gate);
            break;
        default:
            op = qc::StandardOperation(nqubits, 0, gate);
    }

    ASSERT_NO_THROW({ e = dd->multiply(op.getDD(dd), op.getInverseDD(dd)); });
    dd->incRef(e);

    EXPECT_EQ(ident, e);
}

TEST_F(DDFunctionality, build_circuit) {
    qc::QuantumComputation qc(nqubits);

    qc.emplace_back<qc::StandardOperation>(nqubits, 0, qc::X);
    qc.emplace_back<qc::StandardOperation>(nqubits, std::vector<dd::Qubit>{0, 1}, qc::SWAP);
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
    qc.emplace_back<qc::StandardOperation>(nqubits, std::vector<dd::Qubit>{0, 1}, qc::SWAP);
    qc.emplace_back<qc::StandardOperation>(nqubits, 0, qc::X);

    e = dd->multiply(qc.buildFunctionality(dd), e);

    EXPECT_EQ(ident, e);

    qc.emplace_back<qc::StandardOperation>(nqubits, 0, qc::X);
    e = dd->multiply(qc.buildFunctionality(dd), e);
    dd->incRef(e);

    EXPECT_NE(ident, e);
}

TEST_F(DDFunctionality, non_unitary) {
    qc::QuantumComputation qc;
    auto                   dummy_map = Permutation{};
    auto                   op        = qc::NonUnitaryOperation(nqubits, {0, 1, 2, 3}, {0, 1, 2, 3});
    EXPECT_FALSE(op.isUnitary());
    try {
        op.getDD(dd);
        FAIL() << "Nothing thrown. Expected qc::QFRException";
    } catch (qc::QFRException const& err) {
        std::cout << err.what() << std::endl;
        SUCCEED();
    } catch (...) {
        FAIL() << "Expected qc::QFRException";
    }
    try {
        op.getInverseDD(dd);
        FAIL() << "Nothing thrown. Expected qc::QFRException";
    } catch (qc::QFRException const& err) {
        std::cout << err.what() << std::endl;
        SUCCEED();
    } catch (...) {
        FAIL() << "Expected qc::QFRException";
    }
    try {
        op.getDD(dd, dummy_map);
        FAIL() << "Nothing thrown. Expected qc::QFRException";
    } catch (qc::QFRException const& err) {
        std::cout << err.what() << std::endl;
        SUCCEED();
    } catch (...) {
        FAIL() << "Expected qc::QFRException";
    }
    try {
        op.getInverseDD(dd, dummy_map);
        FAIL() << "Nothing thrown. Expected qc::QFRException";
    } catch (qc::QFRException const& err) {
        std::cout << err.what() << std::endl;
        SUCCEED();
    } catch (...) {
        FAIL() << "Expected qc::QFRException";
    }
}
