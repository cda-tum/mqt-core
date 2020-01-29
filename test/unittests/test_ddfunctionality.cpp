/*
 * This file is part of IIC-JKU QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#include "gtest/gtest.h"

#include "QuantumComputation.hpp"

class DDFunctionality : public testing::Test {
    protected:
        void TearDown() override {
            if (!dd->isTerminal(e))
                dd->decRef(e);
            dd->garbageCollect(true);

            // number of complex table entries after clean-up should equal initial number of entries
            EXPECT_EQ(dd->cn.count, initialComplexCount);
            // number of available cache entries after clean-up should equal initial number of entries
            EXPECT_EQ(dd->cn.cacheCount, initialCacheCount);
        }

        void SetUp() override {
            // dd 
            dd                  = std::make_unique<dd::Package>();
            initialCacheCount   = dd->cn.cacheCount;
            initialComplexCount = dd->cn.count;
            dd->useMatrixNormalization(true);

            // initial state preparation
            line.fill(qc::LINE_DEFAULT);
            ASSERT_NO_THROW({
                qc::StandardOperation op(nqubits, 0, qc::X);
                e     = op.getDD(dd, line);
                ident = dd->multiply(dd->makeIdent(0, 3), e);
                dd->incRef(ident);
            });
        }

        unsigned short                          nqubits             = 4;
        long                                    initialCacheCount   = 0;
        long                                    initialComplexCount = 0;
        std::array<short, qc::MAX_QUBITS>       line{};
        dd::Edge                                e{}, ident{}, state_preparation{};
        std::unique_ptr<dd::Package>            dd;
};

class DDFunctionalityParameters : public DDFunctionality,
                                  public testing::WithParamInterface<unsigned short> {
};

INSTANTIATE_TEST_SUITE_P(DDFunctionalityParameters,
                         DDFunctionalityParameters,
                         testing::Values(qc::I, qc::H, qc::X, qc::Y, qc::Z, qc::S, qc::Sdag, qc::T, qc::Tdag, qc::V, 
                                         qc::Vdag, qc::U3, qc::U2, qc::U1, qc::RX, qc::RY, qc::RZ, qc::P, qc::Pdag, 
                                         qc::SWAP, qc::iSWAP),
                         [](const testing::TestParamInfo<DDFunctionalityParameters::ParamType>& info) {
                             auto gate = (qc::Gate)info.param;
	                         switch (gate) {
                                case qc::I:     return "I";
                                case qc::H:     return "H";
                                case qc::X:     return "X";
                                case qc::Y:     return "U3";
                                case qc::Z:     return "Z";
                                case qc::S:     return "S";
                                case qc::Sdag:  return "sdg";
                                case qc::T:     return "T";
                                case qc::Tdag:  return "tdg";
                                case qc::V:     return "V";
                                case qc::Vdag:  return "vdg";
                                case qc::U3:    return "u3";
                                case qc::U2:    return "u2";
                                case qc::U1:    return "u1";
                                case qc::RX:    return "rx";
                                case qc::RY:    return "ry";
                                case qc::RZ:    return "rz";
                                case qc::SWAP:  return "swap";
                                case qc::iSWAP: return "iswap";
                                case qc::P:     return "P";
                                case qc::Pdag:  return "pdag";
                                default:        return "unknownGate";
                            }       
                         }); 


TEST_P(DDFunctionalityParameters, standard_op_build_inverse_build) {
    auto gate = (qc::Gate)GetParam();
    
    qc::StandardOperation             op;
    switch(gate) {
        case qc::U2: 
            op = qc::StandardOperation(nqubits, 0,  gate, 1);
			break;
        case qc::RX:
		case qc::RY:
        case qc::RZ:
		case qc::U1:
            op = qc::StandardOperation(nqubits, 0,  gate, qc::PI);
            break;

        case qc::SWAP:
        case qc::iSWAP:
            op = qc::StandardOperation(nqubits, std::vector<unsigned short>{0, 1},  gate);
            break;
        case qc::P:
		case qc::Pdag: 
            op = qc::StandardOperation(nqubits, std::vector<qc::Control>{qc::Control(0), qc::Control(1)}, 2, 3, gate);
            break;
        default:
            op = qc::StandardOperation(nqubits, 0,  gate);
    }

    ASSERT_NO_THROW({e = dd->multiply(op.getDD(dd, line), e);});
    ASSERT_NO_THROW({e = dd->multiply(op.getInverseDD(dd, line), e);});
    dd->incRef(e);
   
    //dd->printVector(e);
    //dd->printVector(ident);

    EXPECT_TRUE(dd::Package::equals(ident, e));
}

TEST_F(DDFunctionality, build_circuit) {
    qc::QuantumComputation qc(nqubits);

    qc.emplace_back<qc::StandardOperation>(nqubits, 0,  qc::X);
    qc.emplace_back<qc::StandardOperation>(nqubits, std::vector<unsigned short>{0, 1},  qc::SWAP);
    qc.emplace_back<qc::StandardOperation>(nqubits, 0,  qc::H);
    qc.emplace_back<qc::StandardOperation>(nqubits, 3,  qc::S);
    qc.emplace_back<qc::StandardOperation>(nqubits, 2,  qc::Sdag);
    qc.emplace_back<qc::StandardOperation>(nqubits, 0,  qc::V);
    qc.emplace_back<qc::StandardOperation>(nqubits, 1,  qc::T);
    qc.emplace_back<qc::StandardOperation>(nqubits, qc::Control(0), 1,  qc::X);
    qc.emplace_back<qc::StandardOperation>(nqubits, qc::Control(3), 2,  qc::X);
    qc.emplace_back<qc::StandardOperation>(nqubits, std::vector<qc::Control>({qc::Control(3), qc::Control(2)}), 0,  qc::X);
    
    qc.emplace_back<qc::StandardOperation>(nqubits, std::vector<qc::Control>({qc::Control(3), qc::Control(2)}), 0,  qc::X);
    qc.emplace_back<qc::StandardOperation>(nqubits, qc::Control(3), 2,  qc::X);
    qc.emplace_back<qc::StandardOperation>(nqubits, qc::Control(0), 1,  qc::X);
    qc.emplace_back<qc::StandardOperation>(nqubits, 1,  qc::Tdag);
    qc.emplace_back<qc::StandardOperation>(nqubits, 0,  qc::Vdag);
    qc.emplace_back<qc::StandardOperation>(nqubits, 2,  qc::S);
    qc.emplace_back<qc::StandardOperation>(nqubits, 3,  qc::Sdag);
    qc.emplace_back<qc::StandardOperation>(nqubits, 0,  qc::H);
    qc.emplace_back<qc::StandardOperation>(nqubits, std::vector<unsigned short>{0, 1},  qc::SWAP);
    qc.emplace_back<qc::StandardOperation>(nqubits, 0,  qc::X);


    e = dd->multiply(qc.buildFunctionality(dd), e);
   
    //dd->printVector(e);
    //dd->printVector(ident);

    EXPECT_TRUE(dd::Package::equals(ident, e));

    qc.emplace_back<qc::StandardOperation>(nqubits, 0,  qc::X);
    e = dd->multiply(qc.buildFunctionality(dd), e);
    dd->incRef(e);

    EXPECT_FALSE(dd::Package::equals(ident, e));
}


