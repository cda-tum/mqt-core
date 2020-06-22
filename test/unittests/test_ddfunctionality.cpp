/*
 * This file is part of IIC-JKU QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#include "gtest/gtest.h"
#include <random>

#include "QuantumComputation.hpp"

class DDFunctionality : public testing::TestWithParam<unsigned short> {
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
            e = ident = dd->makeIdent(0, (short)(nqubits-1));
            dd->incRef(ident);

	        std::array<std::mt19937_64::result_type , std::mt19937_64::state_size> random_data{};
	        std::random_device rd;
	        std::generate(begin(random_data), end(random_data), [&](){return rd();});
	        std::seed_seq seeds(begin(random_data), end(random_data));
	        mt.seed(seeds);
	        dist = std::uniform_real_distribution<fp> (0.0, 2 * qc::PI);
        }

        unsigned short                          nqubits             = 4;
        long                                    initialCacheCount   = 0;
        unsigned int                            initialComplexCount = 0;
        std::array<short, qc::MAX_QUBITS>       line{};
        dd::Edge                                e{}, ident{};
        std::unique_ptr<dd::Package>            dd;
		std::mt19937_64                         mt;
		std::uniform_real_distribution<fp>      dist;
};

INSTANTIATE_TEST_SUITE_P(Parameters,
                         DDFunctionality,
                         testing::Values(qc::I, qc::H, qc::X, qc::Y, qc::Z, qc::S, qc::Sdag, qc::T, qc::Tdag, qc::V, 
                                         qc::Vdag, qc::U3, qc::U2, qc::U1, qc::RX, qc::RY, qc::RZ, qc::P, qc::Pdag, 
                                         qc::SWAP, qc::iSWAP),
                         [](const testing::TestParamInfo<DDFunctionality::ParamType>& info) {
                             auto gate = (qc::OpType)info.param;
	                         switch (gate) {
                                case qc::I:     return "i";
                                case qc::H:     return "h";
                                case qc::X:     return "x";
                                case qc::Y:     return "y";
                                case qc::Z:     return "z";
                                case qc::S:     return "s";
                                case qc::Sdag:  return "sdg";
                                case qc::T:     return "t";
                                case qc::Tdag:  return "tdg";
                                case qc::V:     return "v";
                                case qc::Vdag:  return "vdg";
                                case qc::U3:    return "u3";
                                case qc::U2:    return "u2";
                                case qc::U1:    return "u1";
                                case qc::RX:    return "rx";
                                case qc::RY:    return "ry";
                                case qc::RZ:    return "rz";
                                case qc::SWAP:  return "swap";
                                case qc::iSWAP: return "iswap";
                                case qc::P:     return "p";
                                case qc::Pdag:  return "pdag";
                                default:        return "unknownGate";
                            }
                         }); 


TEST_P(DDFunctionality, standard_op_build_inverse_build) {
    auto gate = (qc::OpType)GetParam();
    
    qc::StandardOperation op;
    switch(gate) {
    	case qc::U3:
		    op = qc::StandardOperation(nqubits, 0,  gate, dist(mt), dist(mt), dist(mt));
		    break;
        case qc::U2: 
            op = qc::StandardOperation(nqubits, 0,  gate, dist(mt), dist(mt));
			break;
        case qc::RX:
		case qc::RY:
        case qc::RZ:
		case qc::U1:
            op = qc::StandardOperation(nqubits, 0,  gate, dist(mt));
            break;

        case qc::SWAP:
        case qc::iSWAP:
            op = qc::StandardOperation(nqubits, std::vector<unsigned short>{0, 1},  gate);
            break;
        case qc::P:
		case qc::Pdag: 
            op = qc::StandardOperation(nqubits, std::vector<qc::Control>{qc::Control(0)}, 1, 2, gate);
            break;
        default:
            op = qc::StandardOperation(nqubits, 0,  gate);
    }

    ASSERT_NO_THROW({e = dd->multiply(op.getDD(dd, line), op.getInverseDD(dd, line));});
    dd->incRef(e);

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

TEST_F(DDFunctionality, non_unitary) {
	qc::QuantumComputation qc;
	auto dummy_map = std::map<unsigned short, unsigned short>{};
	auto op = qc::NonUnitaryOperation(nqubits, {0,1,2,3}, {0,1,2,3});
	EXPECT_FALSE(op.isUnitary());
	try {
		op.getDD(dd, line);
		FAIL() << "Nothing thrown. Expected qc::QFRException";
	} catch (qc::QFRException const & err) {
		std::cout << err.what() << std::endl;
		SUCCEED();
	} catch (...) {
		FAIL() << "Expected qc::QFRException";
	}
	try {
		op.getInverseDD(dd, line);
		FAIL() << "Nothing thrown. Expected qc::QFRException";
	} catch (qc::QFRException const & err) {
		std::cout << err.what() << std::endl;
		SUCCEED();
	} catch (...) {
		FAIL() << "Expected qc::QFRException";
	}
	try {
		op.getDD(dd, line, dummy_map);
		FAIL() << "Nothing thrown. Expected qc::QFRException";
	} catch (qc::QFRException const & err) {
		std::cout << err.what() << std::endl;
		SUCCEED();
	} catch (...) {
		FAIL() << "Expected qc::QFRException";
	}
	try {
		op.getInverseDD(dd, line, dummy_map);
		FAIL() << "Nothing thrown. Expected qc::QFRException";
	} catch (qc::QFRException const & err) {
		std::cout << err.what() << std::endl;
		SUCCEED();
	} catch (...) {
		FAIL() << "Expected qc::QFRException";
	}
	EXPECT_TRUE(op.actsOn(0));
	op = qc::NonUnitaryOperation(nqubits, {0,1,2,3});
	EXPECT_TRUE(op.actsOn(0));
}
