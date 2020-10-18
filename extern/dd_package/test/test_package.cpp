#include <memory>

#include "DDpackage.h"
#include "util.h"
#include "gtest/gtest.h"

TEST(DDPackageTest, TrivialTest) {
    auto dd = std::make_unique<dd::Package>();

    short line[2] = {2};
    dd::Edge x_gate = dd->makeGateDD(Xmat, 1, line);
    dd::Edge h_gate = dd->makeGateDD(Hmat, 1, line);

    ASSERT_EQ(dd->getValueByPath(h_gate, "0"), (dd::ComplexValue{dd::SQRT_2, 0}));

    dd::Edge zero_state = dd->makeZeroState(1);
    dd::Edge h_state = dd->multiply(h_gate, zero_state);
    dd::Edge one_state = dd->multiply(x_gate, zero_state);

    ASSERT_EQ(dd->fidelity(zero_state, one_state), 0.0);
    ASSERT_NEAR(dd->fidelity(zero_state, h_state), 0.5, dd::ComplexNumbers::TOLERANCE);
    ASSERT_NEAR(dd->fidelity(one_state, h_state), 0.5, dd::ComplexNumbers::TOLERANCE);
}

TEST(DDPackageTest, BellState) {
    auto dd = std::make_unique<dd::Package>();

    short line[2] = {-1,2};
    dd::Edge h_gate = dd->makeGateDD(Hmat, 2, line);
    dd::Edge cx_gate = dd->makeGateDD({Xmat[0][0], Xmat[0][1], Xmat[1][0], Xmat[1][1]}, 2, {2,1});
    dd::Edge zero_state = dd->makeZeroState(2);

    dd::Edge bell_state = dd->multiply(dd->multiply(cx_gate, h_gate), zero_state);

    ASSERT_EQ(dd->getValueByPath(bell_state, "00"), (dd::ComplexValue{dd::SQRT_2, 0}));
    ASSERT_EQ(dd->getValueByPath(bell_state, "02"), (dd::ComplexValue{0, 0}));
    ASSERT_EQ(dd->getValueByPath(bell_state, "20"), (dd::ComplexValue{0, 0}));
    ASSERT_EQ(dd->getValueByPath(bell_state, "22"), (dd::ComplexValue{dd::SQRT_2, 0}));

    ASSERT_DOUBLE_EQ(dd->fidelity(zero_state, bell_state), 0.5);

    dd->printDD(bell_state, 64);
}

TEST(DDPackageTest, IdentityTrace) {
    auto dd = std::make_unique<dd::Package>();
    auto fullTrace = dd->trace(dd->makeIdent(0, 3));

    ASSERT_EQ(fullTrace, (dd::ComplexValue{16,0}));
}

TEST(DDPackageTest, StateGenerationManipulation) {
	auto dd = std::make_unique<dd::Package>();

	auto b = std::bitset<dd::MAXN>{2};
	auto e = dd->makeBasisState(6, b);
	auto f = dd->makeBasisState(6, {dd::BasisStates::zero,
								            dd::BasisStates::one,
								            dd::BasisStates::plus,
								            dd::BasisStates::minus,
								            dd::BasisStates::left,
								            dd::BasisStates::right});
	dd->incRef(e);
	dd->incRef(f);
	dd->incRef(e);
	auto g = dd->add(e, f);
	auto h = dd->transpose(g);
	auto i = dd->conjugateTranspose(f);
	dd->decRef(e);
	dd->decRef(f);
	auto j = dd->kronecker(h, i);
	dd->incRef(j);
	dd->printActive(6);
	dd->printUniqueTable(6);
	dd->printInformation();
}

