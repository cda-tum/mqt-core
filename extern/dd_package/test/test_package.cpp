/*
 * This file is part of the JKQ DD Package which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum_dd/ for more information.
 */

#include "DDexport.h"
#include "DDpackage.h"
#include "util.h"

#include "gtest/gtest.h"
#include <memory>

TEST(DDPackageTest, OperationLookupTest) {
    auto dd = std::make_unique<dd::Package>();

    // dd::ATrue is not the operation that is being stored, but for the test it doesn't matter
    auto tmp_op = dd->OperationLookup(dd::ATrue, {2}, 1);
    EXPECT_TRUE(tmp_op.p == nullptr);

    dd::Edge x_gate = dd->makeGateDD(Xmat, 1, {2});
    dd->OperationInsert(dd::ATrue, {2}, x_gate, 1);
    tmp_op = dd->OperationLookup(dd::ATrue, {2}, 1);
    EXPECT_TRUE(tmp_op.p == x_gate.p);

    tmp_op = dd->multiply(tmp_op, x_gate);

    // I only check this, so that the above test is evaluated when compiled using release mode
    EXPECT_TRUE(tmp_op.p != nullptr);

    dd->garbageCollect(true);
    tmp_op = dd->OperationLookup(dd::ATrue, {2}, 1);
    EXPECT_TRUE(tmp_op.p == nullptr);
}

TEST(DDPackageTest, TrivialTest) {
    auto dd = std::make_unique<dd::Package>();

    dd::Edge x_gate = dd->makeGateDD(Xmat, 1, {2});
    dd::Edge h_gate = dd->makeGateDD(Hmat, 1, {2});

    ASSERT_EQ(dd->getValueByPath(h_gate, "0"), (dd::ComplexValue{dd::SQRT_2, 0}));

    dd::Edge zero_state = dd->makeZeroState(1);
    dd::Edge h_state    = dd->multiply(h_gate, zero_state);
    dd::Edge one_state  = dd->multiply(x_gate, zero_state);

    ASSERT_EQ(dd->fidelity(zero_state, one_state), 0.0);
    ASSERT_NEAR(dd->fidelity(zero_state, h_state), 0.5, dd::ComplexNumbers::TOLERANCE);
    ASSERT_NEAR(dd->fidelity(one_state, h_state), 0.5, dd::ComplexNumbers::TOLERANCE);
}

TEST(DDPackageTest, BellState) {
    auto dd = std::make_unique<dd::Package>();

    dd::Edge h_gate     = dd->makeGateDD(Hmat, 2, {-1, 2});
    dd::Edge cx_gate    = dd->makeGateDD(Xmat, 2, {2, 1});
    dd::Edge zero_state = dd->makeZeroState(2);

    dd::Edge bell_state = dd->multiply(dd->multiply(cx_gate, h_gate), zero_state);

    ASSERT_EQ(dd->getValueByPath(bell_state, "00"), (dd::ComplexValue{dd::SQRT_2, 0}));
    ASSERT_EQ(dd->getValueByPath(bell_state, "02"), (dd::ComplexValue{0, 0}));
    ASSERT_EQ(dd->getValueByPath(bell_state, "20"), (dd::ComplexValue{0, 0}));
    ASSERT_EQ(dd->getValueByPath(bell_state, "22"), (dd::ComplexValue{dd::SQRT_2, 0}));

    ASSERT_EQ(dd->getValueByPath(bell_state, 0, 0), (dd::ComplexValue{dd::SQRT_2, 0}));
    ASSERT_EQ(dd->getValueByPath(bell_state, 1, 0), (dd::ComplexValue{0, 0}));
    ASSERT_EQ(dd->getValueByPath(bell_state, 2, 0), (dd::ComplexValue{0, 0}));
    ASSERT_EQ(dd->getValueByPath(bell_state, 3, 0), (dd::ComplexValue{dd::SQRT_2, 0}));

    auto goal_state = std::vector<std::pair<float, float>>{{dd::SQRT_2, 0.}, {0., 0.}, {0., 0.}, {dd::SQRT_2, 0.}};
    ASSERT_EQ(dd->getVector(bell_state), goal_state);

    ASSERT_DOUBLE_EQ(dd->fidelity(zero_state, bell_state), 0.5);

    dd->printDD(bell_state, 64);
    dd::export2Dot(bell_state, "bell_state_colored_labels.dot", true, true, true, false, false);
    dd::export2Dot(bell_state, "bell_state_colored_labels_classic.dot", true, true, true, true, false);
    dd::export2Dot(bell_state, "bell_state_mono_labels.dot", true, false, true, false, false);
    dd::export2Dot(bell_state, "bell_state_mono_labels_classic.dot", true, false, true, true, false);
    dd::export2Dot(bell_state, "bell_state_colored.dot", true, true, false, false, false);
    dd::export2Dot(bell_state, "bell_state_colored_classic.dot", true, true, false, true, false);
    dd::export2Dot(bell_state, "bell_state_mono.dot", true, false, false, false, false);
    dd::export2Dot(bell_state, "bell_state_mono_classic.dot", true, false, false, true, false);
}

TEST(DDPackageTest, IdentityTrace) {
    auto dd        = std::make_unique<dd::Package>();
    auto fullTrace = dd->trace(dd->makeIdent(4));

    ASSERT_EQ(fullTrace, (dd::ComplexValue{16, 0}));
}

TEST(DDPackageTest, PartialIdentityTrace) {
    auto dd  = std::make_unique<dd::Package>();
    auto tr  = dd->partialTrace(dd->makeIdent(2), std::bitset<dd::MAXN>(1));
    auto mul = dd->multiply(tr, tr);
    EXPECT_EQ(CN::val(mul.w.r), 4.0);
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

TEST(DDPackageTest, VectorSerializationTest) {
    auto dd = std::make_unique<dd::Package>();

    dd::Edge h_gate     = dd->makeGateDD(Hmat, 2, {-1, 2});
    dd::Edge cx_gate    = dd->makeGateDD(Xmat, 2, {2, 1});
    dd::Edge zero_state = dd->makeZeroState(2);

    dd::Edge bell_state = dd->multiply(dd->multiply(cx_gate, h_gate), zero_state);

    dd::serialize(bell_state, "bell_state.dd", true, false);
    auto deserialized_bell_state = dd::deserialize(dd, "bell_state.dd", true, false);
    EXPECT_TRUE(dd->equals(bell_state, deserialized_bell_state));

    dd::serialize(bell_state, "bell_state_binary.dd", true, true);
    deserialized_bell_state = dd::deserialize(dd, "bell_state_binary.dd", true, true);
    EXPECT_TRUE(dd->equals(bell_state, deserialized_bell_state));
}

TEST(DDPackageTest, BellMatrix) {
    auto dd = std::make_unique<dd::Package>();

    dd::Edge h_gate  = dd->makeGateDD(Hmat, 2, {-1, 2});
    dd::Edge cx_gate = dd->makeGateDD(Xmat, 2, {2, 1});

    dd::Edge bell_matrix = dd->multiply(cx_gate, h_gate);

    ASSERT_EQ(dd->getValueByPath(bell_matrix, "00"), (dd::ComplexValue{dd::SQRT_2, 0}));
    ASSERT_EQ(dd->getValueByPath(bell_matrix, "02"), (dd::ComplexValue{0, 0}));
    ASSERT_EQ(dd->getValueByPath(bell_matrix, "20"), (dd::ComplexValue{0, 0}));
    ASSERT_EQ(dd->getValueByPath(bell_matrix, "22"), (dd::ComplexValue{dd::SQRT_2, 0}));

    ASSERT_EQ(dd->getValueByPath(bell_matrix, 0, 0), (dd::ComplexValue{dd::SQRT_2, 0}));
    ASSERT_EQ(dd->getValueByPath(bell_matrix, 1, 0), (dd::ComplexValue{0, 0}));
    ASSERT_EQ(dd->getValueByPath(bell_matrix, 2, 0), (dd::ComplexValue{0, 0}));
    ASSERT_EQ(dd->getValueByPath(bell_matrix, 3, 0), (dd::ComplexValue{dd::SQRT_2, 0}));

    ASSERT_EQ(dd->getValueByPath(bell_matrix, 0, 1), (dd::ComplexValue{0, 0}));
    ASSERT_EQ(dd->getValueByPath(bell_matrix, 1, 1), (dd::ComplexValue{dd::SQRT_2, 0}));
    ASSERT_EQ(dd->getValueByPath(bell_matrix, 2, 1), (dd::ComplexValue{dd::SQRT_2, 0}));
    ASSERT_EQ(dd->getValueByPath(bell_matrix, 3, 1), (dd::ComplexValue{0, 0}));

    ASSERT_EQ(dd->getValueByPath(bell_matrix, 0, 2), (dd::ComplexValue{dd::SQRT_2, 0}));
    ASSERT_EQ(dd->getValueByPath(bell_matrix, 1, 2), (dd::ComplexValue{0, 0}));
    ASSERT_EQ(dd->getValueByPath(bell_matrix, 2, 2), (dd::ComplexValue{0, 0}));
    ASSERT_EQ(dd->getValueByPath(bell_matrix, 3, 2), (dd::ComplexValue{-dd::SQRT_2, 0}));

    ASSERT_EQ(dd->getValueByPath(bell_matrix, 0, 3), (dd::ComplexValue{0, 0}));
    ASSERT_EQ(dd->getValueByPath(bell_matrix, 1, 3), (dd::ComplexValue{dd::SQRT_2, 0}));
    ASSERT_EQ(dd->getValueByPath(bell_matrix, 2, 3), (dd::ComplexValue{-dd::SQRT_2, 0}));
    ASSERT_EQ(dd->getValueByPath(bell_matrix, 3, 3), (dd::ComplexValue{0, 0}));

    auto goal_row_0  = dd::Package::CVec{{dd::SQRT_2, 0.}, {0., 0.}, {dd::SQRT_2, 0.}, {0., 0.}};
    auto goal_row_1  = dd::Package::CVec{{0., 0.}, {dd::SQRT_2, 0.}, {0., 0.}, {dd::SQRT_2, 0.}};
    auto goal_row_2  = dd::Package::CVec{{0., 0.}, {dd::SQRT_2, 0.}, {0., 0.}, {-dd::SQRT_2, 0.}};
    auto goal_row_3  = dd::Package::CVec{{dd::SQRT_2, 0.}, {0., 0.}, {-dd::SQRT_2, 0.}, {0., 0.}};
    auto goal_matrix = dd::Package::CMat{goal_row_0, goal_row_1, goal_row_2, goal_row_3};
    ASSERT_EQ(dd->getMatrix(bell_matrix), goal_matrix);

    dd->printDD(bell_matrix, 64);
    dd::export2Dot(bell_matrix, "bell_matrix_colored_labels.dot", false, true, true, false, false);
    dd::export2Dot(bell_matrix, "bell_matrix_colored_labels_classic.dot", false, true, true, true, false);
    dd::export2Dot(bell_matrix, "bell_matrix_mono_labels.dot", false, false, true, false, false);
    dd::export2Dot(bell_matrix, "bell_matrix_mono_labels_classic.dot", false, false, true, true, false);
    dd::export2Dot(bell_matrix, "bell_matrix_colored.dot", false, true, false, false, false);
    dd::export2Dot(bell_matrix, "bell_matrix_colored_classic.dot", false, true, false, true, false);
    dd::export2Dot(bell_matrix, "bell_matrix_mono.dot", false, false, false, false, false);
    dd::export2Dot(bell_matrix, "bell_matrix_mono_classic.dot", false, false, false, true, false);
}

TEST(DDPackageTest, MatrixSerializationTest) {
    auto dd = std::make_unique<dd::Package>();

    dd::Edge h_gate  = dd->makeGateDD(Hmat, 2, {-1, 2});
    dd::Edge cx_gate = dd->makeGateDD(Xmat, 2, {2, 1});

    dd::Edge bell_matrix = dd->multiply(cx_gate, h_gate);

    dd::serialize(bell_matrix, "bell_matrix.dd", false, false);
    auto deserialized_bell_matrix = dd::deserialize(dd, "bell_matrix.dd", false, false);
    EXPECT_TRUE(dd->equals(bell_matrix, deserialized_bell_matrix));

    dd::serialize(bell_matrix, "bell_matrix_binary.dd", false, true);
    deserialized_bell_matrix = dd::deserialize(dd, "bell_matrix_binary.dd", false, true);
    EXPECT_TRUE(dd->equals(bell_matrix, deserialized_bell_matrix));
}

TEST(DDPackageTest, SerializationErrors) {
    auto dd = std::make_unique<dd::Package>();

    dd::Edge h_gate     = dd->makeGateDD(Hmat, 2, {-1, 2});
    dd::Edge cx_gate    = dd->makeGateDD(Xmat, 2, {2, 1});
    dd::Edge zero_state = dd->makeZeroState(2);
    dd::Edge bell_state = dd->multiply(dd->multiply(cx_gate, h_gate), zero_state);

    // test non-existing file
    dd::serialize(bell_state, "./path/that/does/not/exist/filename.dd");
    auto e = dd::deserialize(dd, "./path/that/does/not/exist/filename.dd", true);
    EXPECT_TRUE(dd::Package::equals(e, dd::Package::DDzero));

    // test wrong version number
    std::stringstream ss{};
    ss << 2.0 << std::endl;
    EXPECT_THROW(dd::deserialize(dd, ss, false, false), std::runtime_error);
    ss.str("");
    fp version = 2.0;
    ss.write(reinterpret_cast<const char*>(&version), sizeof(fp));
    EXPECT_THROW(dd::deserialize(dd, ss, false, true), std::runtime_error);

    // test wrong format
    ss.str("");
    ss << "0.1" << std::endl;
    ss << "not_complex" << std::endl;
    EXPECT_THROW(dd::deserialize(dd, ss), std::runtime_error);

    ss.str("");
    ss << "0.1" << std::endl;
    ss << "1.0" << std::endl;
    ss << "no_node_here" << std::endl;
    EXPECT_THROW(dd::deserialize(dd, ss), std::runtime_error);
}

TEST(DDPackageTest, TestConsistency) {
    auto dd = std::make_unique<dd::Package>();

    dd::Edge h_gate     = dd->makeGateDD(Hmat, 2, {2, -1});
    dd::Edge cx_gate    = dd->makeGateDD(Xmat, 2, {1, 2});
    dd::Edge zero_state = dd->makeZeroState(2);

    dd::Edge bell_matrix = dd->multiply(cx_gate, h_gate);
    dd->incRef(bell_matrix);
    auto local = dd->is_locally_consistent_dd(bell_matrix);
    EXPECT_TRUE(local);
    auto global = dd->is_globally_consistent_dd(bell_matrix);
    EXPECT_TRUE(global);
    dd->debugnode(bell_matrix.p);

    dd::Edge bell_state = dd->multiply(bell_matrix, zero_state);
    dd->incRef(bell_state);
    local = dd->is_locally_consistent_dd(bell_state);
    EXPECT_TRUE(local);
    global = dd->is_globally_consistent_dd(bell_state);
    EXPECT_TRUE(global);
    dd->debugnode(bell_state.p);
}

TEST(DDPackageTest, ToffoliTable) {
    auto dd = std::make_unique<dd::Package>();

    // try to search for a toffoli in an empty table
    auto toffoli = dd->TTlookup(3, static_cast<unsigned short>(2), 2, {0, 1, 2});
    EXPECT_EQ(toffoli.p, nullptr);
    if (toffoli.p == nullptr) {
        toffoli = dd->makeGateDD(Xmat, 3, {0, 1, 2});
        dd->TTinsert(3, static_cast<unsigned short>(2), 2, {0, 1, 2}, toffoli);
    }

    // try again with same toffoli
    auto toffoliTableEntry = dd->TTlookup(3, static_cast<unsigned short>(2), 2, {0, 1, 2});
    EXPECT_NE(toffoliTableEntry.p, nullptr);
    EXPECT_TRUE(dd->equals(toffoli, toffoliTableEntry));

    // try again with different controlled toffoli
    toffoliTableEntry = dd->TTlookup(3, static_cast<unsigned short>(2), 2, {1, 1, 2});
    EXPECT_EQ(toffoliTableEntry.p, nullptr);

    // try again with different qubit toffoli
    toffoliTableEntry = dd->TTlookup(4, static_cast<unsigned short>(3), 3, {1, 1, 1, 2});
    EXPECT_EQ(toffoliTableEntry.p, nullptr);
}

TEST(DDPackageTest, Extend) {
    auto dd = std::make_unique<dd::Package>();

    auto id = dd->makeIdent(3);
    dd->printDD(id, 64);
    EXPECT_EQ(id.p->v, 2);
    EXPECT_TRUE(dd->equals(id.p->e[0], id.p->e[3]));
    EXPECT_TRUE(dd->equals(id.p->e[1], id.p->e[2]));
    EXPECT_TRUE(id.p->ident);

    auto ext = dd->extend(id, 0, 1);
    EXPECT_EQ(ext.p->v, 3);
    EXPECT_TRUE(dd->equals(ext.p->e[0], ext.p->e[3]));
    EXPECT_TRUE(dd->equals(ext.p->e[1], ext.p->e[2]));
    EXPECT_TRUE(ext.p->ident);
}

TEST(DDPackageTest, Identity) {
    auto dd = std::make_unique<dd::Package>();
    EXPECT_TRUE(dd->equals(dd->makeIdent(0), dd->DDone));
    EXPECT_TRUE(dd->equals(dd->makeIdent(0, -1), dd->DDone));
    auto id3 = dd->makeIdent(0, 2);
    EXPECT_TRUE(dd->equals(dd->makeIdent(3), id3));
    auto id2 = dd->makeIdent(0, 1); // should be found in IdTable
    EXPECT_TRUE(dd->equals(dd->makeIdent(2), id2));
    auto id4 = dd->makeIdent(0, 3); // should use id3 and extend it
    EXPECT_TRUE(dd->equals(dd->makeIdent(4), id4));
    auto idCached = dd->makeIdent(0, 3);
    EXPECT_TRUE(dd::Package::equals(id4, idCached));
}

TEST(DDPackageTest, TestLocalInconsistency) {
    auto dd = std::make_unique<dd::Package>();

    dd::Edge h_gate     = dd->makeGateDD(Hmat, 2, {2, -1});
    dd::Edge cx_gate    = dd->makeGateDD(Xmat, 2, {1, 2});
    dd::Edge zero_state = dd->makeZeroState(2);

    dd::Edge bell_state = dd->multiply(dd->multiply(cx_gate, h_gate), zero_state);
    auto     local      = dd->is_locally_consistent_dd(bell_state);
    EXPECT_FALSE(local);
    bell_state.p->ref = 1;
    local             = dd->is_locally_consistent_dd(bell_state);
    EXPECT_FALSE(local);
    bell_state.p->ref = 0;
    dd->incRef(bell_state);

    bell_state.p->v = 2;
    local           = dd->is_locally_consistent_dd(bell_state);
    EXPECT_FALSE(local);
    bell_state.p->v = 1;

    bell_state.p->e[0].w.r->ref = 0;
    local                       = dd->is_locally_consistent_dd(bell_state);
    EXPECT_FALSE(local);
    bell_state.p->e[0].w.r->ref = 1;
}

TEST(DDPackageTest, Ancillaries) {
    auto dd          = std::make_unique<dd::Package>();
    auto h_gate      = dd->makeGateDD(Hmat, 2, {2, -1});
    auto cx_gate     = dd->makeGateDD(Xmat, 2, {1, 2});
    auto bell_matrix = dd->multiply(cx_gate, h_gate);

    auto reduced_bell_matrix = dd->reduceAncillae(bell_matrix, {0b00});
    EXPECT_EQ(bell_matrix, reduced_bell_matrix);
    reduced_bell_matrix = dd->reduceAncillae(bell_matrix, {0b100});
    EXPECT_EQ(bell_matrix, reduced_bell_matrix);

    auto extended_bell_matrix = dd->extend(bell_matrix, 2);
    reduced_bell_matrix       = dd->reduceAncillae(extended_bell_matrix, {0b1100});
    EXPECT_EQ(reduced_bell_matrix.p->e[1], dd::Package::DDzero);
    EXPECT_EQ(reduced_bell_matrix.p->e[2], dd::Package::DDzero);
    EXPECT_EQ(reduced_bell_matrix.p->e[3], dd::Package::DDzero);

    EXPECT_EQ(reduced_bell_matrix.p->e[0].p->e[0].p, bell_matrix.p);
    EXPECT_EQ(reduced_bell_matrix.p->e[0].p->e[1], dd::Package::DDzero);
    EXPECT_EQ(reduced_bell_matrix.p->e[0].p->e[2], dd::Package::DDzero);
    EXPECT_EQ(reduced_bell_matrix.p->e[0].p->e[3], dd::Package::DDzero);

    reduced_bell_matrix = dd->reduceAncillae(extended_bell_matrix, {0b1100}, false);
    EXPECT_EQ(reduced_bell_matrix.p->e[1], dd::Package::DDzero);
    EXPECT_EQ(reduced_bell_matrix.p->e[2], dd::Package::DDzero);
    EXPECT_EQ(reduced_bell_matrix.p->e[3], dd::Package::DDzero);

    EXPECT_EQ(reduced_bell_matrix.p->e[0].p->e[0].p, bell_matrix.p);
    EXPECT_EQ(reduced_bell_matrix.p->e[0].p->e[1], dd::Package::DDzero);
    EXPECT_EQ(reduced_bell_matrix.p->e[0].p->e[2], dd::Package::DDzero);
    EXPECT_EQ(reduced_bell_matrix.p->e[0].p->e[3], dd::Package::DDzero);
}

TEST(DDPackageTest, Garbage) {
    auto dd          = std::make_unique<dd::Package>();
    auto h_gate      = dd->makeGateDD(Hmat, 2, {2, -1});
    auto cx_gate     = dd->makeGateDD(Xmat, 2, {1, 2});
    auto bell_matrix = dd->multiply(cx_gate, h_gate);

    auto reduced_bell_matrix = dd->reduceGarbage(bell_matrix, {0b00});
    EXPECT_EQ(bell_matrix, reduced_bell_matrix);
    reduced_bell_matrix = dd->reduceGarbage(bell_matrix, {0b100});
    EXPECT_EQ(bell_matrix, reduced_bell_matrix);

    reduced_bell_matrix = dd->reduceGarbage(bell_matrix, {0b10});
    auto mat            = dd->getMatrix(reduced_bell_matrix);
    auto zero           = dd::Package::CVec{{0., 0.}, {0., 0.}, {0., 0.}, {0., 0.}};
    EXPECT_EQ(mat[2], zero);
    EXPECT_EQ(mat[3], zero);

    reduced_bell_matrix = dd->reduceGarbage(bell_matrix, {0b01});
    mat                 = dd->getMatrix(reduced_bell_matrix);
    EXPECT_EQ(mat[1], zero);
    EXPECT_EQ(mat[3], zero);

    reduced_bell_matrix = dd->reduceGarbage(bell_matrix, {0b10}, false);
    EXPECT_EQ(reduced_bell_matrix.p->e[1], dd::Package::DDzero);
    EXPECT_EQ(reduced_bell_matrix.p->e[3], dd::Package::DDzero);
}

TEST(DDPackageTest, InvalidMakeBasisState) {
    auto dd         = std::make_unique<dd::Package>();
    auto basisState = std::vector<dd::BasisStates>{dd::BasisStates::zero};
    auto nqubits    = 2;
    EXPECT_THROW(dd->makeBasisState(nqubits, basisState), std::invalid_argument);
}

TEST(DDPackageTest, InvalidDecRef) {
    auto dd = std::make_unique<dd::Package>();
    auto e  = dd->makeIdent(2);
    EXPECT_THROW(dd->decRef(e), std::runtime_error);
}
