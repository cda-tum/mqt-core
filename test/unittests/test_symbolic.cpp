#include "Expression.hpp"
#include "QuantumComputation.hpp"
#include "SymbolicQuantumComputation.hpp"
#include "dd/Definitions.hpp"
#include "operations/SymbolicOperation.hpp"

#include "gtest/gtest.h"
#include <iostream>
#include <memory>
#include <sstream>

using namespace dd;
using namespace qc;
using namespace sym;
class SymbolicTest: public ::testing::Test {
public:
    Variable x = Variable("x");
    Variable y = Variable("y");
    Variable z = Variable("z");

    Symbolic xMonom = Symbolic{Term<fp>{x}};
    Symbolic yMonom = Symbolic{Term<fp>{y}};
    Symbolic zMonom = Symbolic{Term<fp>{z}};

    std::unique_ptr<QuantumComputation> symQc = std::make_unique<QuantumComputation>(3);
    std::unique_ptr<QuantumComputation> qc    = std::make_unique<QuantumComputation>(3);

protected:
    // void TearDown override{};

    void SetUp() override{};
};

TEST_F(SymbolicTest, Gates) {
    auto xVal = PI_4 / 2;
    auto yVal = PI_4 / 4;
    auto zVal = PI / 3;

    std::unique_ptr<QuantumComputation> noRealSymQc =
            std::make_unique<QuantumComputation>(3);

    symQc->u3(0, {1_pc, 2_nc}, xMonom, yMonom, zMonom);
    symQc->u3(0, {1_pc}, xMonom, yMonom, zMonom);
    symQc->u3(0, xMonom, yMonom, zMonom);

    symQc->u2(0, {1_pc, 2_nc}, xMonom, yMonom);
    symQc->u2(0, {1_pc}, xMonom, yMonom);
    symQc->u2(0, xMonom, yMonom);

    symQc->phase(0, {1_pc, 2_nc}, xMonom);
    symQc->phase(0, {1_pc}, xMonom);
    symQc->phase(0, xMonom);

    symQc->rx(0, {1_pc, 2_nc}, xMonom);
    symQc->rx(0, {1_pc}, xMonom);
    symQc->rx(0, xMonom);

    symQc->ry(0, {1_pc, 2_nc}, xMonom);
    symQc->ry(0, {1_pc}, xMonom);
    symQc->ry(0, xMonom);

    symQc->rz(0, {1_pc, 2_nc}, xMonom);
    symQc->rz(0, {1_pc}, xMonom);
    symQc->rz(0, xMonom);

    // EXPECT_FALSE(symQc->isVariableFree());

    // normal circuit
    qc->u3(0, {1_pc, 2_nc}, xVal, yVal, zVal);
    qc->u3(0, {1_pc}, xVal, yVal, zVal);
    qc->u3(0, xVal, yVal, zVal);

    qc->u2(0, {1_pc, 2_nc}, xVal, yVal);
    qc->u2(0, {1_pc}, xVal, yVal);
    qc->u2(0, xVal, yVal);

    qc->phase(0, {1_pc, 2_nc}, xVal);
    qc->phase(0, {1_pc}, xVal);
    qc->phase(0, xVal);

    qc->rx(0, {1_pc, 2_nc}, xVal);
    qc->rx(0, {1_pc}, xVal);
    qc->rx(0, xVal);

    qc->ry(0, {1_pc, 2_nc}, xVal);
    qc->ry(0, {1_pc}, xVal);
    qc->ry(0, xVal);

    qc->rz(0, {1_pc, 2_nc}, xVal);
    qc->rz(0, {1_pc}, xVal);
    qc->rz(0, xVal);

    // symbolic but variable free circuit
    noRealSymQc->u3(0, {1_pc, 2_nc}, xVal, yVal, zVal);
    noRealSymQc->u3(0, {1_pc}, xVal, yVal, zVal);
    noRealSymQc->u3(0, xVal, yVal, zVal);

    noRealSymQc->u2(0, {1_pc, 2_nc}, xVal, yVal);
    noRealSymQc->u2(0, {1_pc}, xVal, yVal);
    noRealSymQc->u2(0, xVal, yVal);

    noRealSymQc->phase(0, {1_pc, 2_nc}, xVal);
    noRealSymQc->phase(0, {1_pc}, xVal);
    noRealSymQc->phase(0, xVal);

    noRealSymQc->rx(0, {1_pc, 2_nc}, xVal);
    noRealSymQc->rx(0, {1_pc}, xVal);
    noRealSymQc->rx(0, xVal);

    noRealSymQc->ry(0, {1_pc, 2_nc}, xVal);
    noRealSymQc->ry(0, {1_pc}, xVal);
    noRealSymQc->ry(0, xVal);

    noRealSymQc->rz(0, {1_pc, 2_nc}, xVal);
    noRealSymQc->rz(0, {1_pc}, xVal);
    noRealSymQc->rz(0, xVal);

    EXPECT_TRUE(noRealSymQc->isVariableFree());

    for (auto it1 = symQc->begin(), it2 = noRealSymQc->begin(); it1 != symQc->end() && it2 != noRealSymQc->end(); ++it1, ++it2) {
        const auto* symOp1 = dynamic_cast<SymbolicOperation*>(it1->get());
        const auto* symOp2 = dynamic_cast<SymbolicOperation*>(it2->get());
        // EXPECT_FALSE(symOp1->equalsSymbolic(*symOp1));
        EXPECT_FALSE((*it1)->equals(*(*it2)));
    }

    VariableAssignment assignment{{x, xVal}, {y, yVal}, {z, zVal}};

    symQc->instantiate(assignment);

    for (auto it1 = symQc->begin(), it2 = noRealSymQc->begin(); it1 != symQc->end() && it2 != noRealSymQc->end(); ++it1, ++it2) {
        const auto* symOp1 = dynamic_cast<SymbolicOperation*>(it1->get());
        const auto* symOp2 = dynamic_cast<SymbolicOperation*>(it2->get());
        // EXPECT_TRUE(symOp1->equalsSymbolic(*symOp2));
        EXPECT_TRUE((*it1)->equals(*(*it2)));
    }
}
