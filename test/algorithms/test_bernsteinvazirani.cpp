#include "algorithms/BernsteinVazirani.hpp"
#include "gtest/gtest.h"

class BernsteinVazirani : public testing::Test {
protected:

	const unsigned int hInt = 2;
};

TEST_F(BernsteinVazirani, FunctionTest) {
	auto dd = std::make_unique<dd::Package>();
	std::unique_ptr<qc::BernsteinVazirani> qc;
	dd::Edge e{};

	// Create the QuantumCircuite with the hidden integer
	qc = std::make_unique<qc::BernsteinVazirani>(hInt);

	dd::Edge r = dd->multiply(e, dd->makeZeroState(qc->size));
	
	// Test the Values
	ASSERT_EQ(dd->getValueByPath(r, std::string(qc->size, '1')), (dd::ComplexValue{ 1, 0 }));
	ASSERT_EQ(dd->getValueByPath(r, std::string(qc->size, '0')), (dd::ComplexValue{ 0, 0 }));
}
