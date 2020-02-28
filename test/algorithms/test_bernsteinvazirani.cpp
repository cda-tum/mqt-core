#include "algorithms/BernsteinBazirani.hpp"
#include "gtest/gtest.h"

#include <string>

/* Copy & Paste*/
class BernsteinVazirani : public testing::TestWithParam<unsigned int> {
protected:
	void TearDown() override {}
	void SetUp() override {}
};

INSTANTIATE_TEST_SUITE_P(BernsteinVazirani, BernsteinVazirani,
	testing::Range((unsigned int)6, (unsigned int)256, 10),
	[](const testing::TestParamInfo<BernsteinVazirani::ParamType>& info) {
		// Generate names for test cases
		unsigned short nqubits = info.param;
		std::stringstream ss{};
		ss << nqubits << "_qubits";
		return ss.str();
	});

TEST_P(Entanglement, FunctionTest) {
	const unsigned short nq = GetParam();

	auto dd = std::make_unique<dd::Package>();
	std::unique_ptr<qc::BernsteinVazirani> qc;
	dd::Edge e{};

	ASSERT_NO_THROW({ qc = std::make_unique<qc::BernsteinVazirani>(nq); });
	ASSERT_NO_THROW({ e = qc->buildFunctionality(dd); });

	ASSERT_EQ(qc->getNops(), nq);
	dd::Edge r = dd->multiply(e, dd->makeZeroState(nq));

	ASSERT_EQ(dd->getValueByPath(r, std::string(nq, '0')), (dd::ComplexValue{ dd::SQRT_2, 0 }));
	ASSERT_EQ(dd->getValueByPath(r, std::string(nq, '2')), (dd::ComplexValue{ dd::SQRT_2, 0 }));
}