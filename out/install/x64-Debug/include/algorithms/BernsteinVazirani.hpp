#ifndef QFR_BernsteinVazirani_H
#define QFR_BernsteinVazirani_H

#include <QuantumComputation.hpp>

namespace qc {
	class BernsteinVazirani : public QuantumComputation {
	protected:
		void setup(QuantumComputation& qc);

		void oracle(QuantumComputation& qc);

		void postProcessing(QuantumComputation& qc);

		void full_BernsteinVazirani(QuantumComputation& qc);

	public:
		unsigned int       hiddenInteger = 0;
		unsigned int	   size = -1;
		bool               includeSetup = true;

		explicit BernsteinVazirani(unsigned int hiddenInt);

		~BernsteinVazirani() override = default;

		dd::Edge simulate(const dd::Edge& in, std::unique_ptr<dd::Package>& dd) override;

		std::ostream& printStatistics(std::ostream& os = std::cout) override;

	};
}
#endif //QFR_ENTANGLEMENT_H
