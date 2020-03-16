#ifndef QFR_BV_H
#define QFR_BV_H

#include <QuantumComputation.hpp>

namespace qc {
	class BernsteinVazirani : public QuantumComputation {
	protected:
		void setup(QuantumComputation& qc);

		void oracle(QuantumComputation& qc);

		void postProcessing(QuantumComputation& qc);

		void full_BernsteinVazirani(QuantumComputation& qc);

	public:
		unsigned long       hiddenInteger = 0;
		unsigned int	   size = -1;

		explicit BernsteinVazirani(unsigned long hiddenInt);

		~BernsteinVazirani() override = default;

		std::ostream& printStatistics(std::ostream& os = std::cout) override;

	};
}
#endif //QFR_BV_H
