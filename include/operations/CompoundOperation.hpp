//
// Created by Lukas Burgholzer on 23.10.19.
//

#ifndef INTERMEDIATEREPRESENTATION_COMPOUNDOPERATION_H
#define INTERMEDIATEREPRESENTATION_COMPOUNDOPERATION_H

#include <vector>
#include <memory>

#include "Operation.hpp"

namespace qc {

	class CompoundOperation : public Operation {
	protected:
		std::vector<std::shared_ptr<Operation>> ops{ };
	public:
		explicit CompoundOperation(unsigned short nq) {
			strcpy(name, "Compound");
			nqubits = nq;
		}

		template<class T, class... Args>
		auto emplace_back(Args&& ... args) {
			parameter[0]++;
			return ops.emplace_back(std::make_shared<T>(args ...));
		}

		void setNqubits(unsigned short nq) override {
			nqubits = nq;
			for (auto& op:ops) {
				op->setNqubits(nq);
			}
		}

		dd::Edge getDD(std::unique_ptr<dd::Package>& dd, std::array<short, MAX_QUBITS>& line) override {
			dd::Edge e = dd->makeIdent(0, nqubits - 1);
			for (auto& op: ops) {
				e = dd->multiply(op->getDD(dd, line), e);
			}
			return e;
		}

		dd::Edge getInverseDD(std::unique_ptr<dd::Package>& dd, std::array<short, MAX_QUBITS>& line) override {
			dd::Edge e = dd->makeIdent(0, nqubits - 1);
			for (auto& op: ops) { 
				e = dd->multiply(e, op->getInverseDD(dd, line));
			}
			return e;
		}

		std::ostream& print(std::ostream& os) const override {
			for (unsigned long i = 0; i < ops.size() - 1; ++i) {
				os << *(ops[i]) << std::endl << "\t";
			}
			os << *(ops.back());

			return os;
		}
	};
}
#endif //INTERMEDIATEREPRESENTATION_COMPOUNDOPERATION_H
