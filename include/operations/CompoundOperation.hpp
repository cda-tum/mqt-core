/*
 * This file is part of IIC-JKU QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#ifndef INTERMEDIATEREPRESENTATION_COMPOUNDOPERATION_H
#define INTERMEDIATEREPRESENTATION_COMPOUNDOPERATION_H

#include "Operation.hpp"

namespace qc {

	class CompoundOperation : public Operation {
	protected:
		std::vector<std::unique_ptr<Operation>> ops{ };
	public:
		explicit CompoundOperation(unsigned short nq) {
			std::strcpy(name, "Compound operation:");
			nqubits = nq;
			type = Compound;
		}

		template<class T>
		void emplace_back(std::unique_ptr<T>& op) {
			parameter[0]++;
			ops.emplace_back(std::move(op));
		}

		template<class T, class... Args>
		void emplace_back(Args&& ... args) {
			parameter[0]++;
			ops.emplace_back(std::make_unique<T>(args ...));
		}

		void setNqubits(unsigned short nq) override {
			nqubits = nq;
			for (auto& op:ops) {
				op->setNqubits(nq);
			}
		}

		bool isCompoundOperation() const override {
			return true;
		}

		bool isNonUnitaryOperation() const override {
			bool isNonUnitary = false;
			for (const auto& op: ops) {
				isNonUnitary |= op->isNonUnitaryOperation();
			}
			return isNonUnitary;
		}

		dd::Edge getDD(std::unique_ptr<dd::Package>& dd, std::array<short, MAX_QUBITS>& line) const override {
			dd::Edge e = dd->makeIdent(0, short(nqubits - 1));
			for (auto& op: ops) {
				e = dd->multiply(op->getDD(dd, line), e);
			}
			return e;
		}

		dd::Edge getInverseDD(std::unique_ptr<dd::Package>& dd, std::array<short, MAX_QUBITS>& line) const override {
			dd::Edge e = dd->makeIdent(0, short(nqubits - 1));
			for (auto& op: ops) { 
				e = dd->multiply(e, op->getInverseDD(dd, line));
			}
			return e;
		}

		dd::Edge getDD(std::unique_ptr<dd::Package>& dd, std::array<short, MAX_QUBITS>& line, std::map<unsigned short, unsigned short>& permutation) const override {
			dd::Edge e = dd->makeIdent(0, short(nqubits - 1));
			for (auto& op: ops) {
				e = dd->multiply(op->getDD(dd, line, permutation), e);
			}
			return e;
		}

		dd::Edge getInverseDD(std::unique_ptr<dd::Package>& dd, std::array<short, MAX_QUBITS>& line, std::map<unsigned short, unsigned short>& permutation) const override {
			dd::Edge e = dd->makeIdent(0, short(nqubits - 1));
			for (auto& op: ops) {
				e = dd->multiply(e, op->getInverseDD(dd, line, permutation));
			}
			return e;
		}

		std::ostream& print(std::ostream& os) const override {
			return print(os, standardPermutation);
		}

		std::ostream& print(std::ostream& os, const std::map<unsigned short, unsigned short>& permutation) const override {
			os << name;
			for (const auto & op : ops) {
				os << std::endl << "\t";
				op->print(os, permutation);
			}

			return os;
		}

		bool actsOn(unsigned short i) override {
			for (const auto& op: ops) {
				if(op->actsOn(i))
					return true;
			}
			return false;
		}

		void dumpOpenQASM(std::ostream& of, const regnames_t& qreg, const regnames_t& creg) const override {
			for (auto& op: ops) {
				op->dumpOpenQASM(of, qreg, creg);
			}
		}

		void dumpReal(std::ostream& of) const override {
			for (auto& op: ops) {
				op->dumpReal(of);
			}
		}

		void dumpQiskit(std::ostream& of, const regnames_t& qreg, const regnames_t& creg, const char *anc_reg_name) const override {
			for (auto& op: ops) {
				op->dumpQiskit(of, qreg, creg, anc_reg_name);
			}
		}

		/**
		 * Pass-Through
		 */

		// Iterators (pass-through)
		auto begin()            noexcept { return ops.begin();   }
		auto begin()      const noexcept { return ops.begin();   }
		auto cbegin()     const noexcept { return ops.cbegin();  }
		auto end()              noexcept { return ops.end();     }
		auto end()        const noexcept { return ops.end();	  }
		auto cend()       const noexcept { return ops.cend();	  }
		auto rbegin()           noexcept { return ops.rbegin();  }
		auto rbegin()     const noexcept { return ops.rbegin();  }
		auto crbegin()    const noexcept { return ops.crbegin(); }
		auto rend()             noexcept { return ops.rend();    }
		auto rend()       const noexcept {	return ops.rend();    }
		auto crend()      const noexcept { return ops.crend();   }

		// Capacity (pass-through)
		bool   empty()    const noexcept { return ops.empty();    }
		size_t size()     const noexcept { return ops.size();     }
		size_t max_size() const noexcept { return ops.max_size(); }
		size_t capacity() const noexcept { return ops.capacity(); }

		void reserve(size_t new_cap)     { ops.reserve(new_cap);  }
		void shrink_to_fit()             { ops.shrink_to_fit();   }

		// Modifiers (pass-through)
		void clear()              noexcept { ops.clear();           }
		void pop_back()                    { return ops.pop_back(); }
		void resize(size_t count)          { ops.resize(count);     }
		std::vector<std::unique_ptr<Operation>>::iterator erase( std::vector<std::unique_ptr<Operation>>::const_iterator pos ) { return ops.erase(pos); }
		std::vector<std::unique_ptr<Operation>>::iterator erase( std::vector<std::unique_ptr<Operation>>::const_iterator first, std::vector<std::unique_ptr<Operation>>::const_iterator last ) { return ops.erase(first, last); }
	};
}
#endif //INTERMEDIATEREPRESENTATION_COMPOUNDOPERATION_H
