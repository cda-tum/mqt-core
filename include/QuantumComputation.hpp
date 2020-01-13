/*
 * This file is part of IIC-JKU QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#ifndef INTERMEDIATEREPRESENTATION_QUANTUMCOMPUTATION_H
#define INTERMEDIATEREPRESENTATION_QUANTUMCOMPUTATION_H

#include <vector>
#include <memory>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <map>
#include <algorithm>
#include <regex>
#include <limits>

#include "StandardOperation.hpp"
#include "NonUnitaryOperation.hpp"
#include "ClassicControlledOperation.hpp"
#include "Parser.hpp"

namespace qc {
	using registerMap    = std::map<std::string, std::pair<unsigned short, unsigned short>>;
	using permutationMap = std::map<unsigned short, unsigned short>;

	static constexpr char DEFAULT_QREG = 'q';
	static constexpr char DEFAULT_CREG = 'c';

	class QuantumComputation {

	protected:
		std::vector<std::unique_ptr<Operation>> ops{ };
		unsigned short nqubits   = 0;
		unsigned short nclassics = 0;
		std::string name;

		registerMap qregs{ };
		registerMap cregs{ };

		permutationMap inputPermutation{ };
		permutationMap outputPermutation{ };

		void importReal(std::istream& is);

		void readRealHeader(std::istream& is);

		void readRealGateDescriptions(std::istream& is);

		void importOpenQASM(std::istream& is);

		void importGRCS(std::istream& is, const std::string& filename);

		//void compareAndEmplace(std::vector<short>& controls, unsigned short target, Gate gate = X, fp lambda = 0.L, fp phi = 0.L, fp theta = 0.L);
		void create_reg_array(const registerMap& regs, regnames_t& regnames, unsigned short defaultnumber, char defaultname);
	public:
		QuantumComputation() = default;
		explicit QuantumComputation(unsigned short nqubits): nqubits(nqubits) { }

		virtual ~QuantumComputation() = default;

		virtual  size_t getNops()    const { return ops.size();	}
		unsigned short  getNqubits() const { return nqubits;	}
		std::string     getName()    const { return name;       }

		const permutationMap& getInputPermutation()  const { return inputPermutation; }
		const permutationMap& getOutputPermutation() const { return outputPermutation; }
		
		unsigned long long getNindividualOps() const;

		void import(const std::string& filename, Format format);


		virtual dd::Edge buildFunctionality(std::unique_ptr<dd::Package>& dd);

		virtual dd::Edge simulate(const dd::Edge& in, std::unique_ptr<dd::Package>& dd);

		/// Obtain vector/matrix entry for row i (and column j). Does not include common factor e.w!
		/// \param dd package to use
		/// \param e vector/matrix dd
		/// \param i row index
		/// \param j column index
		/// \return temporary complex value representing the vector/matrix entry
		virtual dd::Complex getEntry(std::unique_ptr<dd::Package>& dd, dd::Edge e, unsigned long long i, unsigned long long j=0);

		/**
		 * printing
		 */ 
		virtual std::ostream& print(std::ostream& os = std::cout) const;

		friend std::ostream& operator<<(std::ostream& os, const QuantumComputation& qc) { return qc.print(os); }

		virtual std::ostream& printMatrix(std::unique_ptr<dd::Package>& dd, dd::Edge e, std::ostream& os = std::cout);

		static void printBin(unsigned long long n, std::stringstream& ss);

		virtual std::ostream& printCol(std::unique_ptr<dd::Package>& dd, dd::Edge e, unsigned long long j=0, std::ostream& os = std::cout);

		virtual std::ostream& printVector(std::unique_ptr<dd::Package>& dd, dd::Edge e, std::ostream& os = std::cout);

		virtual std::ostream& printStatistics(std::ostream& os = std::cout);

		virtual void dump(const std::string& filename, Format format);

		virtual void reset() {
			ops.clear();
			nqubits = 0;
			nclassics = 0;
			qregs.clear();
			cregs.clear();
			inputPermutation.clear();
			outputPermutation.clear();
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
		
		template<class T>
		void push_back(const T& op) {
			if (ops.size() >= 1 && !op.isControlled() && !ops.back()->isControlled()) {
				std::cerr << op.getName() << std::endl;
			}

			ops.push_back(std::make_unique<T>(op));
		}

		template<class T, class... Args>
		auto emplace_back(Args&& ... args) { 
			return ops.emplace_back(std::make_unique<T>(args ...)); 
		}
		
	};
}
#endif //INTERMEDIATEREPRESENTATION_QUANTUMCOMPUTATION_H
