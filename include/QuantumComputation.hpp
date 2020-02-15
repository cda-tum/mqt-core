/*
 * This file is part of IIC-JKU QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#ifndef INTERMEDIATEREPRESENTATION_QUANTUMCOMPUTATION_H
#define INTERMEDIATEREPRESENTATION_QUANTUMCOMPUTATION_H

#include "StandardOperation.hpp"
#include "NonUnitaryOperation.hpp"
#include "ClassicControlledOperation.hpp"
#include "Parser.hpp"

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
#include <string>

#define DEBUG_MODE_QC 0

namespace qc {
	using reg            = std::pair<unsigned short, unsigned short>;
	using registerMap    = std::map<std::string, reg>;
	using permutationMap = std::map<unsigned short, unsigned short>;

	static constexpr char DEFAULT_QREG[2]{"q"};
	static constexpr char DEFAULT_CREG[2]{"c"};
	static constexpr char DEFAULT_ANCREG[4]{"anc"};
	static constexpr char DEFAULT_MCTREG[4]{"mct"};

	class QuantumComputation {

	protected:
		std::vector<std::unique_ptr<Operation>> ops{ };
		unsigned short nqubits      = 0;
		unsigned short nclassics    = 0;
		unsigned short nancillae    = 0;
		unsigned short max_controls = 0;
		std::string name;

		// reg[reg_name] = {start_index, length}
		registerMap qregs{ };
		registerMap cregs{ };
		registerMap ancregs{ };

		void importReal(std::istream& is);

		void readRealHeader(std::istream& is);

		void readRealGateDescriptions(std::istream& is);

		void importOpenQASM(std::istream& is);

		void importGRCS(std::istream& is);

		static void create_reg_array(const registerMap& regs, regnames_t& regnames, unsigned short defaultnumber, const char* defaultname);

		bool isIdleQubit(unsigned short i);

	public:
		QuantumComputation() = default;
		explicit QuantumComputation(unsigned short nqubits) {
			addQubitRegister(nqubits);
			addClassicalRegister(nqubits);
		}
		explicit QuantumComputation(const std::string& filename) {
			import(filename);
		}

		virtual ~QuantumComputation() = default;

		virtual  size_t getNops()                   const { return ops.size();	}
		unsigned short  getNqubits()                const { return nqubits + nancillae;	}
		unsigned short getNancillae()               const { return nancillae; }
		unsigned short getNqubitsWithoutAncillae()  const { return nqubits; }
		std::string     getName()                   const { return name;       }
		const registerMap& getQregs()               const { return qregs; }
		const registerMap& getCregs()               const { return cregs; }
		const registerMap& getANCregs()               const { return ancregs; }

		// initialLayout[physical_qubit] = logical_qubit
		permutationMap initialLayout{ };
		permutationMap outputPermutation{ };
		
		unsigned long long getNindividualOps() const;

		std::string getQubitRegister(unsigned short physical_qubit_index);
		unsigned short getHighestLogicalQubitIndex();
		std::pair<std::string, unsigned short> getQubitRegisterAndIndex(unsigned short physical_qubit_index);
		bool isAncilla(unsigned short i);
		void reduceAncillae(dd::Edge& e, std::unique_ptr<dd::Package>& dd);
		void reduceGarbage(dd::Edge& e, std::unique_ptr<dd::Package>& dd);
		dd::Edge createInitialMatrix(std::unique_ptr<dd::Package>& dd); // creates identity matrix, which is reduced with respect to the ancillary qubits

		void stripTrailingIdleQubits();

		void import(const std::string& filename);
		void import(const std::string& filename, Format format);
		void import(std::istream& is, Format format);

		// search through .qasm file and look for IO layout information of the form
		//      'i Q_i Q_j ... Q_k' meaning, e.g. q_0 is mapped to Q_i, q_1 to Q_j, etc.
		//      'o Q_i Q_j ... Q_k' meaning, e.g. q_0 is found at Q_i, q_1 at Q_j, etc.
		bool lookForOpenQASM_IO_Layout(std::istream& ifs);

		// optimize circuit by fusing CX-CX(-CX) gates
		void fuseCXtoSwap();

		// this function augments a given circuit by additional registers
		void addQubitRegister(unsigned short nq, const char* reg_name = DEFAULT_QREG);
		void addClassicalRegister(unsigned short nc, const char* reg_name = DEFAULT_CREG);
		void addAncillaryRegister(unsigned short nq, const char* reg_name = DEFAULT_ANCREG);

		// removes the a specific logical qubit and returns the index of the physical qubit in the initial layout
		// as well as the index of the removed physical qubit's output permutation
		// i.e., initialLayout[physical_qubit] = logical_qubit and outputPermutation[physicalQubit] = output_qubit
		std::pair<unsigned short, short> removeQubit(unsigned short logical_qubit_index);

		// adds physical qubit as ancillary qubit and gives it the appropriate output mapping
		void addAncillaryQubit(unsigned short physical_qubit_index, short output_qubit_index);
		// try to add logical qubit to circuit and assign it to physical qubit with certain output permutation value
		void addQubit(unsigned short logical_qubit_index, unsigned short physical_qubit_index, short output_qubit_index);

		void updateMaxControls(unsigned short ncontrols) {
			max_controls = std::max(ncontrols, max_controls);
		}

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

		std::ostream& printRegisters(std::ostream& os = std::cout);

		static std::ostream& printPermutationMap(const permutationMap& map, std::ostream& os = std::cout);

		virtual void dump(const std::string& filename, Format format);
		virtual void dump(const std::string& filename);

		virtual void reset() {
			ops.clear();
			nqubits = 0;
			nclassics = 0;
			nancillae = 0;
			qregs.clear();
			cregs.clear();
			ancregs.clear();
			initialLayout.clear();
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
			if (!ops.empty() && !op.isControlled() && !ops.back()->isControlled()) {
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
