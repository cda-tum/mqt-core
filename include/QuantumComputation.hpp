/*
 * This file is part of IIC-JKU QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#ifndef INTERMEDIATEREPRESENTATION_QUANTUMCOMPUTATION_H
#define INTERMEDIATEREPRESENTATION_QUANTUMCOMPUTATION_H

#include "operations/StandardOperation.hpp"
#include "operations/NonUnitaryOperation.hpp"
#include "operations/ClassicControlledOperation.hpp"
#include "parsers/qasm_parser/Parser.hpp"

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
#include <locale>

#define DEBUG_MODE_QC 0

namespace qc {
	using reg            = std::pair<unsigned short, unsigned short>;
	using registerMap    = std::map<std::string, reg, std::greater<>>;
	using permutationMap = std::map<unsigned short, unsigned short>;

	static constexpr char DEFAULT_QREG[2]{"q"};
	static constexpr char DEFAULT_CREG[2]{"c"};
	static constexpr char DEFAULT_ANCREG[4]{"anc"};
	static constexpr char DEFAULT_MCTREG[4]{"mct"};

	class CircuitOptimizer;

	class QuantumComputation {
		friend class CircuitOptimizer;

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

		void importOpenQASM(std::istream& is);
		void importReal(std::istream& is);
		int readRealHeader(std::istream& is);
		void readRealGateDescriptions(std::istream& is, int line);
		void importTFC(std::istream& is);
		int readTFCHeader(std::istream& is, std::map<std::string, unsigned short>& varMap);
		void readTFCGateDescriptions(std::istream& is, int line, std::map<std::string, unsigned short>& varMap);
		void importQC(std::istream& is);
		int readQCHeader(std::istream& is, std::map<std::string, unsigned short>& varMap);
		void readQCGateDescriptions(std::istream& is, int line, std::map<std::string, unsigned short>& varMap);
		void importGRCS(std::istream& is);

		static void printSortedRegisters(const registerMap& regmap, const std::string& identifier, std::ostream& of);
		static void consolidateRegister(registerMap& regs);

		static void create_reg_array(const registerMap& regs, regnames_t& regnames, unsigned short defaultnumber, const char* defaultname);

		unsigned short getSmallestAncillary() const {
			for (size_t i=0; i<ancillary.size(); ++i) {
				if (ancillary.test(i))
					return i;
			}
			return ancillary.size();
		}

		unsigned short getSmallestGarbage() const {
			for (size_t i=0; i<garbage.size(); ++i) {
				if (garbage.test(i))
					return i;
			}
			return garbage.size();
		}
		bool isLastOperationOnQubit(decltype(ops.begin())& opIt) {
			auto end = ops.end();
			return isLastOperationOnQubit(opIt, end);
		}


	public:
		QuantumComputation() = default;
		explicit QuantumComputation(unsigned short nqubits) {
			addQubitRegister(nqubits);
			addClassicalRegister(nqubits);
		}
		explicit QuantumComputation(const std::string& filename) {
			import(filename);
		}
		QuantumComputation(const QuantumComputation& qc) = delete;
		QuantumComputation(QuantumComputation&& qc) noexcept = default;
		QuantumComputation& operator=(const QuantumComputation& qc) = delete;
		QuantumComputation& operator=(QuantumComputation&& qc) noexcept = default;
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

		std::bitset<MAX_QUBITS> ancillary{};
		std::bitset<MAX_QUBITS> garbage{};

		unsigned long long getNindividualOps() const;

		std::string getQubitRegister(unsigned short physical_qubit_index);
		std::string getClassicalRegister(unsigned short classical_index);
		static unsigned short getHighestLogicalQubitIndex(const permutationMap& map);
		unsigned short getHighestLogicalQubitIndex() const { return getHighestLogicalQubitIndex(initialLayout); };
		std::pair<std::string, unsigned short> getQubitRegisterAndIndex(unsigned short physical_qubit_index);
		std::pair<std::string, unsigned short> getClassicalRegisterAndIndex(unsigned short classical_index);

		unsigned short getIndexFromQubitRegister(const std::pair<std::string, unsigned short>& qubit);
		unsigned short getIndexFromClassicalRegister(const std::pair<std::string, unsigned short>& clbit);
		bool isIdleQubit(unsigned short physical_qubit);
		static bool isLastOperationOnQubit(decltype(ops.begin())& opIt, decltype(ops.end())& end);
		bool physicalQubitIsAncillary(unsigned short physical_qubit_index);
		bool logicalQubitIsAncillary(unsigned short logical_qubit_index) const { return ancillary.test(logical_qubit_index); }
		void setLogicalQubitAncillary(unsigned short logical_qubit_index) { ancillary.set(logical_qubit_index); }
		dd::Edge reduceAncillae(dd::Edge& e, std::unique_ptr<dd::Package>& dd, bool regular = true);
		dd::Edge reduceAncillaeRecursion(dd::Edge& e, std::unique_ptr<dd::Package>& dd, unsigned short lowerbound, bool regular = true);
		bool logicalQubitIsGarbage(unsigned short logical_qubit_index) const { return garbage.test(logical_qubit_index); }
		void setLogicalQubitGarbage(unsigned short logical_qubit_index) { garbage.set(logical_qubit_index); }
		// works for reversible circuits --- to be tested for quantum circuits
		dd::Edge reduceGarbage(dd::Edge& e, std::unique_ptr<dd::Package>& dd, bool regular = true);
		dd::Edge reduceGarbageRecursion(dd::Edge& e, std::unique_ptr<dd::Package>& dd, unsigned short lowerbound, bool regular = true);
		dd::Edge createInitialMatrix(std::unique_ptr<dd::Package>& dd); // creates identity matrix, which is reduced with respect to the ancillary qubits

		/// strip away qubits with no operations applied to them and which do not pop up in the output permutation
		/// \param force if true, also strip away idle qubits occurring in the output permutation
		void stripIdleQubits(bool force = false, bool reduceIOpermutations = true);
		// apply swaps 'on' DD in order to change 'from' to 'to'
		// where |from| >= |to|
		static void changePermutation(dd::Edge& on, qc::permutationMap& from, const qc::permutationMap& to, std::array<short, qc::MAX_QUBITS>& line, std::unique_ptr<dd::Package>& dd, bool regular = true);

		void import(const std::string& filename);
		void import(const std::string& filename, Format format);
		void import(std::istream& is, Format format) {
			import(std::move(is), format);
		}
		void import(std::istream&& is, Format format);
		void initializeIOMapping();

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
		virtual dd::Complex getEntry(std::unique_ptr<dd::Package>& dd, dd::Edge e, unsigned long long i, unsigned long long j);

		/**
		 * printing
		 */ 
		virtual std::ostream& print(std::ostream& os) const;

		friend std::ostream& operator<<(std::ostream& os, const QuantumComputation& qc) { return qc.print(os); }

		virtual std::ostream& printMatrix(std::unique_ptr<dd::Package>& dd, dd::Edge e, std::ostream& os);

		static void printBin(unsigned long long n, std::stringstream& ss);

		virtual std::ostream& printCol(std::unique_ptr<dd::Package>& dd, dd::Edge e, unsigned long long j, std::ostream& os);

		virtual std::ostream& printVector(std::unique_ptr<dd::Package>& dd, dd::Edge e, std::ostream& os);

		virtual std::ostream& printStatistics(std::ostream& os);

		std::ostream& printRegisters(std::ostream& os = std::cout);

		static std::ostream& printPermutationMap(const permutationMap& map, std::ostream& os = std::cout);

		virtual void dump(const std::string& filename, Format format);
		virtual void dump(const std::string& filename);
		virtual void dump(std::ostream& of, Format format) {
			dump(std::move(of), format);
		}
		virtual void dump(std::ostream&& of, Format format);
		virtual void dumpOpenQASM(std::ostream& of);

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
		std::vector<std::unique_ptr<Operation>>::iterator erase( std::vector<std::unique_ptr<Operation>>::const_iterator pos ) { return ops.erase(pos); }
		std::vector<std::unique_ptr<Operation>>::iterator erase( std::vector<std::unique_ptr<Operation>>::const_iterator first, std::vector<std::unique_ptr<Operation>>::const_iterator last ) { return ops.erase(first, last); }
		
		template<class T>
		void push_back(const T& op) {
			if (!ops.empty() && !op.isControlled() && !ops.back()->isControlled()) {
				std::cerr << op.getName() << std::endl;
			}

			ops.push_back(std::make_unique<T>(op));
		}

		template<class T, class... Args>
		void emplace_back(Args&& ... args) {
			ops.emplace_back(std::make_unique<T>(args ...));
		}

		template<class T>
		std::vector<std::unique_ptr<Operation>>::iterator insert(std::vector<std::unique_ptr<Operation>>::const_iterator pos, T&& op) { return ops.insert(pos, std::forward<T>(op)); }

	};
}
#endif //INTERMEDIATEREPRESENTATION_QUANTUMCOMPUTATION_H
