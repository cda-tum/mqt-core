//
// Created by Lukas Burgholzer on 25.09.19.
//

#ifndef INTERMEDIATEREPRESENTATION_QUANTUMCOMPUTATION_H
#define INTERMEDIATEREPRESENTATION_QUANTUMCOMPUTATION_H

#include <vector>
#include <memory>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <map>

#include "Operations/StandardOperation.h"
#include "Operations/NonUnitaryOperation.h"
#include "Operations/ClassicControlledOperation.h"
#include "Parser.h"

namespace qc {

	enum Format {
		Real, OpenQASM
	};

	using registerMap = std::map<std::string, std::pair<unsigned short, unsigned short>>;
	using permutationMap = std::map<unsigned short, unsigned short>;

	class QuantumComputation {

	protected:
		std::vector<std::unique_ptr<Operation>> ops{ };
		unsigned short nqubits = 0;
		unsigned short nclassics = 0;

		registerMap qregs{ };
		registerMap cregs{ };

		permutationMap inputPermutation{ };
		permutationMap outputPermutation{ };

		void importReal(std::istream& is);

		void readRealHeader(std::istream& is);

		void readRealGateDescriptions(std::istream& is);

		void importOpenQASM(std::istream& is);

		void compareAndEmplace(std::vector<short>& controls, unsigned short target, Gate gate = X, fp lambda = 0.L, fp phi = 0.L, fp theta = 0.L);

	public:
		QuantumComputation() = default;
		explicit QuantumComputation(unsigned short nqubits): nqubits(nqubits) { }

		virtual ~QuantumComputation() = default;

		// Iterators (pass-through)
		auto begin() noexcept {
			return ops.begin();
		}

		auto begin() const noexcept {
			return ops.begin();
		}

		auto cbegin() const noexcept {
			return ops.cbegin();
		}

		auto end() noexcept {
			return ops.end();
		}

		auto end() const noexcept {
			return ops.end();
		}

		auto cend() const noexcept {
			return ops.cend();
		}

		auto rbegin() noexcept {
			return ops.rbegin();
		}

		auto rbegin() const noexcept {
			return ops.rbegin();
		}

		auto crbegin() const noexcept {
			return ops.crbegin();
		}

		auto rend() noexcept {
			return ops.rend();
		}

		auto rend() const noexcept {
			return ops.rend();
		}

		auto crend() const noexcept {
			return ops.crend();
		}

		// Capacity (pass-through)
		bool empty() const noexcept {
			return ops.empty();
		}

		size_t size() const noexcept {
			return ops.size();
		}

		size_t max_size() const noexcept {
			return ops.max_size();
		}

		void reserve(size_t new_cap) {
			ops.reserve(new_cap);
		}

		size_t capacity() const noexcept {
			return ops.capacity();
		}

		void shrink_to_fit() {
			ops.shrink_to_fit();
		}

		// Modifiers (pass-through)
		void clear() noexcept {
			ops.clear();
		}

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

		void pop_back() {
			return ops.pop_back();
		}

		void resize(size_t count) {
			ops.resize(count);
		}

		/// ----------------------------------------------------- ///

		virtual size_t getNops() const {
			return ops.size();
		}

		unsigned long long getNindividualOps() const {
			unsigned long long nops = 0;
			for (const auto& op: ops) {
				for (int i = 0; i < op->getNqubits(); ++i) {
					if (op->getLine()[i] == 2)
						nops++;
				}
			}

			return nops;
		};

		unsigned short getNqubits() const {
			return nqubits;
		}

		const permutationMap& getInputPermutation() const {
			return inputPermutation;
		}

		const permutationMap& getOutputPermutation() const {
			return outputPermutation;
		}

		void import(const std::string& filename, Format format) {
			auto ifs = std::ifstream(filename);
			if (!ifs.good()) {
				std::cerr << "Error opening/reading from file: " << filename << std::endl;
				exit(3);
			}

			switch (format) {
				case Real: importReal(ifs);
					break;
				case OpenQASM: importOpenQASM(ifs);
					break;
				default: std::cerr << "Format " << format << " not yet supported." << std::endl;
					exit(1);
			}
		}

		inline virtual std::ostream& print(std::ostream& os) const {
			size_t i = 0;
			for (const auto& op:ops) {
				os << std::setw(std::log10(ops.size())+1) << ++i << ": " << *op << "\n";
			}
			return os;
		}

		friend std::ostream& operator<<(std::ostream& os, const QuantumComputation& qc) {
			return qc.print(os);
		}

		virtual dd::Edge buildFunctionality(std::unique_ptr<dd::Package>& dd, int nops = -1);

		virtual dd::Edge simulate(const dd::Edge& in, std::unique_ptr<dd::Package>& dd);

		virtual std::ostream& printStatistics(std::ostream& os = std::cout) {
			os << "QC Statistics:\n";
			os << "\tn: " << nqubits << std::endl;
			os << "\tm: " << ops.size() << std::endl;
			os << "--------------" << std::endl;
			return os;
		}
	};
}
#endif //INTERMEDIATEREPRESENTATION_QUANTUMCOMPUTATION_H
