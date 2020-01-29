/*
 * This file is part of IIC-JKU QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#ifndef INTERMEDIATEREPRESENTATION_OPERATION_H
#define INTERMEDIATEREPRESENTATION_OPERATION_H

#include <array>
#include <vector>
#include <map>
#include <iostream>
#include <iomanip>
#include <memory>
#include <fstream>
#include <cstring>

#include "DDpackage.h"

#define DEBUG_MODE_OPERATIONS 0
#define UNUSED(x) {(void) x;}


namespace qc {
	using regnames_t=std::vector<std::pair<std::string, std::string>>;
	enum Format {
		Real, OpenQASM, GRCS, Qiskit
	};	

	struct Control {
		enum controlType: bool {pos = true, neg = false};

		unsigned short qubit = 0;
		controlType    type  = pos;

		explicit Control(unsigned short qubit = 0, controlType type = pos): qubit(qubit), type(type) {};
	};

	// Math Constants
	static constexpr fp PI   = 3.141592653589793238462643383279502884197169399375105820974L;
	static constexpr fp PI_2 = 1.570796326794896619231321691639751442098584699687552910487L;
	static constexpr fp PI_4 = 0.785398163397448309615660845819875721049292349843776455243L;

	// Operation Constants
	constexpr std::size_t MAX_QUBITS        = dd::MAXN; // Max. qubits supported
	constexpr std::size_t MAX_PARAMETERS    =        3; // Max. parameters of an operation
	constexpr std::size_t MAX_STRING_LENGTH =       20; // Ensure short-string-optimizations

	static constexpr short LINE_TARGET      = dd::RADIX;
	static constexpr short LINE_CONTROL_POS = 1;
	static constexpr short LINE_CONTROL_NEG = 0;
	static constexpr short LINE_DEFAULT     = -1;

	class Operation {
	protected:
		//std::array<short, MAX_QUBITS>     line{ };
		std::vector<unsigned short>       targets;
		std::vector<Control>              controls;
		std::array<fp,    MAX_PARAMETERS> parameter{ };
		
		unsigned short nqubits     = 0;
		bool           multiTarget = false; // flag to distinguish multi target operations
		bool           controlled  = false; // flag to distinguish multi control operations
		char           name[MAX_STRING_LENGTH]{ };

		static bool isWholeQubitRegister(const regnames_t& reg, unsigned short start, unsigned short end) {
			return !reg.empty() && reg[start].first == reg[end].first
					&& (start == 0             || reg[start].first != reg[start - 1].first)
					&& (end   == reg.size() -1 || reg[end].first != reg[end   + 1].first);
		}

	public:
		Operation() = default;

		// Virtual Destructor
		virtual ~Operation() = default;

		// Getters
		const std::vector<unsigned short>& getTargets() const {
			return targets;
		}
		
		std::vector<unsigned short>& getTargets() {
			return targets;
		}

		const std::vector<Control>& getControls() const {
			return controls;
		}
		
		std::vector<Control>& getControls() {
			return controls;
		}

		unsigned short getNqubits() const { 
			return nqubits; 
		}

		const std::array<fp, MAX_PARAMETERS>& getParameter() const {
			return parameter;
		}
		
		std::array<fp, MAX_PARAMETERS>& getParameter() {
			return parameter;
		}

		const char *getName() const {
			return name;
		}

		// Setter
		virtual void setNqubits(unsigned short nq) {
			nqubits = nq;
		}

		virtual void setTargets(const std::vector<unsigned short>& t) {
			Operation::targets = t;
		}

		virtual void setControls(const std::vector<Control>& c) {
			Operation::controls = c;
		}

		virtual void setParameter(const std::array<fp, MAX_PARAMETERS>& p) {
			Operation::parameter = p;
		}
		
		virtual void setControlled(bool contr) { 
			controlled = contr; 
		}
		
		virtual void setMultiTarget(bool multi) { 
			multiTarget = multi; 
		}

		// Public Methods
		// The methods with a permutation parameter apply these operations according to the mapping specified by the permutation, e.g.
		//      if perm[0] = 1 and perm[1] = 0
		//      then cx 0 1 will be translated to cx perm[0] perm[1] == cx 1 0
		void setLine(std::array<short, MAX_QUBITS>& line) const;
		void setLine(std::array<short, MAX_QUBITS>& line, const std::map<unsigned short, unsigned short>& permutation) const;
		void resetLine(std::array<short, MAX_QUBITS>& line) const;
		void resetLine(std::array<short, MAX_QUBITS>& line, const std::map<unsigned short, unsigned short>& permutation) const;

		virtual dd::Edge getDD(std::unique_ptr<dd::Package>& dd, std::array<short, MAX_QUBITS>& line) const = 0;
		virtual dd::Edge getDD(std::unique_ptr<dd::Package>& dd, std::array<short, MAX_QUBITS>& line, std::map<unsigned short, unsigned short>& permutation) const = 0;

		virtual dd::Edge getInverseDD(std::unique_ptr<dd::Package>& dd, std::array<short, MAX_QUBITS>& line) const = 0;
		virtual dd::Edge getInverseDD(std::unique_ptr<dd::Package>& dd, std::array<short, MAX_QUBITS>& line, std::map<unsigned short, unsigned short>& permutation) const = 0;

		inline virtual bool isUnitary() const { 
			return true; 
		}

		inline virtual bool isControlled() const  { 
			return controlled; 
		}

		inline virtual bool isMultiTarget() const { 
			return multiTarget; 
		}

		inline virtual bool actsOn(unsigned short i) {
			for (const auto t:targets) {
				if (t == i)
					return true;
			}

			for (const auto c:controls) {
				if (c.qubit == i)
					return true;
			}
			return false;
		}

		virtual std::ostream& print(std::ostream& os) const;
		virtual std::ostream& print(std::ostream& os, const std::map<unsigned short, unsigned short>& permutation) const;

		friend std::ostream& operator<<(std::ostream& os, const Operation& op) {
			return op.print(os);
		}

		virtual void dumpOpenQASM(std::ofstream& of, const regnames_t& qreg, const regnames_t& creg) const { UNUSED(of); UNUSED(qreg); UNUSED(creg);
			std::cerr << "Dump of " << name << " operation to OpenQASM not yet supported" << std::endl;
		}
		virtual void dumpReal(std::ofstream& of) const { UNUSED(of);
			std::cerr << "Dump of " << name << " operation to Real not yet supported" << std::endl;
		}
		virtual void dumpQiskit(std::ofstream& of, const regnames_t& qreg, const regnames_t& creg, const char* anc_reg_name) const { UNUSED(of); UNUSED(qreg); UNUSED(creg); UNUSED(anc_reg_name);
			std::cerr << "Dump of " << name << " operation to Qiskit not yet supported" << std::endl;
		}
	};
}
#endif //INTERMEDIATEREPRESENTATION_OPERATION_H
