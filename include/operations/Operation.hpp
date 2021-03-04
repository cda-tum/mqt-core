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
#include "DDexport.h"

#define DEBUG_MODE_OPERATIONS 0

namespace qc {
	class QFRException : public std::invalid_argument {
		std::string msg;
	public:
		explicit QFRException(std::string  msg) : std::invalid_argument("QFR Exception"), msg(std::move(msg)) { }

		[[nodiscard]] const char *what() const noexcept override {
			return msg.c_str();
		}
	};

	using regnames_t=std::vector<std::pair<std::string, std::string>>;
	enum Format {
		Real, OpenQASM, GRCS, Qiskit, TFC, QC
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

	// Supported Operations
	enum OpType {
		None,
		// Standard Operations
		I, H, X, Y, Z, S, Sdag, T, Tdag, V, Vdag, U3, U2, Phase, SX, SXdag, RX, RY, RZ, SWAP, iSWAP, Peres, Peresdag,
		// Compound Operation
		Compound,
		// Non Unitary Operations
		Measure, Reset, Snapshot, ShowProbabilities, Barrier, Teleportation,
		// Classically-controlled Operation
		ClassicControlled
	};

	class Operation {
	protected:
		std::vector<unsigned short>       targets{};
		std::vector<Control>              controls{};
		std::array<fp, MAX_PARAMETERS>    parameter{ };
		
		unsigned short nqubits     = 0;
		OpType         type = None;         // Op type
		bool           multiTarget = false; // flag to distinguish multi target operations
		bool           controlled  = false; // flag to distinguish multi control operations
		char           name[MAX_STRING_LENGTH]{ };

		static bool isWholeQubitRegister(const regnames_t& reg, unsigned short start, unsigned short end) {
			return !reg.empty() && reg[start].first == reg[end].first
					&& (start == 0             || reg[start].first != reg[start - 1].first)
					&& (end   == reg.size() -1 || reg[end].first != reg[end   + 1].first);
		}

		static std::map<unsigned short, unsigned short> create_standard_permutation() {
			std::map<unsigned short, unsigned short> permutation{};
			for (unsigned short i=0; i < MAX_QUBITS; ++i)
				permutation.insert({i, i});
			return permutation;
		}
		static std::map<unsigned short, unsigned short> standardPermutation;

	public:
		Operation() = default;
		Operation(const Operation& op) = delete;
		Operation(Operation&& op) noexcept = default;
		Operation& operator=(const Operation& op) = delete;
		Operation& operator=(Operation&& op) noexcept = default;
		// Virtual Destructor
		virtual ~Operation() = default;

		// Getters
		[[nodiscard]] const std::vector<unsigned short>& getTargets() const {
			return targets;
		}
		std::vector<unsigned short>& getTargets() {
			return targets;
		}
		[[nodiscard]] size_t getNtargets() const {
			return targets.size();
		}

		[[nodiscard]] const std::vector<Control>& getControls() const {
			return controls;
		}
		std::vector<Control>& getControls() {
			return controls;
		}
		[[nodiscard]] size_t getNcontrols() const {
			return controls.size();
		}

		[[nodiscard]] unsigned short getNqubits() const {
			return nqubits; 
		}

		[[nodiscard]] const std::array<fp, MAX_PARAMETERS>& getParameter() const {
			return parameter;
		}
		std::array<fp, MAX_PARAMETERS>& getParameter() {
			return parameter;
		}

		[[nodiscard]] const char *getName() const {
			return name;
		}
		[[nodiscard]] virtual OpType getType() const {
			return type;
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

		virtual void setName();

		virtual void setGate(OpType g) {
			type = g;
			setName();
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
		virtual void setLine(std::array<short, MAX_QUBITS>& line, const std::map<unsigned short, unsigned short>& permutation) const;
		void setLine(std::array<short, MAX_QUBITS>& line) const {
			setLine(line, standardPermutation);
		}
		virtual void resetLine(std::array<short, MAX_QUBITS>& line, const std::map<unsigned short, unsigned short>& permutation) const;
		void resetLine(std::array<short, MAX_QUBITS>& line) const {
			resetLine(line, standardPermutation);
		}

		virtual dd::Edge getDD(std::unique_ptr<dd::Package>& dd, std::array<short, MAX_QUBITS>& line) const = 0;
		virtual dd::Edge getDD(std::unique_ptr<dd::Package>& dd, std::array<short, MAX_QUBITS>& line, std::map<unsigned short, unsigned short>& permutation) const = 0;

		virtual dd::Edge getInverseDD(std::unique_ptr<dd::Package>& dd, std::array<short, MAX_QUBITS>& line) const = 0;
		virtual dd::Edge getInverseDD(std::unique_ptr<dd::Package>& dd, std::array<short, MAX_QUBITS>& line, std::map<unsigned short, unsigned short>& permutation) const = 0;

		[[nodiscard]] inline virtual bool isUnitary() const {
			return true; 
		}

		[[nodiscard]] inline virtual bool isStandardOperation() const {
			return false;
		}

		[[nodiscard]] inline virtual bool isCompoundOperation() const {
			return false;
		}

		[[nodiscard]] inline virtual bool isNonUnitaryOperation() const {
			return false;
		}

		[[nodiscard]] inline virtual bool isClassicControlledOperation() const {
			return false;
		}

		[[nodiscard]] inline virtual bool isControlled() const  {
			return controlled; 
		}

		[[nodiscard]] inline virtual bool isMultiTarget() const {
			return multiTarget; 
		}

		[[nodiscard]] inline virtual bool actsOn(unsigned short i) const {
			for (const auto& t:targets) {
				if (t == i)
					return true;
			}

			if (std::any_of(controls.cbegin(), controls.cend(), [&i](const Control& c) { return c.qubit == i; }))
				return true;

			return false;
		}

		virtual std::ostream& print(std::ostream& os) const;
		virtual std::ostream& print(std::ostream& os, const std::map<unsigned short, unsigned short>& permutation) const;

		friend std::ostream& operator<<(std::ostream& os, const Operation& op) {
			return op.print(os);
		}

		virtual void dumpOpenQASM(std::ostream& of, const regnames_t& qreg, const regnames_t& creg) const = 0;
		virtual void dumpReal(std::ostream& of) const = 0;
		virtual void dumpQiskit(std::ostream& of, const regnames_t& qreg, const regnames_t& creg, const char* anc_reg_name) const = 0;
	};
}
#endif //INTERMEDIATEREPRESENTATION_OPERATION_H
