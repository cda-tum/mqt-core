//
// Created by Lukas Burgholzer on 25.09.19.
//

#ifndef INTERMEDIATEREPRESENTATION_OPERATION_H
#define INTERMEDIATEREPRESENTATION_OPERATION_H

#include <array>
#include "DDpackage.h"

namespace qc {

	// Math Constants
	static constexpr fp PI = 3.141592653589793238462643383279502884197169399375105820974L;
	static constexpr fp PI_2 = 1.570796326794896619231321691639751442098584699687552910487L;
	static constexpr fp PI_4 = 0.785398163397448309615660845819875721049292349843776455243L;

	// Operation Constants
	constexpr std::size_t MAX_QUBITS = 225; // Max. qubits supported
	constexpr std::size_t MAX_PARAMETERS = 3; // Max. parameters of an operation
	constexpr std::size_t MAX_STRING_LENGTH = 20; // Ensure short-string-optimizations

	class Operation {
	protected:
		std::array<short, MAX_QUBITS> line{ };
		unsigned short nqubits = 0;
		std::array<fp, MAX_PARAMETERS> parameter{ };
		char name[MAX_STRING_LENGTH]{ };
		bool multiTarget = false; // flag to distinguish multi target operations
		bool controlled = false; // flag to distinguish multi control operations

		// TODO: std::string toReal(... )

		// TODO: std::string toQASM(... )

	public:
		Operation() = default;

		// Virtual Destructor
		virtual ~Operation() = default;

		// Getters
		const std::array<short, MAX_QUBITS>& getLine() const {
			return line;
		}
		std::array<short, MAX_QUBITS>& getLine() {
			return line;
		}

		unsigned short getNqubits() const { return nqubits; }

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
		virtual void setControlled(bool contr) { controlled = contr; }
		virtual void setMultiTarget(bool multi) { multiTarget = multi; }

		// Public Methods
		virtual dd::Edge getDD(std::unique_ptr<dd::Package>& dd) = 0;

		virtual dd::Edge getInverseDD(std::unique_ptr<dd::Package>& dd) = 0;

		inline virtual bool isUnitary() const { return true; }
		inline virtual bool isControlled() const  { return controlled; }
		inline virtual bool isMultiTarget() const { return multiTarget; }

		inline virtual std::ostream& print(std::ostream& os) const {
			const auto prec_before = std::cout.precision(20);

			os << name << "\t";

			for (int i = 0; i < nqubits; i++) {
				if (line[i] < 0) {
					os << "|\t";
				} else if (line[i] == 0) {
					os << "\033[31m" << "c\t" << "\033[0m";
				} else if (line[i] == 1) {
					os << "\033[32m" << "c\t" << "\033[0m";
				} else {
					os << "\033[1m\033[36m" << name[0] << name[1] << "\t\033[0m";
				}
			}

			bool isZero = true;
			for (size_t i = 0; i < MAX_PARAMETERS; ++i) {
				if (parameter[i] != 0.L)
					isZero = false;
			}
			if (!isZero) {
				os << "\tp: ";
				CN::printFormattedReal(os, parameter[0]);
				os << " ";
				for (size_t j = 1; j < MAX_PARAMETERS; ++j) {
					isZero = true;
					for (size_t i = j; i < MAX_PARAMETERS; ++i) {
						if (parameter[i] != 0.L)
							isZero = false;
					}
					if (isZero) break;
					CN::printFormattedReal(os, parameter[j]);
					os << " ";
				}
			}

			std::cout.precision(prec_before);

			return os;
		}

		friend std::ostream& operator<<(std::ostream& os, const Operation& op) {
			return op.print(os);
		}

		// TODO: void export(File dst, Format format)
	};
}
#endif //INTERMEDIATEREPRESENTATION_OPERATION_H
