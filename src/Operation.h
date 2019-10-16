//
// Created by Lukas Burgholzer on 25.09.19.
//

#ifndef INTERMEDIATEREPRESENTATION_OPERATION_H
#define INTERMEDIATEREPRESENTATION_OPERATION_H

#include <array>
#include <limits>
#include <cmath>
#include "DDpackage.h"

// Operation Constants
constexpr std::size_t MAX_QUBITS = 225; // Max. qubits supported
constexpr std::size_t MAX_PARAMETERS = 3; // Max. parameters of an operation
constexpr std::size_t MAX_STRING_LENGTH = 22; // Max. chars that fit in a short string (64bit machine)

class Operation {
protected:
	std::array<short,MAX_QUBITS> line{};
	unsigned short nqubits = 0;
	std::array<double, MAX_PARAMETERS> parameter{};
	char name[MAX_STRING_LENGTH]{};

public:
	Operation() = default;
	// Virtual Destructor
	virtual ~Operation() = default;

	// Getters
	const char *getName() const {
		return name;
	}
	const std::array<short, MAX_QUBITS>& getLine() const {
		return line;
	}
	const std::array<double, MAX_PARAMETERS>& getParameter() const {
		return parameter;
	}

	// Public Methods
	virtual dd::Edge getDD(dd::Package* dd) = 0;
	virtual dd::Edge getInverseDD(dd::Package *dd) = 0;
	inline virtual bool isMeasurement() const {return false;}

	// TODO: Function to dump representation in specific format
};


#endif //INTERMEDIATEREPRESENTATION_OPERATION_H
