//
// Created by Lukas Burgholzer on 25.09.19.
//

#ifndef INTERMEDIATEREPRESENTATION_QUANTUMCOMPUTATION_H
#define INTERMEDIATEREPRESENTATION_QUANTUMCOMPUTATION_H

#include <vector>

#include "Operation.h"

class QuantumComputation {

protected:
	std::vector<Operation*> ops;

public:
	virtual ~QuantumComputation() = default;

};


#endif //INTERMEDIATEREPRESENTATION_QUANTUMCOMPUTATION_H
