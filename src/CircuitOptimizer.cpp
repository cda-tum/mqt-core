/*
 * This file is part of IIC-JKU QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#include "CircuitOptimizer.hpp"

namespace qc {

	void CircuitOptimizer::swapGateFusion(QuantumComputation& qc) {
		// print incoming circuit
		//qc.print(std::cout );
		//std::cout << std::endl;

		auto dag = DAG(qc.nqubits + qc.nancillae);

		for (auto & it : qc.ops) {
			if (!it->isUnitary()) {
				std::cerr << "Non unitary operation detected. This is currently not supported. Proceed with caution!" << std::endl;
				continue;
			}
			auto op = dynamic_cast<StandardOperation*>(it.get());
			if(!op) {
				auto compOp = dynamic_cast<CompoundOperation*>(it.get());
				if (compOp) {
					std::cerr << "Compund operation detected. This is currently not supported. Proceed with caution!" << std::endl;
					return;
				} else {
					throw QFRException("Unexpected operation encountered");
				}
			}

			// Operation is not a CNOT
			if (op->getGate() != X || op->getNcontrols() != 1 || op->getControls().at(0).type != Control::pos) {
				addToDag(dag, op);
				continue;
			}

			unsigned short control = op->getControls().at(0).qubit;
			unsigned short target = op->getTargets().at(0);

			// first operation
			if (dag.at(control).empty() || dag.at(target).empty()) {
				addToDag(dag, op);
				continue;
			}

			auto opC = dynamic_cast<StandardOperation*>(dag.at(control).front());
			auto opT = dynamic_cast<StandardOperation*>(dag.at(target).front());

			// previous operation is not a CNOT
			if (opC->getGate() != qc::X || opC->getNcontrols() != 1 ||
				opT->getGate() != qc::X || opT->getNcontrols() != 1) {
				addToDag(dag, op);
				continue;
			}

			auto opCcontrol = opC->getControls().at(0).qubit;
			auto opCtarget = opC->getTargets().at(0);
			auto opTcontrol = opT->getControls().at(0).qubit;
			auto opTtarget = opT->getTargets().at(0);

			// operation at control and target qubit are not the same
			if (opCcontrol != opTcontrol || opCtarget != opTtarget) {
				addToDag(dag, op);
				continue;
			}

			if (control == opCcontrol && target == opCtarget) {
				// elimination
				dag.at(control).pop_front();
				dag.at(target).pop_front();
				opC->setGate(I);
				opC->setControls({});
				op->setGate(I);
				op->setControls({});
			} else if (control == opCtarget && target == opCcontrol) {
				dag.at(control).pop_front();
				dag.at(target).pop_front();

				// replace with SWAP + CNOT
				opC->setGate(SWAP);
				opC->setTargets({target, control});
				opC->setControls({});
				addToDag(dag, opC);

				op->setTargets({control});
				op->setControls({Control(target)});
				addToDag(dag, op);
			} else {
				addToDag(dag, op);
			}
		}

		// delete the identities from circuit
		auto it=qc.ops.begin();
		while(it != qc.ops.end()) {
			auto op = dynamic_cast<StandardOperation*>((*it).get());
			if (op) {
				if (op->getGate() == I) {
					it = qc.ops.erase(it);
				} else {
					++it;
				}
			} else {
				++it;
			}
		}

		// print resulting circuit
		//qc.print(std::cout );
	}

	DAG CircuitOptimizer::constructDAG(QuantumComputation& qc) {
		auto dag = DAG(qc.nqubits + qc.nancillae);

		for (auto & it : qc.ops) {
			if (!it->isUnitary()) {
				std::cerr << "Non unitary operation detected. This is currently not supported. Proceed with caution!" << std::endl;
				continue;
			}
			auto op = dynamic_cast<StandardOperation*>(it.get());
			addToDag(dag, op);
		}

		bool allEmpty = false;
		while (!allEmpty) {
			allEmpty = true;
			for (int i = 0; i < qc.nqubits + qc.nancillae; ++i) {
				if(!dag.at(i).empty()) {
					allEmpty = false;
					for (const auto& c: dag.at(i).front()->getControls()){
						std::cout << c.qubit;
					}
					for (const auto& t: dag.at(i).front()->getTargets()){
						std::cout << t;
					}
					std::cout << dag.at(i).front()->getName();
					dag.at(i).pop_front();
					std::cout << "\t";
				} else {
					std::cout << "\t\t";
				}
			}
			std::cout << std::endl;
		}
		return dag;
	}

	void CircuitOptimizer::addToDag(DAG& dag, Operation *op) {
		for (const auto& control: op->getControls()) {
			dag.at(control.qubit).push_front(op);
		}
		for (const auto& target: op->getTargets()) {
			dag.at(target).push_front(op);
		}
	}

}
