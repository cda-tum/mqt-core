/*
 * This file is part of IIC-JKU QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#include "CircuitOptimizer.hpp"

namespace qc {
	const std::map<qc::OpType, qc::OpType> CircuitOptimizer::inverseMap = {
			{qc::I, qc::I},
			{qc::X, qc::X},
			{qc::Y, qc::Y},
			{qc::Z, qc::Z},
			{qc::H, qc::H},
			{qc::S, qc::Sdag},
			{qc::Sdag, qc::S},
			{qc::T, qc::Tdag},
			{qc::Tdag, qc::T}
	};

	void CircuitOptimizer::removeIdentities(QuantumComputation& qc) {
		// delete the identities from circuit
		auto it=qc.ops.begin();
		while(it != qc.ops.end()) {
			if ((*it)->isStandardOperation()) {
				if ((*it)->getType() == I) {
					it = qc.ops.erase(it);
				} else {
					++it;
				}
			} else if ((*it)->isCompoundOperation()) {
				auto compOp = dynamic_cast<qc::CompoundOperation*>((*it).get());
				auto cit=compOp->cbegin();
				while (cit != compOp->cend()) {
					auto cop = cit->get();
					if (cop->getType()== qc::I) {
						cit = compOp->erase(cit);
					} else {
						++cit;
					}
				}
				if (compOp->empty()) {
					it = qc.ops.erase(it);
				} else {
					if (compOp->size() == 1) {
						// CompoundOperation has degraded to single Operation
						(*it) = std::move(*(compOp->begin()));
					}
					++it;
				}
			} else {
				++it;
			}
		}
	}


	void CircuitOptimizer::swapGateFusion(QuantumComputation& qc) {
		// print incoming circuit
		//qc.print(std::cout );
		//std::cout << std::endl;
		unsigned short highest_physical_qubit = 0;
		for (const auto& q: qc.initialLayout) {
			if (q.first > highest_physical_qubit)
				highest_physical_qubit = q.first;
		}

		auto dag = DAG(highest_physical_qubit + 1);

		for (auto & it : qc.ops) {
			if (!it->isStandardOperation()) {
				// compound operations are added "as-is"
				if (it->isCompoundOperation()) {
					std::cerr << "Compound operation detected. This is currently not supported. Proceed with caution!" << std::endl;
					for (int i = 0; i < it->getNqubits(); ++i) {
						if (it->actsOn(i)) {
							dag.at(i).push_front(&it);
						}
					}
					continue;
				} else if (it->getType() == qc::Measure) {
					for (const auto& c: it->getControls()) {
						dag.at(c.qubit).push_front(&it);
					}
					continue;
				} else if (it->getType() == qc::Barrier || it->getType() == qc::Reset) {
					for (const auto& b: it->getTargets()) {
						dag.at(b).push_front(&it);
					}
					continue;
				} else if (it->isClassicControlledOperation()) {
					auto op = dynamic_cast<ClassicControlledOperation *>(it.get())->getOperation();
					for (const auto& control: op->getControls()) {
						dag.at(control.qubit).push_front(&it);
					}
					for (const auto& target: op->getTargets()) {
						dag.at(target).push_front(&it);
					}
					continue;
				} else {
					throw QFRException("Unexpected operation encountered");
				}
			}

			// Operation is not a CNOT
			if (it->getType() != X || it->getNcontrols() != 1 || it->getControls().at(0).type != Control::pos) {
				addToDag(dag, &it);
				continue;
			}

			unsigned short control = it->getControls().at(0).qubit;
			unsigned short target = it->getTargets().at(0);

			// first operation
			if (dag.at(control).empty() || dag.at(target).empty()) {
				addToDag(dag, &it);
				continue;
			}

			auto opC = dag.at(control).front();
			auto opT = dag.at(target).front();

			// previous operation is not a CNOT
			if ((*opC)->getType() != qc::X || (*opC)->getNcontrols() != 1 || (*opC)->getControls().at(0).type != Control::pos ||
			(*opT)->getType() != qc::X || (*opT)->getNcontrols() != 1 || (*opT)->getControls().at(0).type != Control::pos) {
				addToDag(dag, &it);
				continue;
			}

			auto opCcontrol = (*opC)->getControls().at(0).qubit;
			auto opCtarget = (*opC)->getTargets().at(0);
			auto opTcontrol = (*opT)->getControls().at(0).qubit;
			auto opTtarget = (*opT)->getTargets().at(0);

			// operation at control and target qubit are not the same
			if (opCcontrol != opTcontrol || opCtarget != opTtarget) {
				addToDag(dag, &it);
				continue;
			}

			if (control == opCcontrol && target == opCtarget) {
				// elimination
				dag.at(control).pop_front();
				dag.at(target).pop_front();
				(*opC)->setGate(I);
				(*opC)->setControls({});
				it->setGate(I);
				it->setControls({});
			} else if (control == opCtarget && target == opCcontrol) {
				dag.at(control).pop_front();
				dag.at(target).pop_front();

				// replace with SWAP + CNOT
				(*opC)->setGate(SWAP);
				(*opC)->setTargets({target, control});
				(*opC)->setControls({});
				addToDag(dag, opC);

				it->setTargets({control});
				it->setControls({Control(target)});
				addToDag(dag, &it);
			} else {
				addToDag(dag, &it);
			}
		}

		removeIdentities(qc);

		// print resulting circuit
		//qc.print(std::cout );
	}

	DAG CircuitOptimizer::constructDAG(QuantumComputation& qc) {
		unsigned short highest_physical_qubit = 0;
		for (const auto& q: qc.initialLayout) {
			if (q.first > highest_physical_qubit)
				highest_physical_qubit = q.first;
		}

		auto dag = DAG(highest_physical_qubit + 1);

		for (auto & it : qc.ops) {
			if (!it->isStandardOperation()) {
				// compound operations are added "as-is"
				if (it->isCompoundOperation()) {
					for (int i = 0; i < it->getNqubits(); ++i) {
						if (it->actsOn(i)) {
							dag.at(i).push_front(&it);
						}
					}
					continue;
				} else if (it->getType() == qc::Measure) {
					for (const auto& c: it->getControls()) {
						dag.at(c.qubit).push_front(&it);
					}
					continue;
				} else if (it->getType() == qc::Barrier || it->getType() == qc::Reset) {
					for (const auto& b: it->getTargets()) {
						dag.at(b).push_front(&it);
					}
					continue;
				} else if (it->isClassicControlledOperation()) {
					auto op = dynamic_cast<ClassicControlledOperation *>(it.get())->getOperation();
					for (const auto& control: op->getControls()) {
						dag.at(control.qubit).push_front(&it);
					}
					for (const auto& target: op->getTargets()) {
						dag.at(target).push_front(&it);
					}
					continue;
				} else {
					throw QFRException("Unexpected operation encountered");
				}
			} else {
				addToDag(dag, &it);
			}
		}
		return dag;
	}

	void CircuitOptimizer::addToDag(DAG& dag, std::unique_ptr<Operation> *op) {
		for (const auto& control: (*op)->getControls()) {
			dag.at(control.qubit).push_front(op);
		}
		for (const auto& target: (*op)->getTargets()) {
			dag.at(target).push_front(op);
		}
	}

	void CircuitOptimizer::singleQubitGateFusion(QuantumComputation& qc) {
		unsigned short highest_physical_qubit = 0;
		for (const auto& q: qc.initialLayout) {
			if (q.first > highest_physical_qubit)
				highest_physical_qubit = q.first;
		}

		auto dag = DAG(highest_physical_qubit + 1);

		for (auto & it : qc.ops) {
			if (!it->isStandardOperation()) {
				// compound operations are added "as-is"
				if (it->isCompoundOperation()) {
					for (int i = 0; i < it->getNqubits(); ++i) {
						if (it->actsOn(i)) {
							dag.at(i).push_front(&it);
						}
					}
					continue;
				} else if (it->getType() == qc::Measure) {
					for (const auto& c: it->getControls()) {
						dag.at(c.qubit).push_front(&it);
					}
					continue;
				} else if (it->getType() == qc::Barrier || it->getType() == qc::Reset) {
					for (const auto& b: it->getTargets()) {
						dag.at(b).push_front(&it);
					}
					continue;
				} else if (it->isClassicControlledOperation()) {
					auto op = dynamic_cast<ClassicControlledOperation *>(it.get())->getOperation();
					for (const auto& control: op->getControls()) {
						dag.at(control.qubit).push_front(&it);
					}
					for (const auto& target: op->getTargets()) {
						dag.at(target).push_front(&it);
					}
					continue;
				} else {
					throw QFRException("Unexpected operation encountered");
				}
			}

			// not a single qubit operation TODO: multiple targets could also be considered here
			if (!it->getControls().empty() || it->getTargets().size() > 1) {
				addToDag(dag, &it);
				continue;
			}

			auto target = it->getTargets().at(0);

			// first operation
			if (dag.at(target).empty()) {
				addToDag(dag, &it);
				continue;
			}

			auto dagQubit = dag.at(target);
			auto op = dagQubit.front();

			// no single qubit op to fuse with operation
			if (!(*op)->isCompoundOperation() && (!(*op)->getControls().empty() || (*op)->getTargets().size() > 1)) {
				addToDag(dag, &it);
				continue;
			}

			// compound operation
			if ((*op)->isCompoundOperation()) {
				auto compop = dynamic_cast<CompoundOperation*>(op->get());

				// check if compound operation contains non-single-qubit gates
				unsigned short involvedQubits = 0;
				for (size_t q=0; q<dag.size(); ++q) {
					if (compop->actsOn(q))
						++involvedQubits;
				}
				if (involvedQubits>1) {
					addToDag(dag, &it);
					continue;
				}

				// check if inverse
				auto lastop = (--(compop->end()));
				auto inverseIt = inverseMap.find((*lastop)->getType());
				// check if current operation is the inverse of the previous operation
				if (inverseIt != inverseMap.end() && it->getType() == inverseIt->second) {
					compop->pop_back();
					it->setGate(qc::I);
				} else {
					compop->emplace_back<StandardOperation>(
							it->getNqubits(),
							it->getTargets().at(0),
							it->getType(),
							it->getParameter().at(0),
							it->getParameter().at(1),
							it->getParameter().at(2));
					it->setGate(I);
				}

				continue;
			}

			// single qubit op

			// check if current operation is the inverse of the previous operation
			auto inverseIt = inverseMap.find((*op)->getType());
			if (inverseIt != inverseMap.end() && it->getType() == inverseIt->second) {
				(*op)->setGate(qc::I);
				it->setGate(qc::I);
			} else {
				auto compop = std::make_unique<CompoundOperation>(it->getNqubits());
				compop->emplace_back<StandardOperation>(
						(*op)->getNqubits(),
						(*op)->getTargets().at(0),
						(*op)->getType(),
						(*op)->getParameter().at(0),
						(*op)->getParameter().at(1),
						(*op)->getParameter().at(2));
				compop->emplace_back<StandardOperation>(
						it->getNqubits(),
						it->getTargets().at(0),
						it->getType(),
						it->getParameter().at(0),
						it->getParameter().at(1),
						it->getParameter().at(2));
				it->setGate(I);
				(*op) = std::move(compop);
				dag.at(target).push_front(op);
			}
		}

		removeIdentities(qc);
	}

	bool CircuitOptimizer::removeDiagonalGate(DAG& dag, DAGIterators& dagIterators, unsigned short idx, DAGIterator& it, qc::Operation* op) {
		// not a diagonal gate
		if (std::find(diagonalGates.begin(), diagonalGates.end(), op->getType()) == diagonalGates.end()) {
			it = dag.at(idx).end();
			return false;
		}

		if (op->getNcontrols()!=0) {
			// need to check all controls and targets
			bool onlyDiagonalGates = true;
			for (const auto& control: op->getControls()) {
				auto controlQubit = control.qubit;
				if (controlQubit == idx)
					continue;
				if (control.type == Control::neg) {
					dagIterators.at(controlQubit) = dag.at(controlQubit).end();
					onlyDiagonalGates = false;
					break;
				}
				if (dagIterators.at(controlQubit) == dag.at(controlQubit).end()) {
					onlyDiagonalGates = false;
					break;
				}
				// recursive call at control with this operation as goal
				removeDiagonalGatesBeforeMeasureRecursive(dag, dagIterators, controlQubit, it);
				// check if iteration of control qubit was successful
				if (*dagIterators.at(controlQubit) != *it) {
					onlyDiagonalGates = false;
					break;
				}
			}
			for (const auto& target: op->getTargets()) {
				if (target == idx)
					continue;
				if (dagIterators.at(target) == dag.at(target).end()) {
					onlyDiagonalGates = false;
					break;
				}
				// recursive call at target with this operation as goal
				removeDiagonalGatesBeforeMeasureRecursive(dag, dagIterators, target, it);
				// check if iteration of target qubit was successful
				if (*dagIterators.at(target) != *it) {
					onlyDiagonalGates = false;
					break;
				}
			}
			if (!onlyDiagonalGates) {
				// end qubit
				dagIterators.at(idx) = dag.at(idx).end();
			} else {
				// set operation to identity so that it can be collected by the removeIdentities pass
				op->setGate(qc::I);
			}
			return onlyDiagonalGates;
		} else {
			// set operation to identity so that it can be collected by the removeIdentities pass
			op->setGate(qc::I);
			return true;
		}
	}

	void CircuitOptimizer::removeDiagonalGatesBeforeMeasureRecursive(DAG& dag, DAGIterators& dagIterators, unsigned short idx, const DAGIterator& until) {
		// qubit is finished -> consider next qubit
		if (dagIterators.at(idx) == dag.at(idx).end()) {
			if(idx < dag.size()-1) {
				removeDiagonalGatesBeforeMeasureRecursive(dag, dagIterators, idx + 1, dag.at(idx + 1).end());
			}
			return;
		}
		// check if desired operation was reached
		if (until != dag.at(idx).end()) {
			if (*dagIterators.at(idx) == *until) {
				return;
			}
		}

		auto& it = dagIterators.at(idx);
		while (it != dag.at(idx).end()) {
			// check if desired operation was reached
			if (until != dag.at(idx).end()) {
				if (*dagIterators.at(idx) == *until) {
					break;
				}
			}
			auto op = (*it)->get();
			if (op->getType() == Barrier) {
				// either ignore barrier statement here or end for this qubit;
				++it;
			} else if (op->isStandardOperation()) {
				// try removing gate and upon success increase all corresponding iterators
				auto onlyDiagonalGates = removeDiagonalGate(dag, dagIterators, idx, it, op);
				if (onlyDiagonalGates) {
					for (const auto& control: op->getControls()){
						++(dagIterators.at(control.qubit));
					}
					for (const auto& target: op->getTargets()) {
						++(dagIterators.at(target));
					}
				}

			} else if (op->isCompoundOperation()) {
				// iterate over all gates of compound operation and upon success increase all corresponding iterators
				auto compOp = dynamic_cast<qc::CompoundOperation *>(op);
				bool onlyDiagonalGates = true;
				auto cit = compOp->rbegin();
				while (cit != compOp->rend()) {
					auto cop = (*cit).get();
					onlyDiagonalGates = removeDiagonalGate(dag, dagIterators, idx, it, cop);
					if (!onlyDiagonalGates)
						break;
					++cit;
				}
				if (onlyDiagonalGates) {
					for (size_t q=0; q<dag.size(); ++q) {
						if (compOp->actsOn(q))
							++(dagIterators.at(q));
					}
				}
			} else if (op->isClassicControlledOperation()) {
				// consider the operation that is classically controlled and proceed as above
				auto cop = dynamic_cast<ClassicControlledOperation *>(op)->getOperation();
				bool onlyDiagonalGates = removeDiagonalGate(dag, dagIterators, idx, it, cop);
				if (onlyDiagonalGates) {
					for (const auto& control: cop->getControls()){
						++(dagIterators.at(control.qubit));
					}
					for (const auto& target: cop->getTargets()) {
						++(dagIterators.at(target));
					}
				}
			} else if (op->isNonUnitaryOperation()) {
				// non-unitary operation is not diagonal
				it = dag.at(idx).end();
			} else {
				throw QFRException("Unexpected operation encountered");
			}
		}

		// qubit is finished -> consider next qubit
		if (dagIterators.at(idx) == dag.at(idx).end() && idx < dag.size()-1) {
			removeDiagonalGatesBeforeMeasureRecursive(dag, dagIterators, idx + 1, dag.at(idx + 1).end());
		}
	}

	void CircuitOptimizer::removeDiagonalGatesBeforeMeasure(QuantumComputation& qc) {

		auto dag = constructDAG(qc);

		// initialize iterators
		DAGIterators dagIterators{dag.size()};
		for (size_t q=0; q<dag.size(); ++q) {
			if (dag.at(q).empty() || dag.at(q).front()->get()->getType() != qc::Measure) {
				// qubit is not measured and thus does not have to be considered
				dagIterators.at(q) = dag.at(q).end();
			} else {
				// point to operation before measurement
				dagIterators.at(q) = ++(dag.at(q).begin());
			}
		}
		// iterate over DAG in depth-first fashion
		removeDiagonalGatesBeforeMeasureRecursive(dag, dagIterators, 0, dag.at(0).end());

		// remove resulting identities from circuit
		removeIdentities(qc);
	}
}
