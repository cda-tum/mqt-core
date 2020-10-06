/*
 * This file is part of IIC-JKU QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#include "CircuitOptimizer.hpp"

namespace qc {

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

		bool allEmpty = false;
		while (!allEmpty) {
			allEmpty = true;
			for (int i = 0; i < qc.nqubits + qc.nancillae; ++i) {
				if(!dag.at(i).empty()) {
					allEmpty = false;
					for (const auto& c: dag.at(i).front()->get()->getControls()){
						std::cout << c.qubit;
					}
					for (const auto& t: dag.at(i).front()->get()->getTargets()){
						std::cout << t;
					}
					std::cout << dag.at(i).front()->get()->getName();
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

	void CircuitOptimizer::addToDag(DAG& dag, std::unique_ptr<Operation> *op) {
		for (const auto& control: (*op)->getControls()) {
			dag.at(control.qubit).push_front(op);
		}
		for (const auto& target: (*op)->getTargets()) {
			dag.at(target).push_front(op);
		}
	}

	void CircuitOptimizer::singleGateFusion(QuantumComputation& qc) {
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

			// not a single qubit operation
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

			// no single qubit op to fuse with operation to fuse with
			if (!(*op)->isCompoundOperation() && (!(*op)->getControls().empty() || (*op)->getTargets().size() > 1)) {
				addToDag(dag, &it);
				continue;
			}

			// compound operation
			if ((*op)->isCompoundOperation()) {
				auto compop = dynamic_cast<CompoundOperation*>(op->get());
				auto lastop = (--(compop->end()));

				// compound operation does contain non single-qubit gates
				if (!(*lastop)->getControls().empty() || (*lastop)->getTargets().size() > 1) {
					addToDag(dag, &it);
					continue;
				}

				compop->emplace_back<StandardOperation>(
						it->getNqubits(),
						it->getTargets().at(0),
						it->getType(),
						it->getParameter().at(0),
						it->getParameter().at(1),
						it->getParameter().at(2));
				it->setGate(I);
				continue;
			}

			// single qubit op
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

		removeIdentities(qc);
	}

}
