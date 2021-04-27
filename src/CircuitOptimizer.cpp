/*
 * This file is part of JKQ QFR library which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum/ for more information.
 */

#include "CircuitOptimizer.hpp"

namespace qc {
    void CircuitOptimizer::removeIdentities(QuantumComputation& qc) {
        // delete the identities from circuit
        auto it = qc.ops.begin();
        while (it != qc.ops.end()) {
            if ((*it)->isStandardOperation()) {
                if ((*it)->getType() == I) {
                    it = qc.ops.erase(it);
                } else {
                    ++it;
                }
            } else if ((*it)->isCompoundOperation()) {
                auto compOp = dynamic_cast<qc::CompoundOperation*>((*it).get());
                auto cit    = compOp->cbegin();
                while (cit != compOp->cend()) {
                    auto cop = cit->get();
                    if (cop->getType() == qc::I) {
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

    void CircuitOptimizer::swapReconstruction(QuantumComputation& qc) {
        // print incoming circuit
        //qc.print(std::cout );
        //std::cout << std::endl;
        dd::Qubit highest_physical_qubit = 0;
        for (const auto& q: qc.initialLayout) {
            if (q.first > highest_physical_qubit)
                highest_physical_qubit = q.first;
        }

        auto dag = DAG(highest_physical_qubit + 1);

        for (auto& it: qc.ops) {
            if (!it->isStandardOperation()) {
                // compound operations are added "as-is"
                if (it->isCompoundOperation()) {
                    std::clog << "Skipping compound operation during SWAP reconstruction!" << std::endl;
                    for (dd::QubitCount i = 0; i < it->getNqubits(); ++i) {
                        if (it->actsOn(static_cast<dd::Qubit>(i))) {
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
                    auto op = dynamic_cast<ClassicControlledOperation*>(it.get())->getOperation();
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
            if (it->getType() != X || it->getNcontrols() != 1 || it->getControls().begin()->type != dd::Control::Type::pos) {
                addToDag(dag, &it);
                continue;
            }

            dd::Qubit control = it->getControls().begin()->qubit;
            dd::Qubit target  = it->getTargets().at(0);

            // first operation
            if (dag.at(control).empty() || dag.at(target).empty()) {
                addToDag(dag, &it);
                continue;
            }

            auto opC = dag.at(control).front();
            auto opT = dag.at(target).front();

            // previous operation is not a CNOT
            if ((*opC)->getType() != qc::X || (*opC)->getNcontrols() != 1 || (*opC)->getControls().begin()->type != dd::Control::Type::pos ||
                (*opT)->getType() != qc::X || (*opT)->getNcontrols() != 1 || (*opT)->getControls().begin()->type != dd::Control::Type::pos) {
                addToDag(dag, &it);
                continue;
            }

            auto opCcontrol = (*opC)->getControls().begin()->qubit;
            auto opCtarget  = (*opC)->getTargets().at(0);
            auto opTcontrol = (*opT)->getControls().begin()->qubit;
            auto opTtarget  = (*opT)->getTargets().at(0);

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
                it->setControls({dd::Control{target}});
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
        dd::Qubit highest_physical_qubit = 0;
        for (const auto& q: qc.initialLayout) {
            if (q.first > highest_physical_qubit)
                highest_physical_qubit = q.first;
        }

        auto dag = DAG(highest_physical_qubit + 1);

        for (auto& it: qc.ops) {
            if (!it->isStandardOperation()) {
                // compound operations are added "as-is"
                if (it->isCompoundOperation()) {
                    for (dd::QubitCount i = 0; i < it->getNqubits(); ++i) {
                        if (it->actsOn(static_cast<dd::Qubit>(i))) {
                            dag.at(i).push_front(&it);
                        }
                    }
                    continue;
                } else if (it->getType() == qc::Measure) {
                    for (const auto& t: it->getTargets()) {
                        dag.at(t).push_front(&it);
                    }
                    continue;
                } else if (it->getType() == qc::Barrier || it->getType() == qc::Reset) {
                    for (const auto& b: it->getTargets()) {
                        dag.at(b).push_front(&it);
                    }
                    continue;
                } else if (it->isClassicControlledOperation()) {
                    auto op = dynamic_cast<ClassicControlledOperation*>(it.get())->getOperation();
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

    void CircuitOptimizer::addToDag(DAG& dag, std::unique_ptr<Operation>* op) {
        for (const auto& control: (*op)->getControls()) {
            dag.at(control.qubit).push_front(op);
        }
        for (const auto& target: (*op)->getTargets()) {
            dag.at(target).push_front(op);
        }
    }

    void CircuitOptimizer::singleQubitGateFusion(QuantumComputation& qc) {
        static const std::map<qc::OpType, qc::OpType> inverseMap = {
                {qc::I, qc::I},
                {qc::X, qc::X},
                {qc::Y, qc::Y},
                {qc::Z, qc::Z},
                {qc::H, qc::H},
                {qc::S, qc::Sdag},
                {qc::Sdag, qc::S},
                {qc::T, qc::Tdag},
                {qc::Tdag, qc::T},
                {qc::SX, qc::SXdag},
                {qc::SXdag, qc::SX}};

        dd::Qubit highest_physical_qubit = 0;
        for (const auto& q: qc.initialLayout) {
            if (q.first > highest_physical_qubit)
                highest_physical_qubit = q.first;
        }

        auto dag = DAG(highest_physical_qubit + 1);

        for (auto& it: qc.ops) {
            if (!it->isStandardOperation()) {
                // compound operations are added "as-is"
                if (it->isCompoundOperation()) {
                    for (dd::QubitCount i = 0; i < it->getNqubits(); ++i) {
                        if (it->actsOn(static_cast<dd::Qubit>(i))) {
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
                    auto op = dynamic_cast<ClassicControlledOperation*>(it.get())->getOperation();
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
            auto op       = dagQubit.front();

            // no single qubit op to fuse with operation
            if (!(*op)->isCompoundOperation() && (!(*op)->getControls().empty() || (*op)->getTargets().size() > 1)) {
                addToDag(dag, &it);
                continue;
            }

            // compound operation
            if ((*op)->isCompoundOperation()) {
                auto compop = dynamic_cast<CompoundOperation*>(op->get());

                // check if compound operation contains non-single-qubit gates
                dd::QubitCount involvedQubits = 0;
                for (std::size_t q = 0; q < dag.size(); ++q) {
                    if (compop->actsOn(static_cast<dd::Qubit>(q)))
                        ++involvedQubits;
                }
                if (involvedQubits > 1) {
                    addToDag(dag, &it);
                    continue;
                }

                // check if inverse
                auto lastop    = (--(compop->end()));
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

    bool CircuitOptimizer::removeDiagonalGate(DAG& dag, DAGIterators& dagIterators, dd::Qubit idx, DAGIterator& it, qc::Operation* op) {
        // not a diagonal gate
        if (std::find(diagonalGates.begin(), diagonalGates.end(), op->getType()) == diagonalGates.end()) {
            it = dag.at(idx).end();
            return false;
        }

        if (op->getNcontrols() != 0) {
            // need to check all controls and targets
            bool onlyDiagonalGates = true;
            for (const auto& control: op->getControls()) {
                auto controlQubit = control.qubit;
                if (controlQubit == idx)
                    continue;
                if (control.type == dd::Control::Type::neg) {
                    dagIterators.at(controlQubit) = dag.at(controlQubit).end();
                    onlyDiagonalGates             = false;
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

    void CircuitOptimizer::removeDiagonalGatesBeforeMeasureRecursive(DAG& dag, DAGIterators& dagIterators, dd::Qubit idx, const DAGIterator& until) {
        // qubit is finished -> consider next qubit
        if (dagIterators.at(idx) == dag.at(idx).end()) {
            if (idx < static_cast<dd::Qubit>(dag.size() - 1)) {
                removeDiagonalGatesBeforeMeasureRecursive(dag, dagIterators, static_cast<dd::Qubit>(idx + 1), dag.at(idx + 1).end());
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
                    for (const auto& control: op->getControls()) {
                        ++(dagIterators.at(control.qubit));
                    }
                    for (const auto& target: op->getTargets()) {
                        ++(dagIterators.at(target));
                    }
                }

            } else if (op->isCompoundOperation()) {
                // iterate over all gates of compound operation and upon success increase all corresponding iterators
                auto compOp            = dynamic_cast<qc::CompoundOperation*>(op);
                bool onlyDiagonalGates = true;
                auto cit               = compOp->rbegin();
                while (cit != compOp->rend()) {
                    auto cop          = (*cit).get();
                    onlyDiagonalGates = removeDiagonalGate(dag, dagIterators, idx, it, cop);
                    if (!onlyDiagonalGates)
                        break;
                    ++cit;
                }
                if (onlyDiagonalGates) {
                    for (size_t q = 0; q < dag.size(); ++q) {
                        if (compOp->actsOn(static_cast<dd::Qubit>(q)))
                            ++(dagIterators.at(q));
                    }
                }
            } else if (op->isClassicControlledOperation()) {
                // consider the operation that is classically controlled and proceed as above
                auto cop               = dynamic_cast<ClassicControlledOperation*>(op)->getOperation();
                bool onlyDiagonalGates = removeDiagonalGate(dag, dagIterators, idx, it, cop);
                if (onlyDiagonalGates) {
                    for (const auto& control: cop->getControls()) {
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
        if (dagIterators.at(idx) == dag.at(idx).end() && idx < static_cast<dd::Qubit>(dag.size() - 1)) {
            removeDiagonalGatesBeforeMeasureRecursive(dag, dagIterators, static_cast<dd::Qubit>(idx + 1), dag.at(idx + 1).end());
        }
    }

    void CircuitOptimizer::removeDiagonalGatesBeforeMeasure(QuantumComputation& qc) {
        auto dag = constructDAG(qc);

        // initialize iterators
        DAGIterators dagIterators{dag.size()};
        for (size_t q = 0; q < dag.size(); ++q) {
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

    void CircuitOptimizer::removeMarkedMeasurements(QuantumComputation& qc) {
        // delete the identities from circuit
        auto it = qc.ops.begin();
        while (it != qc.ops.end()) {
            if (!(*it)->isCompoundOperation()) {
                if ((*it)->getType() == I) {
                    it = qc.ops.erase(it);
                } else {
                    ++it;
                }
            } else if ((*it)->isCompoundOperation()) {
                auto compOp = dynamic_cast<qc::CompoundOperation*>((*it).get());
                auto cit    = compOp->cbegin();
                while (cit != compOp->cend()) {
                    auto cop = cit->get();
                    if (cop->getType() == qc::I) {
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

    bool CircuitOptimizer::removeFinalMeasurement(DAG& dag, DAGIterators& dagIterators, dd::Qubit idx, DAGIterator& it, qc::Operation* op) {
        if (op->getNtargets() != 0) {
            // need to check all targets
            bool onlyMeasurments = true;
            for (const auto& target: op->getTargets()) {
                if (target == idx)
                    continue;
                if (dagIterators.at(target) == dag.at(target).end()) {
                    onlyMeasurments = false;
                    break;
                }
                // recursive call at target with this operation as goal
                removeFinalMeasurementsRecursive(dag, dagIterators, target, it);
                // check if iteration of target qubit was successful
                if (*dagIterators.at(target) != *it) {
                    onlyMeasurments = false;
                    break;
                }
            }
            if (!onlyMeasurments) {
                // end qubit
                dagIterators.at(idx) = dag.at(idx).end();
            } else {
                // set operation to identity so that it can be collected by the removeIdentities pass
                op->setGate(qc::I);
            }
            return onlyMeasurments;
        } else {
            return false;
        }
    }

    void CircuitOptimizer::removeFinalMeasurementsRecursive(DAG& dag, DAGIterators& dagIterators, dd::Qubit idx, const DAGIterator& until) {
        if (dagIterators.at(idx) == dag.at(idx).end()) { //we reached the end
            if (idx < static_cast<dd::Qubit>(dag.size() - 1)) {
                removeFinalMeasurementsRecursive(dag, dagIterators, static_cast<dd::Qubit>(idx + 1), dag.at(idx + 1).end());
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
            if (until != dag.at(idx).end()) {
                if (*dagIterators.at(idx) == *until) {
                    break;
                }
            }
            auto op = (*it)->get();
            if (op->isNonUnitaryOperation() && op->getType() == Measure) {
                bool onlyMeasurment = removeFinalMeasurement(dag, dagIterators, idx, it, op);
                if (onlyMeasurment) {
                    for (const auto& target: op->getTargets()) {
                        if (dagIterators.at(target) == dag.at(target).end())
                            break;
                        ++(dagIterators.at(target));
                    }
                }

            } else if (op->isCompoundOperation()) {
                // iterate over all gates of compound operation and upon success increase all corresponding iterators
                auto compOp         = dynamic_cast<qc::CompoundOperation*>(op);
                bool onlyMeasurment = true;
                auto cit            = compOp->rbegin();
                while (cit != compOp->rend()) {
                    auto cop = (*cit).get();
                    if (cop->getNtargets() > 0 && cop->getTargets()[0] != idx) {
                        ++cit;
                        continue;
                    }
                    onlyMeasurment = removeFinalMeasurement(dag, dagIterators, idx, it, cop);
                    if (!onlyMeasurment)
                        break;
                    ++cit;
                }
                if (onlyMeasurment) {
                    ++(dagIterators.at(idx));
                }
            } else {
                //Not a Measurment, we are done
                break;
            }
        }
        if (dagIterators.at(idx) == dag.at(idx).end() && idx < static_cast<dd::Qubit>(dag.size() - 1)) {
            removeFinalMeasurementsRecursive(dag, dagIterators, static_cast<dd::Qubit>(idx + 1), dag.at(idx + 1).end());
        }
    }

    void CircuitOptimizer::removeFinalMeasurements(QuantumComputation& qc) {
        //remove final measurements in the circuit (for use in mapping tool)

        auto         dag = constructDAG(qc);
        DAGIterators dagIterators{dag.size()};
        for (size_t q = 0; q < dag.size(); ++q) {
            //qubit is measured, remove measurements
            dagIterators.at(q) = (dag.at(q).begin());
        }

        removeFinalMeasurementsRecursive(dag, dagIterators, 0, dag.at(0).end());

        removeMarkedMeasurements(qc);
    }

    void CircuitOptimizer::decomposeSWAP(QuantumComputation& qc, bool isDirectedArchitecture) {
        //decompose SWAPS in three cnot and optionally in four H
        auto it = qc.ops.begin();
        while (it != qc.ops.end()) {
            if ((*it)->isStandardOperation()) {
                if ((*it)->getType() == qc::SWAP) {
                    const auto     targets = (*it)->getTargets();
                    dd::QubitCount nqubits = (*it)->getNqubits();
                    it                     = qc.ops.erase(it);
                    it                     = qc.ops.insert(it, std::make_unique<StandardOperation>(nqubits, dd::Control{targets[0]}, targets[1], qc::X));
                    if (isDirectedArchitecture) {
                        it = qc.ops.insert(it, std::make_unique<StandardOperation>(nqubits, targets[0], qc::H));
                        it = qc.ops.insert(it, std::make_unique<StandardOperation>(nqubits, targets[1], qc::H));
                        it = qc.ops.insert(it, std::make_unique<StandardOperation>(nqubits, dd::Control{targets[0]}, targets[1], qc::X));
                        it = qc.ops.insert(it, std::make_unique<StandardOperation>(nqubits, targets[0], qc::H));
                        it = qc.ops.insert(it, std::make_unique<StandardOperation>(nqubits, targets[1], qc::H));
                    } else {
                        it = qc.ops.insert(it, std::make_unique<StandardOperation>(nqubits, dd::Control{targets[1]}, targets[0], qc::X));
                    }
                    it = qc.ops.insert(it, std::make_unique<StandardOperation>(nqubits, dd::Control{targets[0]}, targets[1], qc::X));
                } else {
                    ++it;
                }
            } else if ((*it)->isCompoundOperation()) {
                auto compOp = dynamic_cast<qc::CompoundOperation*>((*it).get());
                auto cit    = compOp->begin();
                while (cit != compOp->end()) {
                    if ((*cit)->isStandardOperation() && (*cit)->getType() == qc::SWAP) {
                        const auto     targets = (*cit)->getTargets();
                        dd::QubitCount nqubits = compOp->getNqubits();
                        cit                    = compOp->erase(cit);
                        cit                    = compOp->insert<StandardOperation>(cit, nqubits, dd::Control{targets[0]}, targets[1], qc::X);
                        if (isDirectedArchitecture) {
                            cit = compOp->insert<StandardOperation>(cit, nqubits, targets[0], qc::H);
                            cit = compOp->insert<StandardOperation>(cit, nqubits, targets[1], qc::H);
                            cit = compOp->insert<StandardOperation>(cit, nqubits, dd::Control{targets[0]}, targets[1], qc::X);
                            cit = compOp->insert<StandardOperation>(cit, nqubits, targets[0], qc::H);
                            cit = compOp->insert<StandardOperation>(cit, nqubits, targets[1], qc::H);
                        } else {
                            cit = compOp->insert<StandardOperation>(cit, nqubits, dd::Control{targets[1]}, targets[0], qc::X);
                        }
                        cit = compOp->insert<StandardOperation>(cit, nqubits, dd::Control{targets[0]}, targets[1], qc::X);
                    } else {
                        ++cit;
                    }
                }
                ++it;
            } else {
                ++it;
            }
        }
    }

    void CircuitOptimizer::decomposeTeleport([[maybe_unused]] QuantumComputation& qc) {
    }

} // namespace qc
