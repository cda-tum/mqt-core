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
            if ((*it)->getType() == I || (*it)->getType() == Barrier) {
                it = qc.ops.erase(it);
            } else if ((*it)->isCompoundOperation()) {
                auto compOp = dynamic_cast<qc::CompoundOperation*>((*it).get());
                auto cit    = compOp->cbegin();
                while (cit != compOp->cend()) {
                    auto cop = cit->get();
                    if (cop->getType() == qc::I || cop->getType() == Barrier) {
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
                            dag.at(i).push_back(&it);
                        }
                    }
                    continue;
                } else if (it->isNonUnitaryOperation()) {
                    if (it->getType() == Barrier)
                        continue;

                    for (const auto& b: it->getTargets()) {
                        dag.at(b).push_back(&it);
                    }
                    continue;
                } else if (it->isClassicControlledOperation()) {
                    auto op = dynamic_cast<ClassicControlledOperation*>(it.get())->getOperation();
                    for (const auto& control: op->getControls()) {
                        dag.at(control.qubit).push_back(&it);
                    }
                    for (const auto& target: op->getTargets()) {
                        dag.at(target).push_back(&it);
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

            auto opC = dag.at(control).back();
            auto opT = dag.at(target).back();

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
                dag.at(control).pop_back();
                dag.at(target).pop_back();
                (*opC)->setGate(I);
                (*opC)->setControls({});
                it->setGate(I);
                it->setControls({});
            } else if (control == opCtarget && target == opCcontrol) {
                dag.at(control).pop_back();
                dag.at(target).pop_back();

                // replace with SWAP + CNOT
                (*opC)->setGate(SWAP);
                if (target > control) {
                    (*opC)->setTargets({control, target});
                } else {
                    (*opC)->setTargets({target, control});
                }
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
                            dag.at(i).push_back(&it);
                        }
                    }
                    continue;
                } else if (it->isNonUnitaryOperation()) {
                    for (const auto& b: it->getTargets()) {
                        dag.at(b).push_back(&it);
                    }
                    continue;
                } else if (it->isClassicControlledOperation()) {
                    auto op = dynamic_cast<ClassicControlledOperation*>(it.get())->getOperation();
                    for (const auto& control: op->getControls()) {
                        dag.at(control.qubit).push_back(&it);
                    }
                    for (const auto& target: op->getTargets()) {
                        dag.at(target).push_back(&it);
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
            dag.at(control.qubit).push_back(op);
        }
        for (const auto& target: (*op)->getTargets()) {
            dag.at(target).push_back(op);
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
                            dag.at(i).push_back(&it);
                        }
                    }
                    continue;
                } else if (it->isNonUnitaryOperation()) {
                    for (const auto& b: it->getTargets()) {
                        dag.at(b).push_back(&it);
                    }
                    continue;
                } else if (it->isClassicControlledOperation()) {
                    auto op = dynamic_cast<ClassicControlledOperation*>(it.get())->getOperation();
                    for (const auto& control: op->getControls()) {
                        dag.at(control.qubit).push_back(&it);
                    }
                    for (const auto& target: op->getTargets()) {
                        dag.at(target).push_back(&it);
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
            auto op       = dagQubit.back();

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
                dag.at(target).push_back(op);
            }
        }

        removeIdentities(qc);
    }

    bool CircuitOptimizer::removeDiagonalGate(DAG& dag, DAGReverseIterators& dagIterators, dd::Qubit idx, DAGReverseIterator& it, qc::Operation* op) {
        // not a diagonal gate
        if (std::find(diagonalGates.begin(), diagonalGates.end(), op->getType()) == diagonalGates.end()) {
            it = dag.at(idx).rend();
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
                    dagIterators.at(controlQubit) = dag.at(controlQubit).rend();
                    onlyDiagonalGates             = false;
                    break;
                }
                if (dagIterators.at(controlQubit) == dag.at(controlQubit).rend()) {
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
                if (dagIterators.at(target) == dag.at(target).rend()) {
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
                dagIterators.at(idx) = dag.at(idx).rend();
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

    void CircuitOptimizer::removeDiagonalGatesBeforeMeasureRecursive(DAG& dag, DAGReverseIterators& dagIterators, dd::Qubit idx, const DAGReverseIterator& until) {
        // qubit is finished -> consider next qubit
        if (dagIterators.at(idx) == dag.at(idx).rend()) {
            if (idx < static_cast<dd::Qubit>(dag.size() - 1)) {
                removeDiagonalGatesBeforeMeasureRecursive(dag, dagIterators, static_cast<dd::Qubit>(idx + 1), dag.at(idx + 1).rend());
            }
            return;
        }
        // check if desired operation was reached
        if (until != dag.at(idx).rend()) {
            if (*dagIterators.at(idx) == *until) {
                return;
            }
        }

        auto& it = dagIterators.at(idx);
        while (it != dag.at(idx).rend()) {
            // check if desired operation was reached
            if (until != dag.at(idx).rend()) {
                if (*dagIterators.at(idx) == *until) {
                    break;
                }
            }
            auto op = (*it)->get();
            if (op->getType() == Barrier || op->getType() == Snapshot || op->getType() == ShowProbabilities) {
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
                it = dag.at(idx).rend();
            } else {
                throw QFRException("Unexpected operation encountered");
            }
        }

        // qubit is finished -> consider next qubit
        if (dagIterators.at(idx) == dag.at(idx).rend() && idx < static_cast<dd::Qubit>(dag.size() - 1)) {
            removeDiagonalGatesBeforeMeasureRecursive(dag, dagIterators, static_cast<dd::Qubit>(idx + 1), dag.at(idx + 1).rend());
        }
    }

    void CircuitOptimizer::removeDiagonalGatesBeforeMeasure(QuantumComputation& qc) {
        auto dag = constructDAG(qc);

        // initialize iterators
        DAGReverseIterators dagIterators{dag.size()};
        for (size_t q = 0; q < dag.size(); ++q) {
            if (dag.at(q).empty() || dag.at(q).back()->get()->getType() != qc::Measure) {
                // qubit is not measured and thus does not have to be considered
                dagIterators.at(q) = dag.at(q).rend();
            } else {
                // point to operation before measurement
                dagIterators.at(q) = ++(dag.at(q).rbegin());
            }
        }
        // iterate over DAG in depth-first fashion
        removeDiagonalGatesBeforeMeasureRecursive(dag, dagIterators, 0, dag.at(0).rend());

        // remove resulting identities from circuit
        removeIdentities(qc);
    }

    bool CircuitOptimizer::removeFinalMeasurement(DAG& dag, DAGReverseIterators& dagIterators, dd::Qubit idx, DAGReverseIterator& it, qc::Operation* op) {
        if (op->getNtargets() != 0) {
            // need to check all targets
            bool onlyMeasurments = true;
            for (const auto& target: op->getTargets()) {
                if (target == idx)
                    continue;
                if (dagIterators.at(target) == dag.at(target).rend()) {
                    onlyMeasurments = false;
                    break;
                }
                // recursive call at target with this operation as goal
                removeFinalMeasurementsRecursive(dag, dagIterators, target, it);
                // check if iteration of target qubit was successful
                if (dagIterators.at(target) == dag.at(target).rend() || *dagIterators.at(target) != *it) {
                    onlyMeasurments = false;
                    break;
                }
            }
            if (!onlyMeasurments) {
                // end qubit
                dagIterators.at(idx) = dag.at(idx).rend();
            } else {
                // set operation to identity so that it can be collected by the removeIdentities pass
                op->setGate(qc::I);
            }
            return onlyMeasurments;
        } else {
            return false;
        }
    }

    void CircuitOptimizer::removeFinalMeasurementsRecursive(DAG& dag, DAGReverseIterators& dagIterators, dd::Qubit idx, const DAGReverseIterator& until) {
        if (dagIterators.at(idx) == dag.at(idx).rend()) { //we reached the end
            if (idx < static_cast<dd::Qubit>(dag.size() - 1)) {
                removeFinalMeasurementsRecursive(dag, dagIterators, static_cast<dd::Qubit>(idx + 1), dag.at(idx + 1).rend());
            }
            return;
        }
        // check if desired operation was reached
        if (until != dag.at(idx).rend()) {
            if (*dagIterators.at(idx) == *until) {
                return;
            }
        }
        auto& it = dagIterators.at(idx);
        while (it != dag.at(idx).rend()) {
            if (until != dag.at(idx).rend()) {
                if (*dagIterators.at(idx) == *until) {
                    break;
                }
            }
            auto op = (*it)->get();
            if (op->getType() == Measure) {
                bool onlyMeasurment = removeFinalMeasurement(dag, dagIterators, idx, it, op);
                if (onlyMeasurment) {
                    for (const auto& target: op->getTargets()) {
                        if (dagIterators.at(target) == dag.at(target).rend())
                            break;
                        ++(dagIterators.at(target));
                    }
                }
            } else if (op->getType() == Barrier || op->getType() == Snapshot || op->getType() == ShowProbabilities) {
                for (const auto& target: op->getTargets()) {
                    if (dagIterators.at(target) == dag.at(target).rend())
                        break;
                    ++(dagIterators.at(target));
                }
            } else if (op->isCompoundOperation() && op->isNonUnitaryOperation()) {
                // iterate over all gates of compound operation and upon success increase all corresponding iterators
                auto compOp          = dynamic_cast<qc::CompoundOperation*>(op);
                bool onlyMeasurement = true;
                auto cit             = compOp->rbegin();
                while (cit != compOp->rend()) {
                    auto cop = (*cit).get();
                    if (cop->getNtargets() > 0 && cop->getTargets()[0] != idx) {
                        ++cit;
                        continue;
                    }
                    onlyMeasurement = removeFinalMeasurement(dag, dagIterators, idx, it, cop);
                    if (!onlyMeasurement)
                        break;
                    ++cit;
                }
                if (onlyMeasurement) {
                    ++(dagIterators.at(idx));
                }
            } else {
                // not a measurement, we are done
                dagIterators.at(idx) = dag.at(idx).rend();
                break;
            }
        }
        if (dagIterators.at(idx) == dag.at(idx).rend() && idx < static_cast<dd::Qubit>(dag.size() - 1)) {
            removeFinalMeasurementsRecursive(dag, dagIterators, static_cast<dd::Qubit>(idx + 1), dag.at(idx + 1).rend());
        }
    }

    void CircuitOptimizer::removeFinalMeasurements(QuantumComputation& qc) {
        auto                dag = constructDAG(qc);
        DAGReverseIterators dagIterators{dag.size()};
        for (size_t q = 0; q < dag.size(); ++q) {
            dagIterators.at(q) = (dag.at(q).rbegin());
        }

        removeFinalMeasurementsRecursive(dag, dagIterators, 0, dag.at(0).rend());

        removeIdentities(qc);
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

    void CircuitOptimizer::eliminateResets(QuantumComputation& qc) {
        //      ┌───┐┌─┐     ┌───┐┌─┐            ┌───┐┌─┐ ░
        // q_0: ┤ H ├┤M├─|0>─┤ H ├┤M├       q_0: ┤ H ├┤M├─░─────────
        //      └───┘└╥┘     └───┘└╥┘   -->      └───┘└╥┘ ░ ┌───┐┌─┐
        // c: 2/══════╩════════════╩═       q_1: ──────╫──░─┤ H ├┤M├
        //            0            1                   ║  ░ └───┘└╥┘
        //                                  c: 2/══════╩══════════╩═
        //                                             0          1
        auto replacementMap = std::map<dd::Qubit, dd::Qubit>();
        auto it             = qc.ops.begin();
        while (it != qc.ops.end()) {
            if ((*it)->getType() == qc::Reset) {
                for (const auto& target: (*it)->getTargets()) {
                    auto indexAddQubit = static_cast<dd::Qubit>(qc.getNqubits());
                    qc.addQubit(indexAddQubit, indexAddQubit, indexAddQubit);
                    auto oldReset = replacementMap.find(target);
                    if (oldReset != replacementMap.end()) {
                        oldReset->second = indexAddQubit;
                    } else {
                        replacementMap.insert(std::pair(static_cast<dd::Qubit>(target), static_cast<dd::Qubit>(indexAddQubit)));
                    }
                }
                it = qc.erase(it);
            } else if (!replacementMap.empty()) {
                if ((*it)->isCompoundOperation()) {
                    auto* compOp   = dynamic_cast<qc::CompoundOperation*>((*it).get());
                    auto  compOpIt = compOp->begin();
                    while (compOpIt != compOp->end()) {
                        if ((*compOpIt)->getType() == qc::Reset) {
                            for (const auto& compTarget: (*compOpIt)->getTargets()) {
                                auto indexAddQubit = static_cast<dd::Qubit>(qc.getNqubits());
                                qc.addQubit(indexAddQubit, indexAddQubit, indexAddQubit);
                                auto oldReset = replacementMap.find(compTarget);
                                if (oldReset != replacementMap.end()) {
                                    oldReset->second = indexAddQubit;
                                } else {
                                    replacementMap.insert(std::pair(static_cast<dd::Qubit>(compTarget), static_cast<dd::Qubit>(indexAddQubit)));
                                }
                            }
                            compOpIt = compOp->erase(compOpIt);
                        } else {
                            if ((*compOpIt)->isStandardOperation() || (*compOpIt)->isClassicControlledOperation()) {
                                auto& targets  = (*compOpIt)->getTargets();
                                auto& controls = (*compOpIt)->getControls();
                                changeTargets(targets, replacementMap);
                                changeControls(controls, replacementMap);
                            } else if ((*compOpIt)->isNonUnitaryOperation()) {
                                auto& targets = (*compOpIt)->getTargets();
                                changeTargets(targets, replacementMap);
                            }
                            compOpIt++;
                        }
                    }
                }
                if ((*it)->isStandardOperation() || (*it)->isClassicControlledOperation()) {
                    auto& targets  = (*it)->getTargets();
                    auto& controls = (*it)->getControls();
                    changeTargets(targets, replacementMap);
                    changeControls(controls, replacementMap);
                } else if ((*it)->isNonUnitaryOperation()) {
                    auto& targets = (*it)->getTargets();
                    changeTargets(targets, replacementMap);
                }
                it++;
            } else {
                it++;
            }
        }
        // if anything has been modified the number of qubits of each gate has to be adjusted
        if (!replacementMap.empty()) {
            for (auto& op: qc.ops) {
                op->setNqubits(qc.getNqubits());
            }
        }
    }

    void CircuitOptimizer::changeTargets(Targets& targets, const std::map<dd::Qubit, dd::Qubit>& replacementMap) {
        for (auto& target: targets) {
            auto newTargetIt = replacementMap.find(target);
            if (newTargetIt != replacementMap.end()) {
                target = newTargetIt->second;
            }
        }
    }

    void CircuitOptimizer::changeControls(dd::Controls& controls, const std::map<dd::Qubit, dd::Qubit>& replacementMap) {
        if (controls.empty() || replacementMap.empty())
            return;

        // iterate over the replacement map and see if any control matches
        for (const auto& [from, to]: replacementMap) {
            auto controlIt = controls.find(from);
            if (controlIt != controls.end()) {
                const auto controlType = controlIt->type;
                controls.erase(controlIt);
                controls.insert(dd::Control{to, controlType});
            }
        }
    }

    void CircuitOptimizer::deferMeasurements(QuantumComputation& qc) {
        //      ┌───┐┌─┐                         ┌───┐     ┌─┐
        // q_0: ┤ H ├┤M├───────             q_0: ┤ H ├──■──┤M├
        //      └───┘└╥┘ ┌───┐                   └───┘┌─┴─┐└╥┘
        // q_1: ──────╫──┤ X ├─     -->     q_1: ─────┤ X ├─╫─
        //            ║  └─╥─┘                        └───┘ ║
        //            ║ ┌──╨──┐             c: 2/═══════════╩═
        // c: 2/══════╩═╡ = 1 ╞                             0
        //            0 └─────┘
        std::unordered_map<dd::Qubit, std::size_t> qubitsToAddMeasurements{};
        auto                                       it = qc.begin();
        while (it != qc.end()) {
            if ((*it)->getType() == qc::Measure) {
                auto*      measurement = dynamic_cast<qc::NonUnitaryOperation*>(it->get());
                const auto targets     = measurement->getTargets();
                const auto classics    = measurement->getClassics();

                if (targets.size() != 1 && classics.size() != 1) {
                    throw QFRException("Deferring measurements with more than 1 target is not yet supported. Try decomposing your measurements.");
                }

                // if this is the last operation, nothing has to be done
                if (*it == qc.ops.back())
                    break;

                // remember q-> c for adding measurements later
                qubitsToAddMeasurements[targets[0]] = classics[0];

                // remove the measurement from the vector of operations
                it = qc.erase(it);

                // starting from the next operation after the measurement (if there is any)
                auto opIt                  = it;
                auto currentInsertionPoint = it;

                // iterate over all subsequent operations
                while (opIt != qc.end()) {
                    const auto operation = opIt->get();
                    if (operation->isUnitary()) {
                        // if an operation does not act on the measured qubit, the insert location for potential operations has to be updated
                        if (!operation->actsOn(targets.at(0))) {
                            ++currentInsertionPoint;
                        }
                        ++opIt;
                        continue;
                    }

                    if (operation->getType() == qc::Reset) {
                        throw QFRException("Reset encountered in deferMeasurements routine. Please use the eliminateResets method before deferring measurements.");
                    }

                    if (operation->getType() == qc::Measure) {
                        const auto* measurement2 = dynamic_cast<qc::NonUnitaryOperation*>((*opIt).get());
                        const auto& targets2     = measurement2->getTargets();
                        const auto& classics2    = measurement2->getClassics();

                        // if this is the same measurement a breakpoint has been reached
                        if (targets == targets2 && classics == classics2) {
                            break;
                        } else {
                            ++currentInsertionPoint;
                            ++opIt;
                            continue;
                        }
                    }

                    if (operation->isClassicControlledOperation()) {
                        auto        classicOp       = dynamic_cast<qc::ClassicControlledOperation*>((*opIt).get());
                        const auto& controlRegister = classicOp->getControlRegister();
                        const auto& expectedValue   = classicOp->getExpectedValue();

                        if (controlRegister.second != 1 && expectedValue <= 1) {
                            throw QFRException("Classic-controlled operations targetted at more than one bit are currently not supported. Try decomposing the operation into individual contributions.");
                        }

                        // if this is not the classical bit that is measured, continue
                        if (controlRegister.first == static_cast<dd::Qubit>(classics.at(0))) {
                            // get the underlying operation
                            const auto* standardOp = dynamic_cast<qc::StandardOperation*>(classicOp->getOperation());

                            // get all the necessary information for reconstructing the operation
                            const auto nqubits = standardOp->getNqubits();
                            const auto type    = standardOp->getType();

                            const auto targs = standardOp->getTargets();
                            for (const auto& target: targs) {
                                if (target == targets[0]) {
                                    throw qc::QFRException("Implicit reset operation in circuit detected. Measuring a qubit and then targeting the same qubit with a classic-controlled operation is not allowed at the moment.");
                                }
                            }

                            // determine the appropriate control to add
                            auto controls     = standardOp->getControls();
                            auto controlQubit = targets.at(0);
                            auto controlType  = (expectedValue == 1) ? dd::Control::Type::pos : dd::Control::Type::neg;
                            controls.emplace(dd::Control{controlQubit, controlType});

                            const auto parameters = standardOp->getParameter();

                            // remove the classic-controlled operation
                            opIt = qc.erase(opIt);

                            // insert the new operation (invalidated all pointer onwards)
                            currentInsertionPoint = qc.insert(currentInsertionPoint,
                                                              std::make_unique<qc::StandardOperation>(nqubits,
                                                                                                      controls,
                                                                                                      targs,
                                                                                                      type,
                                                                                                      parameters[0],
                                                                                                      parameters[1],
                                                                                                      parameters[2]));

                            // advance just after the currently inserted operation
                            ++currentInsertionPoint;
                            // the inner loop also has to restart from here due to the invalidation of the iterators
                            opIt = currentInsertionPoint;
                        } else {
                            if (!operation->actsOn(targets[0])) {
                                ++currentInsertionPoint;
                            }
                            ++opIt;
                            continue;
                        }
                    }
                }
            }
            ++it;
        }
        for (const auto& [qubit, clbit]: qubitsToAddMeasurements) {
            qc.measure(qubit, clbit);
        }
    }

    bool CircuitOptimizer::isDynamicCircuit(QuantumComputation& qc) {
        dd::Qubit highest_physical_qubit = 0;
        for (const auto& q: qc.initialLayout) {
            if (q.first > highest_physical_qubit)
                highest_physical_qubit = q.first;
        }

        auto dag = DAG(highest_physical_qubit + 1);

        bool hasMeasurements = false;

        for (auto& it: qc.ops) {
            if (!it->isStandardOperation()) {
                if (it->isNonUnitaryOperation()) {
                    // whenever a reset operation is encountered the circuit has to be dynamic
                    if (it->getType() == Reset)
                        return true;

                    // record whether the circuit contains measurements
                    if (it->getType() == Measure)
                        hasMeasurements = true;

                    for (const auto& b: it->getTargets()) {
                        dag.at(b).push_back(&it);
                    }
                } else if (it->isClassicControlledOperation()) {
                    // whenever a classic-controlled operation is encountered the circuit has to be dynamic
                    return true;
                } else if (it->isCompoundOperation()) {
                    auto* comp_op = dynamic_cast<CompoundOperation*>(it.get());
                    for (auto& op: *comp_op) {
                        if (op->getType() == Reset || op->isClassicControlledOperation())
                            return true;

                        if (op->getType() == Measure)
                            hasMeasurements = true;

                        if (op->isNonUnitaryOperation()) {
                            for (const auto& b: op->getTargets()) {
                                dag.at(b).push_back(&op);
                            }
                        } else {
                            addToDag(dag, &op);
                        }
                    }
                }
            } else {
                addToDag(dag, &it);
            }
        }

        if (!hasMeasurements)
            return false;

        for (const auto& qubitDAG: dag) {
            bool operation   = false;
            bool measurement = false;
            for (auto it = qubitDAG.rbegin(); it != qubitDAG.rend(); ++it) {
                auto op = *it;
                // once a measurement is encountered the iteration for this qubit can stop
                if (op->get()->getType() == qc::Measure) {
                    measurement = true;
                    break;
                }

                if (op->get()->isStandardOperation() || op->get()->isClassicControlledOperation() || op->get()->isCompoundOperation() || op->get()->getType() == Reset) {
                    operation = true;
                }
            }
            // there was a measurement and then a non-trivial operation, so the circuit is dynamic
            if (measurement && operation)
                return true;
        }

        return false;
    }

    /// this method can be used to reorder the operations of a given quantum computation in order to get a canonical ordering
    /// it uses iterative breadth-first search starting from the topmost qubit
    void CircuitOptimizer::reorderOperations(QuantumComputation& qc) {
        //        std::cout << qc << std::endl;
        auto dag = constructDAG(qc);
        //        printDAG(dag);

        // initialize iterators
        DAGIterators dagIterators{dag.size()};
        for (size_t q = 0; q < dag.size(); ++q) {
            if (dag.at(q).empty()) {
                // qubit is isdle
                dagIterators.at(q) = dag.at(q).end();
            } else {
                // point to first operation
                dagIterators.at(q) = dag.at(q).begin();
            }
        }

        std::vector<std::unique_ptr<qc::Operation>> ops{};

        // iterate over DAG in depth-first fashion starting from the top-most qubit
        const auto msq  = static_cast<dd::Qubit>(dag.size() - 1);
        bool       done = false;
        while (!done) {
            // assume that everything is done
            done = true;

            // iterate over qubits in reverse order
            for (dd::Qubit q = msq; q >= 0; --q) {
                // nothing to be done for this qubit
                if (dagIterators.at(q) == dag.at(q).end()) {
                    continue;
                }
                done = false;

                // get the current operation on the qubit
                auto& it = dagIterators.at(q);
                auto& op = **it;

                // warning for classically controlled operations
                if (op->getType() == ClassicControlled) {
                    std::cerr << "Caution! Reordering operations might not work if the circuit contains classically controlled operations" << std::endl;
                }

                // ignore barrier, snapshot and probabilities statements;
                if (op->getType() == Barrier || op->getType() == Snapshot || op->getType() == ShowProbabilities) {
                    ++it;
                    continue;
                }

                // check whether the gate can be scheduled, i.e. whether all qubits it acts on are at this operation
                bool                                                    executable = true;
                std::bitset<std::numeric_limits<dd::QubitCount>::max()> actsOn{};
                actsOn.set(q);
                for (std::size_t i = 0; i < dag.size(); ++i) {
                    // actually check in reverse order
                    auto qb = static_cast<dd::Qubit>(dag.size() - 1 - i);
                    if (qb != q && op->actsOn(static_cast<dd::Qubit>(qb))) {
                        actsOn.set(qb);

                        assert(dagIterators.at(qb) != dag.at(qb).end());
                        // check whether operation is executable for the currently considered qubit
                        if (*dagIterators.at(qb) != *it) {
                            executable = false;
                            break;
                        }
                    }
                }

                // continue, if this gate is not yet executable
                if (!executable)
                    continue;

                // gate is executable, move it to the new vector
                ops.emplace_back(std::move(op));

                // now increase all corresponding iterators
                for (std::size_t i = 0; i < dag.size(); ++i) {
                    if (actsOn.test(i)) {
                        ++(dagIterators.at(i));
                    }
                }
            }
        }

        // clear all the operations from the quantum circuit
        qc.ops.clear();
        // move all operations from the newly created vector to the original one
        std::move(ops.begin(), ops.end(), std::back_inserter(qc.ops));
    }

    void CircuitOptimizer::printDAG(const DAG& dag) {
        for (const auto& qubitDag: dag) {
            std::cout << " - ";
            for (const auto& op: qubitDag) {
                std::cout << std::hex << (*op).get() << std::dec << "(" << toString((*op)->getType()) << ") - ";
            }
            std::cout << std::endl;
        }
    }
    void CircuitOptimizer::printDAG(const DAG& dag, const DAGIterators& iterators) {
        for (std::size_t i = 0; i < dag.size(); ++i) {
            std::cout << " - ";
            for (auto it = iterators.at(i); it != dag.at(i).end(); ++it) {
                std::cout << std::hex << (**it).get() << std::dec << "(" << toString((**it)->getType()) << ") - ";
            }
            std::cout << std::endl;
        }
    }
} // namespace qc
