/*
 * This file is part of MQT QFR library which is released under the MIT license.
 * See file README.md or go to https://www.cda.cit.tum.de/research/quantum/ for more information.
 */

#include "CircuitOptimizer.hpp"

#include <cassert>

namespace qc {
    void CircuitOptimizer::removeIdentities(QuantumComputation& qc) {
        // delete the identities from circuit
        auto it = qc.ops.begin();
        while (it != qc.ops.end()) {
            if ((*it)->getType() == I || (*it)->getType() == Barrier) {
                it = qc.ops.erase(it);
            } else if ((*it)->isCompoundOperation()) {
                auto* compOp = dynamic_cast<qc::CompoundOperation*>((*it).get());
                auto  cit    = compOp->cbegin();
                while (cit != compOp->cend()) {
                    const auto* cop = cit->get();
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
        Qubit highestPhysicalQubit = 0;
        for (const auto& q: qc.initialLayout) {
            if (q.first > highestPhysicalQubit) {
                highestPhysicalQubit = q.first;
            }
        }

        auto dag = DAG(highestPhysicalQubit + 1);

        for (auto& it: qc.ops) {
            if (!it->isStandardOperation()) {
                addNonStandardOperationToDag(dag, &it);
                continue;
            }

            // Operation is not a CNOT
            if (it->getType() != X || it->getNcontrols() != 1 || it->getControls().begin()->type != Control::Type::Pos) {
                addToDag(dag, &it);
                continue;
            }

            const Qubit control = it->getControls().begin()->qubit;
            const Qubit target  = it->getTargets().at(0);

            // first operation
            if (dag.at(control).empty() || dag.at(target).empty()) {
                addToDag(dag, &it);
                continue;
            }

            auto* opC = dag.at(control).back();
            auto* opT = dag.at(target).back();

            // previous operation is not a CNOT
            if ((*opC)->getType() != qc::X || (*opC)->getNcontrols() != 1 || (*opC)->getControls().begin()->type != Control::Type::Pos ||
                (*opT)->getType() != qc::X || (*opT)->getNcontrols() != 1 || (*opT)->getControls().begin()->type != Control::Type::Pos) {
                addToDag(dag, &it);
                continue;
            }

            const auto opCcontrol = (*opC)->getControls().begin()->qubit;
            const auto opCtarget  = (*opC)->getTargets().at(0);
            const auto opTcontrol = (*opT)->getControls().begin()->qubit;
            const auto opTtarget  = (*opT)->getTargets().at(0);

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
                it->setControls({Control{target}});
                addToDag(dag, &it);
            } else {
                addToDag(dag, &it);
            }
        }

        removeIdentities(qc);
    }

    DAG CircuitOptimizer::constructDAG(QuantumComputation& qc) {
        Qubit highestPhysicalQubit = 0;
        for (const auto& q: qc.initialLayout) {
            if (q.first > highestPhysicalQubit) {
                highestPhysicalQubit = q.first;
            }
        }

        auto dag = DAG(highestPhysicalQubit + 1);

        for (auto& it: qc.ops) {
            if (!it->isStandardOperation()) {
                addNonStandardOperationToDag(dag, &it);
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

    void CircuitOptimizer::addNonStandardOperationToDag(DAG& dag, std::unique_ptr<Operation>* op) {
        const auto& gate = *op;
        // compound operations are added "as-is"
        if (gate->isCompoundOperation()) {
            for (std::size_t i = 0U; i < gate->getNqubits(); ++i) {
                if (gate->actsOn(static_cast<Qubit>(i))) {
                    dag.at(i).push_back(op);
                }
            }
        } else if (gate->isNonUnitaryOperation()) {
            // barriers are not added to a circuit DAG
            if (gate->getType() == Barrier) {
                return;
            }

            for (const auto& b: gate->getTargets()) {
                dag.at(b).push_back(op);
            }
        } else if (gate->isClassicControlledOperation()) {
            auto* cop = dynamic_cast<ClassicControlledOperation*>(gate.get())->getOperation();
            for (const auto& control: cop->getControls()) {
                dag.at(control.qubit).push_back(op);
            }
            for (const auto& target: cop->getTargets()) {
                dag.at(target).push_back(op);
            }
        } else {
            throw QFRException("Unexpected operation encountered");
        }
    }

    void CircuitOptimizer::singleQubitGateFusion(QuantumComputation& qc) {
        static const std::map<qc::OpType, qc::OpType> INVERSE_MAP = {
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

        Qubit highestPhysicalQubit = 0;
        for (const auto& q: qc.initialLayout) {
            if (q.first > highestPhysicalQubit) {
                highestPhysicalQubit = q.first;
            }
        }

        auto dag = DAG(highestPhysicalQubit + 1);

        for (auto& it: qc.ops) {
            if (!it->isStandardOperation()) {
                addNonStandardOperationToDag(dag, &it);
                continue;
            }

            // not a single qubit operation TODO: multiple targets could also be considered here
            if (!it->getControls().empty() || it->getTargets().size() > 1) {
                addToDag(dag, &it);
                continue;
            }

            const auto target = it->getTargets().at(0);

            // first operation
            if (dag.at(target).empty()) {
                addToDag(dag, &it);
                continue;
            }

            auto  dagQubit = dag.at(target);
            auto* op       = dagQubit.back();

            // no single qubit op to fuse with operation
            if (!(*op)->isCompoundOperation() && (!(*op)->getControls().empty() || (*op)->getTargets().size() > 1)) {
                addToDag(dag, &it);
                continue;
            }

            // compound operation
            if ((*op)->isCompoundOperation()) {
                auto* compop = dynamic_cast<CompoundOperation*>(op->get());

                // check if compound operation contains non-single-qubit gates
                std::size_t involvedQubits = 0;
                for (std::size_t q = 0; q < dag.size(); ++q) {
                    if (compop->actsOn(static_cast<Qubit>(q))) {
                        ++involvedQubits;
                    }
                }
                if (involvedQubits > 1) {
                    addToDag(dag, &it);
                    continue;
                }

                // check if inverse
                auto lastop    = (--(compop->end()));
                auto inverseIt = INVERSE_MAP.find((*lastop)->getType());
                // check if current operation is the inverse of the previous operation
                if (inverseIt != INVERSE_MAP.end() && it->getType() == inverseIt->second) {
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
            auto inverseIt = INVERSE_MAP.find((*op)->getType());
            if (inverseIt != INVERSE_MAP.end() && it->getType() == inverseIt->second) {
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

    bool CircuitOptimizer::removeDiagonalGate(DAG& dag, DAGReverseIterators& dagIterators, Qubit idx, DAGReverseIterator& it, qc::Operation* op) {
        // not a diagonal gate
        if (std::find(DIAGONAL_GATES.begin(), DIAGONAL_GATES.end(), op->getType()) == DIAGONAL_GATES.end()) {
            it = dag.at(idx).rend();
            return false;
        }

        if (op->getNcontrols() != 0) {
            // need to check all controls and targets
            bool onlyDiagonalGates = true;
            for (const auto& control: op->getControls()) {
                auto controlQubit = control.qubit;
                if (controlQubit == idx) {
                    continue;
                }
                if (control.type == Control::Type::Neg) {
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
                if (target == idx) {
                    continue;
                }
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
        }
        // set operation to identity so that it can be collected by the removeIdentities pass
        op->setGate(qc::I);
        return true;
    }

    void CircuitOptimizer::removeDiagonalGatesBeforeMeasureRecursive(DAG& dag, DAGReverseIterators& dagIterators, Qubit idx, const DAGReverseIterator& until) {
        // qubit is finished -> consider next qubit
        if (dagIterators.at(idx) == dag.at(idx).rend()) {
            if (idx < static_cast<Qubit>(dag.size() - 1)) {
                removeDiagonalGatesBeforeMeasureRecursive(dag, dagIterators, idx + 1, dag.at(idx + 1).rend());
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
            auto* op = (*it)->get();
            if (op->getType() == Barrier || op->getType() == Snapshot || op->getType() == ShowProbabilities) {
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
                auto* compOp            = dynamic_cast<qc::CompoundOperation*>(op);
                bool  onlyDiagonalGates = true;
                auto  cit               = compOp->rbegin();
                while (cit != compOp->rend()) {
                    auto* cop         = (*cit).get();
                    onlyDiagonalGates = removeDiagonalGate(dag, dagIterators, idx, it, cop);
                    if (!onlyDiagonalGates) {
                        break;
                    }
                    ++cit;
                }
                if (onlyDiagonalGates) {
                    for (size_t q = 0; q < dag.size(); ++q) {
                        if (compOp->actsOn(static_cast<Qubit>(q))) {
                            ++(dagIterators.at(q));
                        }
                    }
                }
            } else if (op->isClassicControlledOperation()) {
                // consider the operation that is classically controlled and proceed as above
                auto*      cop               = dynamic_cast<ClassicControlledOperation*>(op)->getOperation();
                const bool onlyDiagonalGates = removeDiagonalGate(dag, dagIterators, idx, it, cop);
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
        if (dagIterators.at(idx) == dag.at(idx).rend() && idx < static_cast<Qubit>(dag.size() - 1)) {
            removeDiagonalGatesBeforeMeasureRecursive(dag, dagIterators, idx + 1, dag.at(idx + 1).rend());
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

    bool CircuitOptimizer::removeFinalMeasurement(DAG& dag, DAGReverseIterators& dagIterators, Qubit idx, DAGReverseIterator& it, qc::Operation* op) {
        if (op->getNtargets() != 0) {
            // need to check all targets
            bool onlyMeasurments = true;
            for (const auto& target: op->getTargets()) {
                if (target == idx) {
                    continue;
                }
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
        }
        return false;
    }

    void CircuitOptimizer::removeFinalMeasurementsRecursive(DAG& dag, DAGReverseIterators& dagIterators, Qubit idx, const DAGReverseIterator& until) {
        if (dagIterators.at(idx) == dag.at(idx).rend()) { //we reached the end
            if (idx < static_cast<Qubit>(dag.size() - 1)) {
                removeFinalMeasurementsRecursive(dag, dagIterators, idx + 1, dag.at(idx + 1).rend());
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
            auto* op = (*it)->get();
            if (op->getType() == Measure) {
                const bool onlyMeasurment = removeFinalMeasurement(dag, dagIterators, idx, it, op);
                if (onlyMeasurment) {
                    for (const auto& target: op->getTargets()) {
                        if (dagIterators.at(target) == dag.at(target).rend()) {
                            break;
                        }
                        ++(dagIterators.at(target));
                    }
                }
            } else if (op->getType() == Barrier || op->getType() == Snapshot || op->getType() == ShowProbabilities) {
                for (const auto& target: op->getTargets()) {
                    if (dagIterators.at(target) == dag.at(target).rend()) {
                        break;
                    }
                    ++(dagIterators.at(target));
                }
            } else if (op->isCompoundOperation() && op->isNonUnitaryOperation()) {
                // iterate over all gates of compound operation and upon success increase all corresponding iterators
                auto* compOp          = dynamic_cast<qc::CompoundOperation*>(op);
                bool  onlyMeasurement = true;
                auto  cit             = compOp->rbegin();
                while (cit != compOp->rend()) {
                    auto* cop = (*cit).get();
                    if (cop->getNtargets() > 0 && cop->getTargets()[0] != idx) {
                        ++cit;
                        continue;
                    }
                    onlyMeasurement = removeFinalMeasurement(dag, dagIterators, idx, it, cop);
                    if (!onlyMeasurement) {
                        break;
                    }
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
        if (dagIterators.at(idx) == dag.at(idx).rend() && idx < static_cast<Qubit>(dag.size() - 1)) {
            removeFinalMeasurementsRecursive(dag, dagIterators, idx + 1, dag.at(idx + 1).rend());
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
                    const auto targets = (*it)->getTargets();
                    const auto nqubits = (*it)->getNqubits();
                    it                 = qc.ops.erase(it);
                    it                 = qc.ops.insert(it, std::make_unique<StandardOperation>(nqubits, Control{targets[0]}, targets[1], qc::X));
                    if (isDirectedArchitecture) {
                        it = qc.ops.insert(it, std::make_unique<StandardOperation>(nqubits, targets[0], qc::H));
                        it = qc.ops.insert(it, std::make_unique<StandardOperation>(nqubits, targets[1], qc::H));
                        it = qc.ops.insert(it, std::make_unique<StandardOperation>(nqubits, Control{targets[0]}, targets[1], qc::X));
                        it = qc.ops.insert(it, std::make_unique<StandardOperation>(nqubits, targets[0], qc::H));
                        it = qc.ops.insert(it, std::make_unique<StandardOperation>(nqubits, targets[1], qc::H));
                    } else {
                        it = qc.ops.insert(it, std::make_unique<StandardOperation>(nqubits, Control{targets[1]}, targets[0], qc::X));
                    }
                    it = qc.ops.insert(it, std::make_unique<StandardOperation>(nqubits, Control{targets[0]}, targets[1], qc::X));
                } else {
                    ++it;
                }
            } else if ((*it)->isCompoundOperation()) {
                auto* compOp = dynamic_cast<qc::CompoundOperation*>((*it).get());
                auto  cit    = compOp->begin();
                while (cit != compOp->end()) {
                    if ((*cit)->isStandardOperation() && (*cit)->getType() == qc::SWAP) {
                        const auto targets = (*cit)->getTargets();
                        const auto nqubits = compOp->getNqubits();
                        cit                = compOp->erase(cit);
                        cit                = compOp->insert<StandardOperation>(cit, nqubits, Control{targets[0]}, targets[1], qc::X);
                        if (isDirectedArchitecture) {
                            cit = compOp->insert<StandardOperation>(cit, nqubits, targets[0], qc::H);
                            cit = compOp->insert<StandardOperation>(cit, nqubits, targets[1], qc::H);
                            cit = compOp->insert<StandardOperation>(cit, nqubits, Control{targets[0]}, targets[1], qc::X);
                            cit = compOp->insert<StandardOperation>(cit, nqubits, targets[0], qc::H);
                            cit = compOp->insert<StandardOperation>(cit, nqubits, targets[1], qc::H);
                        } else {
                            cit = compOp->insert<StandardOperation>(cit, nqubits, Control{targets[1]}, targets[0], qc::X);
                        }
                        cit = compOp->insert<StandardOperation>(cit, nqubits, Control{targets[0]}, targets[1], qc::X);
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
        auto replacementMap = std::map<Qubit, Qubit>();
        auto it             = qc.ops.begin();
        while (it != qc.ops.end()) {
            if ((*it)->getType() == qc::Reset) {
                for (const auto& target: (*it)->getTargets()) {
                    auto indexAddQubit = static_cast<Qubit>(qc.getNqubits());
                    qc.addQubit(indexAddQubit, indexAddQubit, indexAddQubit);
                    auto oldReset = replacementMap.find(target);
                    if (oldReset != replacementMap.end()) {
                        oldReset->second = indexAddQubit;
                    } else {
                        replacementMap.try_emplace(target, indexAddQubit);
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
                                auto indexAddQubit = static_cast<Qubit>(qc.getNqubits());
                                qc.addQubit(indexAddQubit, indexAddQubit, indexAddQubit);
                                auto oldReset = replacementMap.find(compTarget);
                                if (oldReset != replacementMap.end()) {
                                    oldReset->second = indexAddQubit;
                                } else {
                                    replacementMap.try_emplace(compTarget, indexAddQubit);
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

    void CircuitOptimizer::changeTargets(Targets& targets, const std::map<Qubit, Qubit>& replacementMap) {
        for (auto& target: targets) {
            auto newTargetIt = replacementMap.find(target);
            if (newTargetIt != replacementMap.end()) {
                target = newTargetIt->second;
            }
        }
    }

    void CircuitOptimizer::changeControls(Controls& controls, const std::map<Qubit, Qubit>& replacementMap) {
        if (controls.empty() || replacementMap.empty()) {
            return;
        }

        // iterate over the replacement map and see if any control matches
        for (const auto& [from, to]: replacementMap) {
            auto controlIt = controls.find(from);
            if (controlIt != controls.end()) {
                const auto controlType = controlIt->type;
                controls.erase(controlIt);
                controls.insert(Control{to, controlType});
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
        std::unordered_map<Qubit, std::size_t> qubitsToAddMeasurements{};
        auto                                   it = qc.begin();
        while (it != qc.end()) {
            if ((*it)->getType() == qc::Measure) {
                auto*      measurement = dynamic_cast<qc::NonUnitaryOperation*>(it->get());
                const auto targets     = measurement->getTargets();
                const auto classics    = measurement->getClassics();

                if (targets.size() != 1 && classics.size() != 1) {
                    throw QFRException("Deferring measurements with more than 1 target is not yet supported. Try decomposing your measurements.");
                }

                // if this is the last operation, nothing has to be done
                if (*it == qc.ops.back()) {
                    break;
                }

                // remember q-> c for adding measurements later
                qubitsToAddMeasurements[targets[0]] = classics[0];

                // remove the measurement from the vector of operations
                it = qc.erase(it);

                // starting from the next operation after the measurement (if there is any)
                auto opIt                  = it;
                auto currentInsertionPoint = it;

                // iterate over all subsequent operations
                while (opIt != qc.end()) {
                    const auto* operation = opIt->get();
                    if (operation->isUnitary() || operation->getType() == qc::Barrier) {
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
                        }

                        ++currentInsertionPoint;
                        ++opIt;
                        continue;
                    }

                    if (operation->isClassicControlledOperation()) {
                        auto*       classicOp       = dynamic_cast<qc::ClassicControlledOperation*>((*opIt).get());
                        const auto& controlRegister = classicOp->getControlRegister();
                        const auto& expectedValue   = classicOp->getExpectedValue();

                        if (controlRegister.second != 1 && expectedValue <= 1) {
                            throw QFRException("Classic-controlled operations targeted at more than one bit are currently not supported. Try decomposing the operation into individual contributions.");
                        }

                        // if this is not the classical bit that is measured, continue
                        if (controlRegister.first == static_cast<Qubit>(classics.at(0))) {
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
                            auto controlType  = (expectedValue == 1) ? Control::Type::Pos : Control::Type::Neg;
                            controls.emplace(Control{controlQubit, controlType});

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
        Qubit highestPhysicalQubit = 0;
        for (const auto& q: qc.initialLayout) {
            if (q.first > highestPhysicalQubit) {
                highestPhysicalQubit = q.first;
            }
        }

        auto dag = DAG(highestPhysicalQubit + 1);

        bool hasMeasurements = false;

        for (auto& it: qc.ops) {
            if (!it->isStandardOperation()) {
                if (it->isNonUnitaryOperation()) {
                    // whenever a reset operation is encountered the circuit has to be dynamic
                    if (it->getType() == Reset) {
                        return true;
                    }

                    // record whether the circuit contains measurements
                    if (it->getType() == Measure) {
                        hasMeasurements = true;
                    }

                    for (const auto& b: it->getTargets()) {
                        dag.at(b).push_back(&it);
                    }
                } else if (it->isClassicControlledOperation()) {
                    // whenever a classic-controlled operation is encountered the circuit has to be dynamic
                    return true;
                } else if (it->isCompoundOperation()) {
                    auto* compOp = dynamic_cast<CompoundOperation*>(it.get());
                    for (auto& op: *compOp) {
                        if (op->getType() == Reset || op->isClassicControlledOperation()) {
                            return true;
                        }

                        if (op->getType() == Measure) {
                            hasMeasurements = true;
                        }

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

        if (!hasMeasurements) {
            return false;
        }

        for (const auto& qubitDAG: dag) {
            bool operation   = false;
            bool measurement = false;
            for (auto it = qubitDAG.rbegin(); it != qubitDAG.rend(); ++it) {
                auto* op = *it;
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
            if (measurement && operation) {
                return true;
            }
        }

        return false;
    }

    /// this method can be used to reorder the operations of a given quantum computation in order to get a canonical ordering
    /// it uses iterative breadth-first search starting from the topmost qubit
    void CircuitOptimizer::reorderOperations(QuantumComputation& qc) {
        auto dag = constructDAG(qc);

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
        const auto msq  = dag.size() - 1;
        bool       done = false;
        while (!done) {
            // assume that everything is done
            done = true;

            // iterate over qubits in reverse order
            for (auto q = static_cast<std::make_signed_t<Qubit>>(msq); q >= 0; --q) {
                // nothing to be done for this qubit
                if (dagIterators.at(static_cast<std::size_t>(q)) == dag.at(static_cast<std::size_t>(q)).end()) {
                    continue;
                }
                done = false;

                // get the current operation on the qubit
                auto& it = dagIterators.at(static_cast<std::size_t>(q));
                auto& op = **it;

                // warning for classically controlled operations
                if (op->getType() == ClassicControlled) {
                    std::cerr << "Caution! Reordering operations might not work if the circuit contains classically controlled operations" << std::endl;
                }

                if (op->getType() == Barrier || op->getType() == Snapshot || op->getType() == ShowProbabilities) {
                    ++it;
                    continue;
                }

                // check whether the gate can be scheduled, i.e. whether all qubits it acts on are at this operation
                bool              executable = true;
                std::vector<bool> actsOn(dag.size());
                actsOn[static_cast<std::size_t>(q)] = true;
                for (std::size_t i = 0; i < dag.size(); ++i) {
                    // actually check in reverse order
                    const auto qb = static_cast<std::make_signed_t<Qubit>>(dag.size() - 1 - i);
                    if (qb != q && op->actsOn(static_cast<Qubit>(qb))) {
                        actsOn[static_cast<std::size_t>(qb)] = true;

                        assert(dagIterators.at(static_cast<std::size_t>(qb)) != dag.at(static_cast<std::size_t>(qb)).end());
                        // check whether operation is executable for the currently considered qubit
                        if (*dagIterators.at(static_cast<std::size_t>(qb)) != *it) {
                            executable = false;
                            break;
                        }
                    }
                }

                // continue, if this gate is not yet executable
                if (!executable) {
                    continue;
                }

                // gate is executable, move it to the new vector
                ops.emplace_back(std::move(op));

                // now increase all corresponding iterators
                for (std::size_t i = 0; i < dag.size(); ++i) {
                    if (actsOn[i]) {
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

    void CircuitOptimizer::flattenOperations(QuantumComputation& qc) {
        auto it = qc.begin();
        while (it != qc.end()) {
            if ((*it)->isCompoundOperation()) {
                it = flattenCompoundOperation(qc.ops, it);
            } else {
                ++it;
            }
        }
    }

    CircuitOptimizer::Iterator CircuitOptimizer::flattenCompoundOperation(std::vector<std::unique_ptr<Operation>>& ops, CircuitOptimizer::Iterator it) {
        assert((*it)->isCompoundOperation());
        auto& op   = dynamic_cast<qc::CompoundOperation&>(**it);
        auto  opIt = op.begin();
        while (opIt != op.end()) {
            if ((*opIt)->isCompoundOperation()) {
                // recursively flatten compound operations
                opIt = flattenCompoundOperation(op.getOps(), opIt);
                --opIt;
            } else {
                // move the operation from the compound operation in front of the compound operation in the flattened container.
                // `it` then points to the newly inserted element
                it = ops.insert(it, std::move(*opIt));
                // advance the operation iterator to point past the now moved-from element in the compound operation
                ++opIt;
                // advance the general iterator to again point to the compound operation
                ++it;
            }
        }
        // whenever all the operations have been processed, `it` points to the compound operation and `opIt` to `op.end()`
        // the compound operation can now be deleted safely
        return ops.erase(it);
    }

    void CircuitOptimizer::cancelCNOTs(QuantumComputation& qc) {
        Qubit highestPhysicalQubit = 0;
        for (const auto& q: qc.initialLayout) {
            if (q.first > highestPhysicalQubit) {
                highestPhysicalQubit = q.first;
            }
        }

        auto dag = DAG(highestPhysicalQubit + 1U);

        for (auto& it: qc.ops) {
            if (!it->isStandardOperation()) {
                addNonStandardOperationToDag(dag, &it);
                continue;
            }

            // check whether the operation is a CNOT or SWAP gate
            const auto isCNOT = (it->getType() == X && it->getNcontrols() == 1U && it->getControls().begin()->type == Control::Type::Pos);
            const auto isSWAP = (it->getType() == SWAP && it->getNcontrols() == 0U);

            if (!isCNOT && !isSWAP) {
                addToDag(dag, &it);
                continue;
            }

            const Qubit q0 = it->getTargets().at(0);
            const Qubit q1 = isSWAP ? it->getTargets().at(1) : it->getControls().begin()->qubit;

            // first operation
            if (dag.at(q0).empty() || dag.at(q1).empty()) {
                addToDag(dag, &it);
                continue;
            }

            auto* op0 = dag.at(q0).back()->get();
            auto* op1 = dag.at(q1).back()->get();

            // check whether it's the same operation at both qubits
            if (op0 != op1) {
                addToDag(dag, &it);
                continue;
            }

            // check whether the operation is a CNOT or SWAP gate
            const auto prevOpIsCNOT = (op0->getType() == X && op0->getNcontrols() == 1U && op0->getControls().begin()->type == Control::Type::Pos);
            const auto prevOpIsSWAP = (op0->getType() == SWAP && op0->getNcontrols() == 0U);

            if (!prevOpIsCNOT && !prevOpIsSWAP) {
                addToDag(dag, &it);
                continue;
            }

            const Qubit prevQ0 = op0->getTargets().at(0);
            const Qubit prevQ1 = prevOpIsSWAP ? op0->getTargets().at(1) : op0->getControls().begin()->qubit;

            if (isCNOT && prevOpIsCNOT) {
                // two identical CNOT gates cancel each other
                if (q0 == prevQ0 && q1 == prevQ1) {
                    dag.at(q0).pop_back();
                    dag.at(q1).pop_back();
                    op0->setGate(I);
                    op0->setControls({});
                    it->setGate(I);
                    it->setControls({});
                } else {
                    // two CNOTs with alternating controls and targets
                    // check whether there is a third one which would make this a SWAP gate

                    auto prevPrevOp0It = ++(dag.at(q0).rbegin());
                    auto prevPrevOp1It = ++(dag.at(q1).rbegin());
                    // check whether there is another operation
                    if (prevPrevOp0It == dag.at(q0).rend() || prevPrevOp1It == dag.at(q1).rend()) {
                        addToDag(dag, &it);
                        continue;
                    }

                    auto* prevPrevOp0 = (*prevPrevOp0It)->get();
                    auto* prevPrevOp1 = (*prevPrevOp1It)->get();

                    if (prevPrevOp0 != prevPrevOp1) {
                        addToDag(dag, &it);
                        continue;
                    }

                    // check whether the operation is a CNOT
                    const auto prevPrevOpIsCNOT = (prevPrevOp0->getType() == X && prevPrevOp0->getNcontrols() == 1U && prevPrevOp0->getControls().begin()->type == Control::Type::Pos);

                    if (!prevPrevOpIsCNOT) {
                        addToDag(dag, &it);
                        continue;
                    }

                    const Qubit prevPrevQ0 = prevPrevOp0->getTargets().at(0);
                    const Qubit prevPrevQ1 = prevPrevOp0->getControls().begin()->qubit;

                    if (q0 == prevPrevQ0 && q1 == prevPrevQ1) {
                        // SWAP gate identified
                        prevPrevOp0->setGate(SWAP);
                        prevPrevOp0->setControls({});
                        if (prevQ0 > prevQ1) {
                            prevPrevOp0->setTargets({prevQ1, prevQ0});
                        } else {
                            prevPrevOp0->setTargets({prevQ0, prevQ1});
                        }
                        op0->setGate(I);
                        op0->setControls({});
                        it->setGate(I);
                        it->setControls({});
                        dag.at(q0).pop_back();
                        dag.at(q1).pop_back();
                    } else {
                        addToDag(dag, &it);
                        continue;
                    }
                }
                continue;
            }

            if (isSWAP && prevOpIsSWAP) {
                // two identical SWAP gates cancel each other
                if (std::set{q0, q1} == std::set{prevQ0, prevQ1}) {
                    dag.at(q0).pop_back();
                    dag.at(q1).pop_back();
                    op0->setGate(I);
                    op0->setControls({});
                    it->setGate(I);
                    it->setControls({});
                } else {
                    addToDag(dag, &it);
                }
                continue;
            }

            if (isCNOT && prevOpIsSWAP) {
                // SWAP followed by a CNOT is equivalent to two CNOTs
                op0->setGate(X);
                op0->setControls({Control{q1}});
                op0->setTargets({q0});
                it->setControls({Control{q0}});
                it->setTargets({q1});
                addToDag(dag, &it);
                continue;
            }

            if (isSWAP && prevOpIsCNOT) {
                // CNOT followed by a SWAP is equivalent to two CNOTs
                op0->setControls({Control{prevQ0}});
                op0->setTargets({prevQ1});
                it->setGate(X);
                it->setControls({Control{prevQ1}});
                it->setTargets({prevQ0});
                addToDag(dag, &it);
                continue;
            }
        }

        removeIdentities(qc);
    }
} // namespace qc
