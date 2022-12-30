/*
* This file is part of MQT QFR library which is released under the MIT license.
* See file README.md or go to https://www.cda.cit.tum.de/research/quantum/ for more information.
*/

#include "zx/FunctionalityConstruction.hpp"

#include "Definitions.hpp"
#include "Rational.hpp"
#include "ZXDiagram.hpp"

#include <algorithm>
#include <string>
#include <variant>
#include <vector>

namespace zx {

    bool FunctionalityConstruction::checkSwap(const op_it& it, const op_it& end, const Qubit ctrl,
                                              const Qubit            target,
                                              const qc::Permutation& p) {
        if (it + 1 != end && it + 2 != end) {
            const auto& op1 = *(it + 1);
            const auto& op2 = *(it + 2);
            if (op1->getType() == qc::OpType::X && op2->getType() == qc::OpType::X &&
                op1->getNcontrols() == 1 && op2->getNcontrols() == 1) {
                const auto tar1  = p.at(op1->getTargets().front());
                const auto tar2  = p.at(op2->getTargets().front());
                const auto ctrl1 = p.at((*op1->getControls().begin()).qubit);
                const auto ctrl2 = p.at((*op2->getControls().begin()).qubit);
                return ctrl == static_cast<Qubit>(tar1) && tar1 == ctrl2 && target == static_cast<Qubit>(ctrl1) && ctrl1 == tar2;
            }
        }
        return false;
    }

    void FunctionalityConstruction::addZSpider(ZXDiagram& diag, const zx::Qubit qubit,
                                               std::vector<Vertex>& qubits,
                                               const PiExpression& phase, const EdgeType type) {
        const auto q = static_cast<std::size_t>(qubit);

        auto newVertex = diag.addVertex(qubit, diag.getVData(qubits[q]).value().col + 1, phase, VertexType::Z);

        diag.addEdge(qubits[q], newVertex, type);
        qubits[q] = newVertex;
    }

    void FunctionalityConstruction::addXSpider(ZXDiagram& diag, const Qubit qubit,
                                               std::vector<Vertex>& qubits,
                                               const PiExpression& phase, const EdgeType type) {
        const auto q         = static_cast<std::size_t>(qubit);
        const auto newVertex = diag.addVertex(
                qubit, diag.getVData(qubits[q]).value().col + 1, phase,
                VertexType::X);
        diag.addEdge(qubits[q], newVertex, type);
        qubits[q] = newVertex;
    }

    void FunctionalityConstruction::addCnot(ZXDiagram& diag, const Qubit ctrl, const Qubit target,
                                            std::vector<Vertex>& qubits) {
        addZSpider(diag, ctrl, qubits);
        addXSpider(diag, target, qubits);
        diag.addEdge(qubits[static_cast<std::size_t>(ctrl)], qubits[static_cast<std::size_t>(target)]);
    }

    void
    FunctionalityConstruction::addCphase(ZXDiagram& diag, const PiExpression& phase,
                                         const Qubit ctrl, const Qubit target,
                                         std::vector<Vertex>& qubits) {
        auto newConst = phase.getConst() / 2;
        auto newPhase = phase / 2.0;
        newPhase.setConst(newConst);
        addZSpider(diag, ctrl, qubits, newPhase); //todo maybe should provide a method for int division
        addCnot(diag, ctrl, target, qubits);
        addZSpider(diag, target, qubits, -newPhase);
        addCnot(diag, ctrl, target, qubits);
        addZSpider(diag, target, qubits, newPhase);
    }

    void FunctionalityConstruction::addSwap(ZXDiagram& diag, const Qubit target, const Qubit target2,
                                            std::vector<Vertex>& qubits) {
        const auto c = static_cast<std::size_t>(target);
        const auto t = static_cast<std::size_t>(target2);

        const auto s0 = qubits[t];
        const auto s1 = qubits[c];

        const auto t0 = diag.addVertex(target2, diag.getVData(qubits[t]).value().col + 1);
        const auto t1 = diag.addVertex(target, diag.getVData(qubits[t]).value().col + 1);
        diag.addEdge(s0, t1);
        diag.addEdge(s1, t0);
        qubits[t] = t0;
        qubits[c] = t1;
    }

    void FunctionalityConstruction::addCcx(ZXDiagram& diag, const Qubit ctrl0, const Qubit ctrl1, const Qubit target,
                                           std::vector<Vertex>& qubits) {
        addZSpider(diag, target, qubits, PiExpression(), EdgeType::Hadamard);
        addCnot(diag, ctrl1, target, qubits);
        addZSpider(diag, target, qubits, PiExpression(PiRational(-1, 4)));
        addCnot(diag, ctrl0, target, qubits);
        addZSpider(diag, target, qubits, PiExpression(PiRational(1, 4)));
        addCnot(diag, ctrl1, target, qubits);
        addZSpider(diag, ctrl1, qubits, PiExpression(PiRational(1, 4)));
        addZSpider(diag, target, qubits, PiExpression(PiRational(-1, 4)));
        addCnot(diag, ctrl0, target, qubits);
        addZSpider(diag, target, qubits, PiExpression(PiRational(1, 4)));
        addCnot(diag, ctrl0, ctrl1, qubits);
        addZSpider(diag, ctrl0, qubits, PiExpression(PiRational(1, 4)));
        addZSpider(diag, ctrl1, qubits, PiExpression(PiRational(-1, 4)));
        addZSpider(diag, target, qubits, PiExpression(PiRational(0, 1)),
                   EdgeType::Hadamard);
        addCnot(diag, ctrl0, ctrl1, qubits);
    }

    FunctionalityConstruction::op_it FunctionalityConstruction::parseOp(ZXDiagram& diag, op_it it, op_it end,
                                                                        std::vector<Vertex>& qubits, const qc::Permutation& p) {
        const auto& op = *it;
        // barrier statements are ignored
        if (op->getType() == qc::OpType::Barrier) {
            return it + 1;
        }

        if (!op->isControlled()) {
            // single qubit gates
            const auto target = static_cast<zx::Qubit>(p.at(op->getTargets().front()));
            switch (op->getType()) {
                case qc::OpType::Z:
                    addZSpider(diag, target, qubits,
                               PiExpression(PiRational(1, 1)));
                    break;
                case qc::OpType::RZ: {
                    const auto& param = parseParam(op.get(), 0);
                    diag.addGlobalPhase(-param / 2.0);

                    addZSpider(
                            diag, target, qubits,
                            parseParam(op.get(), 0));
                    break;
                }
                case qc::OpType::Phase:
                    addZSpider(diag, target, qubits, parseParam(op.get(), 0));
                    break;
                case qc::OpType::X:
                    addXSpider(diag, target, qubits,
                               PiExpression(PiRational(1, 1)));
                    break;
                case qc::OpType::RX:
                    addXSpider(diag, target, qubits, parseParam(op.get(), 0));
                    break;
                case qc::OpType::Y:
                    diag.addGlobalPhase(PiExpression{-PiRational(1, 2)});

                    addZSpider(diag, target, qubits,
                               PiExpression(PiRational(1, 1)));
                    addXSpider(diag, target, qubits,
                               PiExpression(PiRational(1, 1)));
                    break;
                case qc::OpType::RY:
                    diag.addGlobalPhase(PiExpression(-PiRational(op->getParameter().front()) / 2 +
                                                     PiRational(1, 2) + PiRational(3, 2)));

                    addXSpider(diag, target, qubits,
                               PiExpression(PiRational(1, 2)));
                    addZSpider(diag, target, qubits,
                               parseParam(op.get(), 0) + PiRational(1, 1));
                    addXSpider(diag, target, qubits,
                               PiExpression(PiRational(1, 2)));
                    addZSpider(diag, target, qubits,
                               PiExpression(PiRational(3, 1)));
                    break;
                case qc::OpType::T:
                    addZSpider(diag, target, qubits,
                               PiExpression(PiRational(1, 4)));
                    break;
                case qc::OpType::Tdag:
                    addZSpider(diag, target, qubits,
                               PiExpression(PiRational(-1, 4)));
                    break;
                case qc::OpType::S:
                    addZSpider(diag, target, qubits,
                               PiExpression(PiRational(1, 2)));
                    break;
                case qc::OpType::Sdag:
                    addZSpider(diag, target, qubits,
                               PiExpression(PiRational(-1, 2)));
                    break;
                case qc::OpType::U2:
                    addZSpider(diag, target, qubits,
                               parseParam(op.get(), 0) - PiRational(1, 2));
                    addXSpider(diag, target, qubits,
                               PiExpression(PiRational(1, 2)));
                    addZSpider(diag, target, qubits,
                               parseParam(op.get(), 1) + PiRational(1, 2));
                    break;
                case qc::OpType::U3:
                    addZSpider(diag, target, qubits, parseParam(op.get(), 0));
                    addXSpider(diag, target, qubits,
                               PiExpression(PiRational(1, 2)));
                    addZSpider(diag, target, qubits,
                               parseParam(op.get(), 2) + PiRational(1, 1));
                    addXSpider(diag, target, qubits,
                               PiExpression(PiRational(1, 2)));
                    addZSpider(diag, target, qubits,
                               parseParam(op.get(), 1) + PiRational(3, 1));
                    break;
                case qc::OpType::SWAP: {
                    const auto target2 = static_cast<zx::Qubit>(p.at(op->getTargets()[1]));
                    addSwap(diag, target, target2, qubits);
                    break;
                }
                case qc::OpType::iSWAP: {
                    const auto target2 = static_cast<zx::Qubit>(p.at(op->getTargets()[1]));
                    addZSpider(diag, target, qubits, PiExpression(PiRational(1, 2)));
                    addZSpider(diag, target2, qubits, PiExpression(PiRational(1, 2)));
                    addZSpider(diag, target, qubits, PiExpression(),
                               EdgeType::Hadamard);
                    addCnot(diag, target, target2, qubits); // NOLINT (readability-suspicious-call-argument)
                    addCnot(diag, target2, target, qubits);
                    addZSpider(diag, target2, qubits, PiExpression(),
                               EdgeType::Hadamard);
                    break;
                }
                case qc::OpType::H:
                    addZSpider(diag, target, qubits, PiExpression(),
                               EdgeType::Hadamard);
                    break;
                case qc::OpType::Measure:
                case qc::OpType::I:
                    break;
                case qc::OpType::SX:
                    addXSpider(diag, target, qubits, PiExpression(PiRational(1, 2)));
                    break;
                case qc::OpType::SXdag:
                    addXSpider(diag, target, qubits, PiExpression(PiRational(-1, 2)));
                    break;
                default:
                    throw ZXException("Unsupported Operation: " +
                                      qc::toString(op->getType()));
            }
        } else if (op->getNcontrols() == 1 && op->getNtargets() == 1) {
            // two-qubit controlled gates
            const auto target = static_cast<zx::Qubit>(p.at(op->getTargets().front()));
            const auto ctrl   = static_cast<zx::Qubit>(p.at((*op->getControls().begin()).qubit));
            switch (op->getType()) { // TODO: any gate can be controlled
                case qc::OpType::X:
                    // check if swap
                    if (checkSwap(it, end, ctrl, target, p)) {
                        addSwap(diag, target, ctrl, qubits);
                        return it + 3;
                    } else {
                        addCnot(diag, ctrl, target, qubits);
                    }

                    break;
                case qc::OpType::Z:
                    addZSpider(diag, target, qubits, PiExpression(),
                               EdgeType::Hadamard);
                    addCnot(diag, ctrl, target, qubits);
                    addZSpider(diag, target, qubits, PiExpression(),
                               EdgeType::Hadamard);

                    break;

                case qc::OpType::I:
                    break;

                case qc::OpType::Phase:
                    addCphase(diag, parseParam(op.get(), 0), ctrl, target,
                              qubits);
                    break;

                case qc::OpType::T:
                    addCphase(diag, zx::PiExpression{PiRational(1, 4)}, ctrl,
                              target, qubits);
                    break;

                case qc::OpType::S:
                    addCphase(diag, zx::PiExpression{PiRational(1, 2)}, ctrl,
                              target, qubits);
                    break;

                case qc::OpType::Tdag:
                    addCphase(diag, zx::PiExpression{PiRational(-1, 4)}, ctrl,
                              target, qubits);
                    break;

                case qc::OpType::Sdag:
                    addCphase(diag, zx::PiExpression{PiRational(-1, 2)}, ctrl,
                              target, qubits);
                    break;
                default:
                    throw ZXException("Unsupported Controlled Operation: " +
                                      qc::toString(op->getType()));
            }
        } else if (op->getNcontrols() == 2) {
            // three-qubit controlled gates (ccx or ccz)
            Qubit      ctrl0  = 0;
            Qubit      ctrl1  = 0;
            const auto target = static_cast<Qubit>(p.at(op->getTargets().front()));
            int        i      = 0;
            for (const auto& ctrl: op->getControls()) {
                if (i++ == 0) {
                    ctrl0 = static_cast<Qubit>(p.at(ctrl.qubit));
                } else {
                    ctrl1 = static_cast<Qubit>(p.at(ctrl.qubit));
                }
            }
            switch (op->getType()) {
                case qc::OpType::X:
                    addCcx(diag, ctrl0, ctrl1, target, qubits);
                    break;

                case qc::OpType::Z:
                    addZSpider(diag, target, qubits, PiExpression(),
                               EdgeType::Hadamard);
                    addCcx(diag, ctrl0, ctrl1, target, qubits);
                    addZSpider(diag, target, qubits, PiExpression(),
                               EdgeType::Hadamard);
                    break;
                default:
                    throw ZXException("Unsupported Multi-control operation: " +
                                      qc::toString(op->getType()));
            }
        } else {
            throw ZXException("Unsupported Multi-control operation (" + std::to_string(op->getNcontrols()) + " ctrls)" + qc::toString(op->getType()));
        }
        return it + 1;
    }

    ZXDiagram FunctionalityConstruction::buildFunctionality(const qc::QuantumComputation* qc) {
        ZXDiagram           diag(qc->getNqubits());
        std::vector<Vertex> qubits(qc->getNqubits());
        for (std::size_t i = 0; i < qc->getNqubits(); ++i) {
            diag.removeEdge(i, i + qc->getNqubits());
            qubits[i] = i;
        }

        for (auto it = qc->cbegin(); it != qc->cend();) {
            const auto& op = *it;

            if (op->getType() == qc::OpType::Compound) {
                const auto* compOp = dynamic_cast<qc::CompoundOperation*>(op.get());
                for (auto subIt = compOp->cbegin(); subIt != compOp->cend();) {
                    subIt = parseOp(diag, subIt, compOp->cend(), qubits, qc->initialLayout);
                }
                ++it;
            } else {
                it = parseOp(diag, it, qc->cend(), qubits, qc->initialLayout);
            }
        }

        for (std::size_t i = 0; i < qubits.size(); ++i) {
            diag.addEdge(qubits[i], diag.getOutput(i));
        }
        return diag;
    }
    bool FunctionalityConstruction::transformableToZX(const qc::QuantumComputation* qc) {
        return std::all_of(qc->cbegin(), qc->cend(), [&](const auto& op) { return transformableToZX(op.get()); });
    }

    bool FunctionalityConstruction::transformableToZX(const qc::Operation* op) {
        if (op->getType() == qc::OpType::Compound) {
            const auto* compOp = dynamic_cast<const qc::CompoundOperation*>(op);

            return std::all_of(compOp->cbegin(), compOp->cend(), [&](const auto& operation) { return transformableToZX(operation.get()); });
        }

        if (op->getType() == qc::OpType::Barrier) {
            return true;
        }

        if (!op->isControlled()) {
            switch (op->getType()) {
                case qc::OpType::Z:
                case qc::OpType::RZ:
                case qc::OpType::Phase:
                case qc::OpType::X:
                case qc::OpType::RX:
                case qc::OpType::Y:
                case qc::OpType::RY:
                case qc::OpType::T:
                case qc::OpType::Tdag:
                case qc::OpType::S:
                case qc::OpType::Sdag:
                case qc::OpType::U2:
                case qc::OpType::U3:
                case qc::OpType::SWAP:
                case qc::OpType::iSWAP:
                case qc::OpType::H:
                case qc::OpType::Measure:
                case qc::OpType::I:
                case qc::OpType::SX:
                case qc::OpType::SXdag:
                    return true;
                default:
                    return false;
            }
        } else if (op->getNcontrols() == 1 && op->getNtargets() == 1) {
            switch (op->getType()) { // TODO: any gate can be controlled
                case qc::OpType::X:
                case qc::OpType::Z:
                case qc::OpType::I:
                case qc::OpType::Phase:
                case qc::OpType::T:
                case qc::OpType::S:
                case qc::OpType::Tdag:
                case qc::OpType::Sdag:
                    return true;

                default:
                    return false;
            }
        } else if (op->getNcontrols() == 2) {
            switch (op->getType()) {
                case qc::OpType::X:
                case qc::OpType::Z:
                    return true;
                default:
                    return false;
            }
        }
        return false;
    }

    PiExpression FunctionalityConstruction::parseParam(const qc::Operation* op,
                                                       const std::size_t    i) {
        if (const auto* symbOp = dynamic_cast<const qc::SymbolicOperation*>(op)) {
            return toPiExpr(symbOp->getParameter(i));
        }
        return PiExpression{zx::PiRational{op->getParameter().at(i)}};
    }
    PiExpression FunctionalityConstruction::toPiExpr(const qc::SymbolOrNumber& param) {
        if (std::holds_alternative<double>(param)) {
            return zx::PiExpression{
                    zx::PiRational{std::get<double>(param)}};
        }
        return std::get<qc::Symbolic>(param).convert<zx::PiRational>();
    }
} // namespace zx
