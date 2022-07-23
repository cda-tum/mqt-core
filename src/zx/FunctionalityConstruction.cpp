#include "zx/FunctionalityConstruction.hpp"

#include "Definitions.hpp"
#include "Rational.hpp"
#include "ZXDiagram.hpp"

#include <algorithm>
#include <iostream>
#include <string>
#include <variant>
#include <vector>

namespace zx {

    bool FunctionalityConstruction::checkSwap(const op_it& it, const op_it& end, Qubit ctrl,
                                              Qubit                  target,
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
                return ctrl == tar1 && tar1 == ctrl2 && target == ctrl1 && ctrl1 == tar2;
            }
        }
        return false;
    }

    void FunctionalityConstruction::addZSpider(ZXDiagram& diag, zx::Qubit qubit,
                                               std::vector<Vertex>& qubits,
                                               const PiExpression& phase, EdgeType type) {
        auto new_vertex = diag.addVertex(
                qubit, diag.getVData(qubits[qubit]).value().col + 1, phase,
                VertexType::Z);

        diag.addEdge(qubits[qubit], new_vertex, type);
        qubits[qubit] = new_vertex;
    }

    void FunctionalityConstruction::addXSpider(ZXDiagram& diag, Qubit qubit,
                                               std::vector<Vertex>& qubits,
                                               const PiExpression& phase, EdgeType type) {
        const auto new_vertex = diag.addVertex(
                qubit, diag.getVData(qubits[qubit]).value().col + 1, phase,
                VertexType::X);
        diag.addEdge(qubits[qubit], new_vertex, type);
        qubits[qubit] = new_vertex;
    }

    void FunctionalityConstruction::addCnot(ZXDiagram& diag, Qubit ctrl, Qubit target,
                                            std::vector<Vertex>& qubits) {
        addZSpider(diag, ctrl, qubits);
        addXSpider(diag, target, qubits);
        diag.addEdge(qubits[ctrl], qubits[target]);
    }

    void
    FunctionalityConstruction::addCphase(ZXDiagram& diag, const PiExpression& phase,
                                         Qubit ctrl, Qubit target,
                                         std::vector<Vertex>& qubits) {
        auto new_const = phase.getConst() / 2;
        auto new_phase = phase / 2.0;
        new_phase.setConst(new_const);
        addZSpider(diag, ctrl, qubits, new_phase); //todo maybe should provide a method for int division
        addCnot(diag, ctrl, target, qubits);
        addZSpider(diag, target, qubits, -new_phase);
        addCnot(diag, ctrl, target, qubits);
        addZSpider(diag, target, qubits, new_phase);
    }

    void FunctionalityConstruction::addSwap(ZXDiagram& diag, Qubit ctrl, Qubit target,
                                            std::vector<Vertex>& qubits) {
        const auto s0 = qubits[target];
        const auto s1 = qubits[ctrl];

        const auto t0 = diag.addVertex(target, diag.getVData(qubits[target]).value().col + 1);
        const auto t1 = diag.addVertex(ctrl, diag.getVData(qubits[target]).value().col + 1);
        diag.addEdge(s0, t1);
        diag.addEdge(s1, t0);
        qubits[target] = t0;
        qubits[ctrl]   = t1;
    }

    void FunctionalityConstruction::addCcx(ZXDiagram& diag, Qubit ctrl0, Qubit ctrl1, Qubit target,
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

    FunctionalityConstruction::op_it FunctionalityConstruction::parse_op(ZXDiagram& diag, op_it it, op_it end,
                                                                         std::vector<Vertex>& qubits, const qc::Permutation& p) {
        const auto& op = *it;
        if (op->getType() == qc::OpType::Barrier) {
            return it + 1;
        }

        if (!op->isControlled()) {
            const auto target = p.at(op->getTargets().front());
            switch (op->getType()) {
                case qc::OpType::Z:
                    addZSpider(diag, target, qubits,
                               PiExpression(PiRational(1, 1)));
                    break;

                case qc::OpType::RZ:
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
                    addZSpider(diag, target, qubits,
                               PiExpression(PiRational(1, 1)));
                    addXSpider(diag, target, qubits,
                               PiExpression(PiRational(1, 1)));
                    break;

                case qc::OpType::RY:
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
                    const auto target2 = p.at(op->getTargets()[1]);
                    addSwap(diag, target, target2, qubits);
                    break;
                }
                case qc::OpType::iSWAP: {
                    const auto target2 = p.at(op->getTargets()[1]);
                    addZSpider(diag, target, qubits, PiExpression(PiRational(1, 2)));
                    addZSpider(diag, target2, qubits, PiExpression(PiRational(1, 2)));
                    addZSpider(diag, target, qubits, PiExpression(),
                               EdgeType::Hadamard);
                    addCnot(diag, target, target2, qubits);
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
            const auto target = p.at(op->getTargets().front());
            const auto ctrl   = p.at((*op->getControls().begin()).qubit);
            switch (op->getType()) { // TODO: any gate can be controlled
                case qc::OpType::X:
                    // check if swap
                    if (checkSwap(it, end, ctrl, target, p)) {
                        addSwap(diag, ctrl, target, qubits);
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
            Qubit       ctrl0  = 0;
            Qubit       ctrl1  = 0;
            const Qubit target = p.at(op->getTargets().front());
            int         i      = 0;
            for (auto& ctrl: op->getControls()) {
                if (i++ == 0)
                    ctrl0 = p.at(ctrl.qubit);
                else
                    ctrl1 = p.at(ctrl.qubit);
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
                    break;
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
                auto* compOp = dynamic_cast<qc::CompoundOperation*>(op.get());
                for (auto subIt = compOp->cbegin(); subIt != compOp->cend();)
                    subIt = parse_op(diag, subIt, compOp->cend(), qubits, qc->initialLayout);
                ++it;
            } else {
                it = parse_op(diag, it, qc->cend(), qubits, qc->initialLayout);
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

    bool FunctionalityConstruction::transformableToZX(qc::Operation* op) {
        if (op->getType() == qc::OpType::Compound) {
            auto* compOp = dynamic_cast<qc::CompoundOperation*>(op);

            return std::all_of(compOp->cbegin(), compOp->cend(), [&](const auto& op) { return transformableToZX(op.get()); });
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
                                                       std::size_t          i) {
        const auto* symbOp = dynamic_cast<const qc::SymbolicOperation*>(op);
        if (symbOp) {
            return toPiExpr(symbOp->getParameter(i));
        } else {
            return PiExpression{zx::PiRational{op->getParameter()[i]}};
        }
    }
    PiExpression FunctionalityConstruction::toPiExpr(const qc::SymbolOrNumber& param) {
        if (std::holds_alternative<double>(param))
            return zx::PiExpression{
                    zx::PiRational{std::get<double>(param)}};
        else {
            return std::get<qc::Symbolic>(param).convert<zx::PiRational>();
        }
    }
} // namespace zx
