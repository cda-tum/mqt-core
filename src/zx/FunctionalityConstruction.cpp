#include "zx/FunctionalityConstruction.hpp"

#include "Definitions.hpp"
#include "ZXDiagram.hpp"

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

namespace zx {

    bool FunctionalityConstruction::checkSwap(op_it it, op_it end, Qubit ctrl,
                                              Qubit target, const qc::Permutation& p) {
        if (it + 1 != end && it + 2 != end) {
            auto& op1 = *(it + 1);
            auto& op2 = *(it + 2);
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
                                               std::vector<Vertex>& qubit_vertices,
                                               const Expression& phase, EdgeType type) {
        auto new_vertex = diag.addVertex(
                qubit, diag.getVData(qubit_vertices[qubit]).value().col + 1, phase,
                VertexType::Z);

        diag.addEdge(qubit_vertices[qubit], new_vertex, type);
        qubit_vertices[qubit] = new_vertex;
    }

    void FunctionalityConstruction::addXSpider(ZXDiagram& diag, Qubit qubit,
                                               std::vector<Vertex>& qubit_vertices,
                                               const Expression& phase, EdgeType type) {
        auto new_vertex = diag.addVertex(
                qubit, diag.getVData(qubit_vertices[qubit]).value().col + 1, phase,
                VertexType::X);
        diag.addEdge(qubit_vertices[qubit], new_vertex, type);
        qubit_vertices[qubit] = new_vertex;
    }

    void FunctionalityConstruction::addCnot(ZXDiagram& diag, Qubit ctrl, Qubit target,
                                            std::vector<Vertex>& qubit_vertices) {
        addZSpider(diag, ctrl, qubit_vertices);
        addXSpider(diag, target, qubit_vertices);
        diag.addEdge(qubit_vertices[ctrl], qubit_vertices[target]);
    }

    void FunctionalityConstruction::addCphase(ZXDiagram& diag, const PiRational& phase, Qubit ctrl, Qubit target,
                                              std::vector<Vertex>& qubit_vertices) {
        addZSpider(diag, ctrl, qubit_vertices, Expression(phase / 2));
        addCnot(diag, ctrl, target, qubit_vertices);
        addZSpider(diag, target, qubit_vertices, Expression(-phase / 2));
        addCnot(diag, ctrl, target, qubit_vertices);
        addZSpider(diag, target, qubit_vertices, Expression(phase / 2));
    }

    void FunctionalityConstruction::addSwap(ZXDiagram& diag, Qubit ctrl, Qubit target,
                                            std::vector<Vertex>& qubit_vertices) {
        const auto s0 = qubit_vertices[target];
        const auto s1 = qubit_vertices[ctrl];

        const auto t0 = diag.addVertex(target, diag.getVData(qubit_vertices[target]).value().col + 1);
        const auto t1 = diag.addVertex(ctrl, diag.getVData(qubit_vertices[target]).value().col + 1);

        diag.addEdge(s0, t1);
        diag.addEdge(s1, t0);

        qubit_vertices[target] = t0;
        qubit_vertices[ctrl]   = t1;
    }

    void FunctionalityConstruction::addCcx(ZXDiagram& diag, Qubit ctrl_0, Qubit ctrl_1, Qubit target,
                                           std::vector<Vertex>& qubit_vertices) {
        addZSpider(diag, target, qubit_vertices, Expression(), EdgeType::Hadamard);
        addCnot(diag, ctrl_1, target, qubit_vertices);
        addZSpider(diag, target, qubit_vertices, Expression(PiRational(-1, 4)));
        addCnot(diag, ctrl_0, target, qubit_vertices);
        addZSpider(diag, target, qubit_vertices, Expression(PiRational(1, 4)));
        addCnot(diag, ctrl_1, target, qubit_vertices);
        addZSpider(diag, ctrl_1, qubit_vertices, Expression(PiRational(1, 4)));
        addZSpider(diag, target, qubit_vertices, Expression(PiRational(-1, 4)));
        addCnot(diag, ctrl_0, target, qubit_vertices);
        addZSpider(diag, target, qubit_vertices, Expression(PiRational(1, 4)));
        addCnot(diag, ctrl_0, ctrl_1, qubit_vertices);
        addZSpider(diag, ctrl_0, qubit_vertices, Expression(PiRational(1, 4)));
        addZSpider(diag, ctrl_1, qubit_vertices, Expression(PiRational(-1, 4)));
        addZSpider(diag, target, qubit_vertices, Expression(PiRational(0, 1)),
                   EdgeType::Hadamard);
        addCnot(diag, ctrl_0, ctrl_1, qubit_vertices);
    }

    FunctionalityConstruction::op_it FunctionalityConstruction::parse_op(ZXDiagram& diag, op_it it, op_it end,
                                                                         std::vector<Vertex>& qubit_vertices, const qc::Permutation& p) {
        auto& op = *it;
        std::cout << op->getName() << ": ";
        if (op->getType() == qc::OpType::Barrier) {
            return it + 1;
        }

        if (!op->isControlled()) {
            const auto target = p.at(op->getTargets().front());
            std::cout << "Target " << static_cast<int>(target);
            std::cout << std::endl;
            switch (op->getType()) {
                case qc::OpType::Z: {
                    addZSpider(diag, target, qubit_vertices,
                               Expression(PiRational(1, 1)));
                    break;
                }

                case qc::OpType::RZ:
                case qc::OpType::Phase: {
                    addZSpider(diag, target, qubit_vertices, Expression(PiRational(op->getParameter().front())));
                    break;
                }
                case qc::OpType::X: {
                    addXSpider(diag, target, qubit_vertices, Expression(PiRational(1, 1)));
                    break;
                }

                case qc::OpType::RX: {
                    addXSpider(diag, target, qubit_vertices, Expression(PiRational(op->getParameter().front())));
                    break;
                }

                case qc::OpType::Y: {
                    addZSpider(diag, target, qubit_vertices, Expression(PiRational(1, 1)));
                    addXSpider(diag, target, qubit_vertices, Expression(PiRational(1, 1)));
                    break;
                }

                case qc::OpType::RY: {
                    addXSpider(diag, target, qubit_vertices, Expression(PiRational(1, 2)));
                    addZSpider(diag, target, qubit_vertices,
                               Expression(PiRational(op->getParameter()[0])) + PiRational(1, 1));
                    addXSpider(diag, target, qubit_vertices, Expression(PiRational(1, 2)));
                    addZSpider(diag, target, qubit_vertices, Expression(PiRational(3, 1)));
                    break;
                }
                case qc::OpType::T: {
                    addZSpider(diag, target, qubit_vertices, Expression(PiRational(1, 4)));
                    break;
                }
                case qc::OpType::Tdag: {
                    addZSpider(diag, target, qubit_vertices, Expression(PiRational(-1, 4)));
                    break;
                }
                case qc::OpType::S: {
                    addZSpider(diag, target, qubit_vertices, Expression(PiRational(1, 2)));
                    break;
                }
                case qc::OpType::Sdag: {
                    addZSpider(diag, target, qubit_vertices, Expression(PiRational(-1, 2)));
                    break;
                }
                case qc::OpType::U2: {
                    addZSpider(diag, target, qubit_vertices,
                               Expression(PiRational(op->getParameter()[0])) - PiRational(1, 2));
                    addXSpider(diag, target, qubit_vertices, Expression(PiRational(1, 2)));
                    addZSpider(diag, target, qubit_vertices,
                               Expression(PiRational(op->getParameter()[1])) + PiRational(1, 2));
                    break;
                }
                case qc::OpType::U3: {
                    addZSpider(diag, target, qubit_vertices, Expression(PiRational(op->getParameter().front())));
                    addXSpider(diag, target, qubit_vertices, Expression(PiRational(1, 2)));
                    addZSpider(diag, target, qubit_vertices,
                               Expression(PiRational(op->getParameter()[2])) + PiRational(1, 1));
                    addXSpider(diag, target, qubit_vertices, Expression(PiRational(1, 2)));
                    addZSpider(diag, target, qubit_vertices,
                               Expression(PiRational(op->getParameter()[1])) + PiRational(3, 1));
                    break;
                }

                case qc::OpType::SWAP: {
                    const auto target2 = p.at(op->getTargets()[1]);
                    addSwap(diag, target, target2, qubit_vertices);
                    break;
                }
                case qc::OpType::H: {
                    addZSpider(diag, target, qubit_vertices, Expression(),
                               EdgeType::Hadamard);
                    break;
                }
                case qc::OpType::Measure:
                case qc::OpType::I: {
                    break;
                }
                default: {
                    throw ZXException("Unsupported Operation: " +
                                      qc::toString(op->getType()));
                }
            }
        } else if (op->getNcontrols() == 1 && op->getNtargets() == 1) {
            const auto target = p.at(op->getTargets().front());
            const auto ctrl   = p.at((*op->getControls().begin()).qubit);
            std::cout << "Target " << static_cast<int>(target);
            std::cout << "Ctrl " << static_cast<int>(ctrl);
            std::cout << std::endl;
            switch (op->getType()) { // TODO: any gate can be controlled
                case qc::OpType::X: {
                    // check if swap
                    if (checkSwap(it, end, ctrl, target, p)) {
                        addSwap(diag, ctrl, target, qubit_vertices);
                        return it + 3;
                    } else {
                        addCnot(diag, ctrl, target, qubit_vertices);
                    }

                    break;
                }
                case qc::OpType::Z: {
                    addZSpider(diag, target, qubit_vertices, Expression(),
                               EdgeType::Hadamard);
                    addCnot(diag, ctrl, target, qubit_vertices);
                    addZSpider(diag, target, qubit_vertices, Expression(),
                               EdgeType::Hadamard);

                    break;
                }

                case qc::OpType::I: {
                    break;
                }

                case qc::OpType::Phase: {
                    const auto phase = PiRational(op->getParameter().front());
                    addCphase(diag, phase, ctrl, target, qubit_vertices);
                    break;
                }

                case qc::OpType::T: {
                    addCphase(diag, PiRational(1, 4), ctrl, target, qubit_vertices);
                    break;
                }

                case qc::OpType::S: {
                    addCphase(diag, PiRational(1, 2), ctrl, target, qubit_vertices);
                    break;
                }

                case qc::OpType::Tdag: {
                    addCphase(diag, PiRational(-1, 4), ctrl, target, qubit_vertices);
                    break;
                }

                case qc::OpType::Sdag: {
                    addCphase(diag, PiRational(-1, 2), ctrl, target, qubit_vertices);
                    break;
                }

                default: {
                    throw ZXException("Unsupported Controlled Operation: " +
                                      qc::toString(op->getType()));
                }
            }
        } else if (op->getNcontrols() == 2) {
            Qubit       ctrl_0 = 0;
            Qubit       ctrl_1 = 0;
            const Qubit target = p.at(op->getTargets().front());
            int         i      = 0;
            for (auto& ctrl: op->getControls()) {
                if (i++ == 0)
                    ctrl_0 = p.at(ctrl.qubit);
                else
                    ctrl_1 = p.at(ctrl.qubit);
            }
            std::cout << "Target " << static_cast<int>(target);
            std::cout << "Ctrl1 " << static_cast<int>(ctrl_0);
            std::cout << "Ctrl2 " << static_cast<int>(ctrl_1);
            std::cout << std::endl;
            switch (op->getType()) {
                case qc::OpType::X: {
                    addCcx(diag, ctrl_0, ctrl_1, target, qubit_vertices);
                    break;
                }

                case qc::OpType::Z: {
                    addZSpider(diag, target, qubit_vertices, Expression(),
                               EdgeType::Hadamard);
                    addCcx(diag, ctrl_0, ctrl_1, target, qubit_vertices);
                    addZSpider(diag, target, qubit_vertices, Expression(),
                               EdgeType::Hadamard);
                    break;
                }
                default: {
                    throw ZXException("Unsupported Multi-control operation: " +
                                      qc::toString(op->getType()));
                    break;
                }
            }
        } else {
            throw ZXException("Unsupported Multi-control operation (" + std::to_string(op->getNcontrols()) + " ctrls)" + qc::toString(op->getType()));
        }
        return it + 1;
    }

    ZXDiagram FunctionalityConstruction::buildFunctionality(const qc::QuantumComputation* qc) {
        for (auto [v, w]: qc->initialLayout)
            std::cout << static_cast<int>(v) << " - " << static_cast<int>(w) << std::endl;
        ZXDiagram           diag(qc->getNqubits());
        std::vector<Vertex> qubit_vertices(qc->getNqubits());
        for (size_t i = 0; i < qc->getNqubits(); ++i) {
            diag.removeEdge(i, i + qc->getNqubits());
            qubit_vertices[i] = i;
        }

        auto initial_layout = qc->initialLayout;

        // if (!initial_layout.empty()) {
        //     std::vector<Vertex> new_qubit_vertices(qc->getNqubits());
        //     for (auto i = 0; i < qc->getNqubits(); i++) {
        //         new_qubit_vertices[i] = qubit_vertices[i];
        //     }
        //     for (auto& [tar, src]:
        //          qc->initialLayout) { // reverse initial permutation
        //         if (tar == src)
        //             continue;

        //         auto v_tar = diag.addVertex(tar, 1);
        //         diag.addEdge(qubit_vertices[src], v_tar);
        //         new_qubit_vertices[tar] = v_tar;
        //     }
        //     qubit_vertices = new_qubit_vertices;
        // }

        for (auto it = qc->cbegin(); it != qc->cend();) {
            auto& op = *it;

            if (op->getType() == qc::OpType::Compound) {
                auto* compOp = dynamic_cast<qc::CompoundOperation*>(op.get());
                for (auto subIt = compOp->cbegin(); subIt != compOp->cend();)
                    subIt = parse_op(diag, subIt, compOp->end(), qubit_vertices, qc->initialLayout);
                ++it;
            } else {
                it = parse_op(diag, it, qc->end(), qubit_vertices, qc->initialLayout);
            }
        }

        for (size_t i = 0; i < qubit_vertices.size(); ++i) {
            diag.addEdge(qubit_vertices[i], diag.getOutputs()[i]);
        }
        return diag;
    }
    bool FunctionalityConstruction::transformableToZX(const qc::QuantumComputation* qc) {
        for (const auto& it: *qc) {
            if (!transformableToZX(it.get()))
                return false;
        }
        return true;
    }

    bool FunctionalityConstruction::transformableToZX(qc::Operation* op) {
        if (op->getType() == qc::OpType::Compound) {
            auto* compOp = dynamic_cast<qc::CompoundOperation*>(op);

            for (const auto& it: *compOp) {
                if (!transformableToZX(it.get()))
                    return false;
            }
            return true;
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
                case qc::OpType::H:
                case qc::OpType::Measure:
                case qc::OpType::I: {
                    return true;
                }
                default: {
                    return false;
                }
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
                case qc::OpType::Sdag: {
                    return true;
                }

                default: {
                    return false;
                }
            }
        } else if (op->getNcontrols() == 2) {
            switch (op->getType()) {
                case qc::OpType::X:
                case qc::OpType::Z: {
                    return true;
                }
                default: {
                    return false;
                }
            }
        } else {
            return false;
        }
        return false;
    }

} // namespace zx
