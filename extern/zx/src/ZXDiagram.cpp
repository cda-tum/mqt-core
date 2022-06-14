#include "ZXDiagram.hpp"

#include "Definitions.hpp"
#include "Rational.hpp"
#include "Utils.hpp"
// #include "dd/Definitions.hpp"
// #include "operations/CompoundOperation.hpp"
// #include "operations/OpType.hpp"
#include <algorithm>
#include <unordered_map>
#include <utility>
#include <vector>

namespace zx {

    ZXDiagram::ZXDiagram(int32_t nqubits) {
        auto qubit_vertices = initGraph(nqubits);
        closeGraph(qubit_vertices);
    }

    // ZXDiagram::ZXDiagram(std::string filename)
    //     : ZXDiagram(qc::QuantumComputation(filename)) {}

    // using op_it =
    //     decltype(std::begin(std::vector<std::unique_ptr<qc::Operation>>()));
    // static bool check_swap(op_it it, op_it end, Qubit ctrl, Qubit target) {
    //   if (it + 1 != end && it + 2 != end) {
    //     auto &op1 = *(it + 1);
    //     auto &op2 = *(it + 2);
    //     if (op1->getType() == qc::OpType::X && op2->getType() == qc::OpType::X &&
    //         op1->getNcontrols() == 1 && op2->getNcontrols() == 1) {
    //       auto tar1 = op1->getTargets()[0];
    //       auto tar2 = op2->getTargets()[0];
    //       auto ctrl1 = (*op1->getControls().begin()).qubit;
    //       auto ctrl2 = (*op2->getControls().begin()).qubit;
    //       return ctrl == tar1 && tar1 == ctrl2 && target == ctrl1 && ctrl1 == tar2;
    //     }
    //   }
    //   return false;
    // }

    // ZXDiagram::ZXDiagram(const qc::QuantumComputation &qc) {
    //   auto qubit_vertices = initGraph(qc.getNqubits());

    //   initial_layout = qc.initialLayout;
    //   output_permutation = qc.outputPermutation;

    //   if (initial_layout.value().size() != 0) {
    //     std::vector<Vertex> new_qubit_vertices(getNQubits());
    //     for (auto i = 0; i < getNQubits(); i++) {
    //       new_qubit_vertices[i] = qubit_vertices[i];
    //     }
    //     for (auto &[tar, src] : qc.initialLayout) { // reverse initial permutation
    //       if (tar == src)
    //         continue;
    //       auto v_tar = addVertex(tar, 1);
    //       addEdge(qubit_vertices[src], v_tar);
    //       new_qubit_vertices[tar] = v_tar;
    //     }
    //     qubit_vertices = new_qubit_vertices;
    //   }

    //   for (auto it = qc.begin(); it != qc.end();) {
    //     auto &op = *it;

    //     if (op->getType() == qc::OpType::Compound) {
    //       qc::CompoundOperation *comp_op =
    //           static_cast<qc::CompoundOperation *>(op.get());
    //       for (op_it sub_it = comp_op->begin(); sub_it != comp_op->end();)
    //         sub_it = parse_op(sub_it, comp_op->end(), qubit_vertices);
    //       ++it;
    //     } else {
    //       it = parse_op(it, qc.end(), qubit_vertices);
    //     }
    //   }

    //   //   reverse_permutation(qc.outputPermutation, qubit_vertices);
    //   // std::cout << "" << "\n";

    //   closeGraph(qubit_vertices);
    // }

    void ZXDiagram::addEdge(Vertex from, Vertex to, EdgeType type) {
        edges[from].push_back({to, type});
        edges[to].push_back({from, type});
        nedges++;
    }

    void ZXDiagram::addEdgeParallelAware(Vertex from, Vertex to,
                                         EdgeType etype) { // TODO: Scalars
        if (from == to) {
            if (type(from) != VertexType::Boundary && etype == EdgeType::Hadamard) {
                addPhase(from, Expression(PiRational(1, 1)));
            }
            return;
        }

        auto edge_it = getEdgePtr(from, to);

        if (edge_it == edges[from].end()) {
            addEdge(from, to, etype);
            return;
        }

        if (type(from) == VertexType::Boundary || type(to) == VertexType::Boundary)
            return;

        if (type(from) == type(to)) {
            if (edge_it->type == EdgeType::Hadamard && etype == EdgeType::Hadamard) {
                edges[from].erase(edge_it);
                removeHalfEdge(to, from);
                nedges--;
            } else if (edge_it->type == EdgeType::Hadamard &&
                       etype == EdgeType::Simple) {
                edge_it->type = EdgeType::Simple;
                getEdgePtr(to, from)->toggle();
                addPhase(from, Expression(PiRational(1, 1)));
            } else if (edge_it->type == EdgeType::Simple &&
                       etype == EdgeType::Hadamard) {
                addPhase(from, Expression(PiRational(1, 1)));
            }
        } else {
            if (edge_it->type == EdgeType::Simple && etype == EdgeType::Simple) {
                edges[from].erase(edge_it);
                removeHalfEdge(to, from);
                nedges--;
            } else if (edge_it->type == EdgeType::Hadamard &&
                       etype == EdgeType::Simple) {
                addPhase(from, Expression(PiRational(1, 1)));
            } else if (edge_it->type == EdgeType::Simple &&
                       etype == EdgeType::Hadamard) {
                edge_it->type = EdgeType::Hadamard;
                getEdgePtr(to, from)->toggle();
                addPhase(from, Expression(PiRational(1, 1)));
            }
        }
    }

    void ZXDiagram::remove_edge(Vertex from, Vertex to) {
        removeHalfEdge(from, to);
        removeHalfEdge(to, from);
        nedges--;
    }

    void ZXDiagram::removeHalfEdge(Vertex from, Vertex to) {
        auto& incident = edges[from];
        incident.erase(std::remove_if(incident.begin(), incident.end(),
                                      [&](auto& edge) { return edge.to == to; }),
                       incident.end());
    }

    Vertex ZXDiagram::addVertex(const VertexData& data) {
        nvertices++;
        Vertex v = 0;
        if (!deleted.empty()) {
            v = deleted.back();
            deleted.pop_back();
            vertices[v] = data;
            edges[v].clear();
            return v;
        } else {
            v = nvertices;
            vertices.push_back(data);
            edges.emplace_back();
        }
        return nvertices - 1;
    }

    Vertex ZXDiagram::addVertex(Qubit qubit, Col col, const Expression& phase,
                                VertexType type) {
        return addVertex({col, qubit, phase, type});
    }

    void ZXDiagram::removeVertex(Vertex to_remove) {
        deleted.push_back(to_remove);
        vertices[to_remove].reset();
        nvertices--;

        for (auto& [to, _]: incidentEdges(to_remove)) {
            removeHalfEdge(to, to_remove);
            nedges--;
        }
    }

    [[nodiscard]] bool ZXDiagram::connected(Vertex from, Vertex to) const {
        if (isDeleted(from) || isDeleted(to))
            return false;

        auto& incident = edges[from];
        auto  edge     = std::find_if(incident.begin(), incident.end(),
                                      [&](auto& edge) { return edge.to == to; });
        return edge != incident.end();
    }

    [[nodiscard]] std::optional<Edge> ZXDiagram::getEdge(Vertex from,
                                                         Vertex to) const {
        std::optional<Edge> ret;
        auto&               incident = edges[from];
        auto                edge     = std::find_if(incident.begin(), incident.end(),
                                                    [&](auto& edge) { return edge.to == to; });
        if (edge != incident.end())
            ret = *edge;
        return ret;
    }

    std::vector<Edge>::iterator ZXDiagram::getEdgePtr(Vertex from, Vertex to) {
        auto& incident = edges[from];
        auto  edge     = std::find_if(incident.begin(), incident.end(),
                                      [&](auto& edge) { return edge.to == to; });
        return edge;
    }

    [[nodiscard]] std::vector<std::pair<Vertex, VertexData&>>
    ZXDiagram::getVertices() {
        Vertices verts(vertices);
        return std::vector<std::pair<Vertex, VertexData&>>(verts.begin(),
                                                           verts.end());
    }

    [[nodiscard]] std::vector<std::pair<Vertex, Vertex>> ZXDiagram::getEdges() {
        Edges es(edges, vertices);
        return std::vector<std::pair<Vertex, Vertex>>(es.begin(), es.end());
    }

    bool ZXDiagram::isInput(Vertex v) const {
        return std::find(inputs.begin(), inputs.end(), v) != inputs.end();
    }
    bool ZXDiagram::isOutput(Vertex v) const {
        return std::find(outputs.begin(), outputs.end(), v) != outputs.end();
    }

    void ZXDiagram::toGraphlike() {
        for (Vertex v = 0; (size_t)v < vertices.size(); v++) {
            if (!vertices[v].has_value())
                continue;
            if (vertices[v].value().type == VertexType::X) {
                for (auto& edge: edges[v]) {
                    edge.toggle();
                    getEdgePtr(edge.to, v)
                            ->toggle(); // toggle corresponding edge in other direction
                }

                vertices[v].value().type = VertexType::Z;
            }
        }
    }

    [[nodiscard]] ZXDiagram ZXDiagram::adjoint() const {
        ZXDiagram copy = *this;
        copy.invert();
        return copy;
    }

    ZXDiagram& ZXDiagram::invert() {
        auto h  = inputs;
        inputs  = outputs;
        outputs = h;

        for (auto& data: vertices) {
            if (data.has_value()) {
                data.value().phase = -data.value().phase;
            }
        }
        return *this;
    }

    ZXDiagram& ZXDiagram::concat(const ZXDiagram& rhs) {
        if (rhs.getNQubits() != this->getNQubits())
            throw ZXException(
                    "Cannot concatenate Diagrams with differing number of qubits!");

        std::unordered_map<Vertex, Vertex> new_vs;
        for (size_t i = 0; i < rhs.vertices.size(); i++) {
            if (!rhs.vertices[i].has_value() || rhs.isInput(i))
                continue;

            auto new_v = addVertex(rhs.vertices[i].value());
            new_vs[i]  = new_v;
        }

        for (size_t i = 0; i < rhs.vertices.size(); i++) { // add new edges
            if (!rhs.vertices[i].has_value() || rhs.isInput(i))
                continue;

            for (auto& [to, type]: rhs.edges[i]) {
                if (!rhs.isInput(to)) {
                    if (i < (size_t)to) { // make sure not to add edge twice
                        addEdge(new_vs[i], new_vs[to], type);
                    }
                } else {
                    auto out_v = outputs[rhs.qubit(to)];
                    for (auto [interior_v, interior_type]:
                         edges[out_v]) { // redirect edges going to outputs
                        // removeHalfEdge(interior_v, out_v);
                        // nedges--;
                        if (interior_type == type) {
                            addEdge(interior_v, new_vs[i], EdgeType::Simple);
                        } else {
                            addEdge(interior_v, new_vs[i], EdgeType::Hadamard);
                        }
                    }
                }
            }
        } // add new edges

        for (size_t i = 0; i < outputs.size(); i++) {
            removeVertex(outputs[i]);
            outputs[i] = new_vs[rhs.outputs[i]];
        }

        return *this;
    }

    bool ZXDiagram::isIdentity() const {
        if ((size_t)nedges != inputs.size())
            return false;

        for (size_t i = 0; i < inputs.size(); i++) {
            if (!connected(inputs[i], outputs[i]))
                return false;
        }
        return true;
    }

    // bool ZXDiagram::isIdentity(const qc::Permutation &perm) const {
    //   if (nedges != inputs.size())
    //     return false;

    //   for (auto &[in, out] : perm) {
    //     if (!connected(inputs[in], outputs[out]))
    //       return false;
    //   }
    //   return true;
    // }

    void ZXDiagram::addZSpider(Qubit                qubit,
                               std::vector<Vertex>& qubit_vertices,
                               const Expression& phase, EdgeType type) {
        auto new_vertex = addVertex(
                {vertices[qubit].value().col + 1, qubit, phase, VertexType::Z});

        addEdge(qubit_vertices[qubit], new_vertex, type);
        qubit_vertices[qubit] = new_vertex;
    }

    void ZXDiagram::addXSpider(Qubit                qubit,
                               std::vector<Vertex>& qubit_vertices,
                               const Expression& phase, EdgeType type) {
        auto new_vertex = addVertex(
                {vertices[qubit].value().col + 1, qubit, phase, VertexType::X});
        addEdge(qubit_vertices[qubit], new_vertex, type);
        qubit_vertices[qubit] = new_vertex;
    }

    void ZXDiagram::addCnot(Qubit ctrl, Qubit target,
                            std::vector<Vertex>& qubit_vertices) {
        addZSpider(ctrl, qubit_vertices);
        addXSpider(target, qubit_vertices);
        addEdge(qubit_vertices[ctrl], qubit_vertices[target]);
    }

    void ZXDiagram::addCphase(PiRational phase, Qubit ctrl, Qubit target,
                              std::vector<Vertex>& qubit_vertices) {
        addZSpider(ctrl, qubit_vertices, Expression(phase / 2));
        addCnot(ctrl, target, qubit_vertices);
        addZSpider(target, qubit_vertices, Expression(-phase / 2));
        addCnot(ctrl, target, qubit_vertices);
        addZSpider(target, qubit_vertices, Expression(phase / 2));
    }

    void ZXDiagram::addSwap(Qubit ctrl, Qubit target,
                            std::vector<Vertex>& qubit_vertices) {
        auto s0 = qubit_vertices[target];
        auto s1 = qubit_vertices[ctrl];

        auto t0 = addVertex(target, vertices[target].value().col + 1);
        auto t1 = addVertex(ctrl, vertices[ctrl].value().col + 1);

        addEdge(s0, t1);
        addEdge(s1, t0);

        qubit_vertices[target] = t0;
        qubit_vertices[ctrl]   = t1;
    }

    void ZXDiagram::addCcx(Qubit ctrl_0, Qubit ctrl_1, Qubit target,
                           std::vector<Vertex>& qubit_vertices) {
        addZSpider(target, qubit_vertices, Expression(), EdgeType::Hadamard);
        addCnot(ctrl_1, target, qubit_vertices);
        addZSpider(target, qubit_vertices, Expression(PiRational(-1, 4)));
        addCnot(ctrl_0, target, qubit_vertices);
        addZSpider(target, qubit_vertices, Expression(PiRational(1, 4)));
        addCnot(ctrl_1, target, qubit_vertices);
        addZSpider(ctrl_1, qubit_vertices, Expression(PiRational(1, 4)));
        addZSpider(target, qubit_vertices, Expression(PiRational(-1, 4)));
        addCnot(ctrl_0, target, qubit_vertices);
        addZSpider(target, qubit_vertices, Expression(PiRational(1, 4)));
        addCnot(ctrl_0, ctrl_1, qubit_vertices);
        addZSpider(ctrl_0, qubit_vertices, Expression(PiRational(1, 4)));
        addZSpider(ctrl_1, qubit_vertices, Expression(PiRational(-1, 4)));
        addZSpider(target, qubit_vertices, Expression(PiRational(0, 1)),
                   EdgeType::Hadamard);
        addCnot(ctrl_0, ctrl_1, qubit_vertices);
    }

    std::vector<Vertex> ZXDiagram::initGraph(int nqubits) {
        std::vector<Vertex> qubit_vertices(nqubits);

        for (size_t i = 0; i < qubit_vertices.size(); i++) {
            auto v = addVertex(
                    {1, static_cast<Qubit>(i), Expression(), VertexType::Boundary});
            qubit_vertices[i] = v;
            inputs.push_back(v);
        }

        return qubit_vertices;
    }

    void ZXDiagram::closeGraph(std::vector<Vertex>& qubit_vertices) {
        for (Vertex v: qubit_vertices) {
            VertexData v_data = vertices[v].value();
            Vertex     new_v  = addVertex(
                         {v_data.col + 1, v_data.qubit, Expression(), VertexType::Boundary});
            addEdge(v, new_v);
            outputs.push_back(new_v);
        }
    }

    void ZXDiagram::makeAncilla(Qubit qubit) {
        auto in_v  = inputs[qubit];
        auto out_v = outputs[qubit];
        // auto new_in = addVertex(qubit, 0, PiRational(0,1), VertexType::Boundary);
        // auto new_out = addVertex(qubit, 0, PiRational(0,1), VertexType::Boundary);
        // inputs[qubit] = new_in;
        // outputs[qubit] = new_out;
        inputs.erase(inputs.begin() + qubit);
        outputs.erase(outputs.begin() + qubit);

        setType(in_v, VertexType::X);
        setType(in_v, VertexType::X);
        removeVertex(in_v);
        removeVertex(out_v);
    }

    // op_it ZXDiagram::parse_op(op_it it, op_it end,
    //                           std::vector<Vertex> &qubit_vertices) {
    //   auto &op = *it;

    //   if (op->getType() == qc::OpType::Barrier) {
    //     return it + 1;
    //   }

    //   if (op->getNcontrols() == 0 && op->getNtargets() == 1) {
    //     auto target = op->getTargets()[0];
    //     switch (op->getType()) {
    //     case qc::OpType::Z: {
    //       addZSpider(target, qubit_vertices, Expression(PiRational(1, 1)));
    //       break;
    //     }

    //     case qc::OpType::RZ:
    //     case qc::OpType::Phase: {

    //       addZSpider(target, qubit_vertices, Expression(PiRational(op->getParameter()[0])));
    //       break;
    //     }
    //     case qc::OpType::X: {
    //       addXSpider(target, qubit_vertices, Expression(PiRational(1, 1)));
    //       break;
    //     }

    //     case qc::OpType::RX: {
    //       addXSpider(target, qubit_vertices, Expression(PiRational(op->getParameter()[0])));
    //       break;
    //     }

    //     case qc::OpType::Y: {
    //       addZSpider(target, qubit_vertices, Expression(PiRational(1, 1)));
    //       addXSpider(target, qubit_vertices, Expression(PiRational(1, 1)));
    //       break;
    //     }

    //     case qc::OpType::RY: {
    //       // addZSpider(target, qubit_vertices,
    //       // PiRational(op->getParameter()[2]));
    //       addXSpider(target, qubit_vertices, Expression(PiRational(1, 2)));
    //       addZSpider(target, qubit_vertices,
    //                    Expression(PiRational(op->getParameter()[0])) + PiRational(1, 1));
    //       addXSpider(target, qubit_vertices, Expression(PiRational(1, 2)));
    //       addZSpider(target, qubit_vertices, Expression(PiRational(3, 1)));
    //       break;
    //     }
    //     case qc::OpType::T: {
    //       addZSpider(target, qubit_vertices, Expression(PiRational(1, 4)));
    //       break;
    //     }
    //     case qc::OpType::Tdag: {
    //       addZSpider(target, qubit_vertices, Expression(PiRational(-1, 4)));
    //       break;
    //     }
    //     case qc::OpType::S: {
    //       addZSpider(target, qubit_vertices, Expression(PiRational(1, 2)));
    //       break;
    //     }
    //     case qc::OpType::Sdag: {
    //       addZSpider(target, qubit_vertices, Expression(PiRational(-1, 2)));
    //       break;
    //     }
    //     case qc::OpType::U2: {
    //       addZSpider(target, qubit_vertices,
    //                    Expression(PiRational(op->getParameter()[0])) - PiRational(1, 2));
    //       addXSpider(target, qubit_vertices, Expression(PiRational(1, 2)));
    //       addZSpider(target, qubit_vertices,
    //                    Expression(PiRational(op->getParameter()[1])) + PiRational(1, 2));
    //       break;
    //     }
    //     case qc::OpType::U3: {
    //       addZSpider(target, qubit_vertices, Expression(PiRational(op->getParameter()[0])));
    //       addXSpider(target, qubit_vertices, Expression(PiRational(1, 2)));
    //       addZSpider(target, qubit_vertices,
    //                    Expression(PiRational(op->getParameter()[2])) + PiRational(1, 1));
    //       addXSpider(target, qubit_vertices, Expression(PiRational(1, 2)));
    //       addZSpider(target, qubit_vertices,
    //                    Expression(PiRational(op->getParameter()[1])) + PiRational(3, 1));
    //       break;
    //     }

    //     case qc::OpType::SWAP: {
    //       auto target2 = op->getTargets()[0];
    //       addSwap(target, target2, qubit_vertices);

    //       // addCnot(target, target2, qubit_vertices);
    //       // addCnot(target2, target, qubit_vertices);
    //       // addCnot(target, target2, qubit_vertices);
    //       break;
    //     }
    //     case qc::OpType::H: {
    //       addZSpider(target, qubit_vertices, Expression(),
    //                    EdgeType::Hadamard);
    //       break;
    //     }
    //     case qc::OpType::Measure:
    //     case qc::OpType::I: {
    //       break;
    //     }
    //     default: {
    //       throw ZXException("Unsupported Operation: " +
    //                         qc::toString(op->getType()));
    //     }
    //     }
    //   } else if (op->getNcontrols() == 1 && op->getNtargets() == 1) {
    //     auto target = op->getTargets()[0];
    //     auto ctrl = (*op->getControls().begin()).qubit;
    //     switch (op->getType()) { // TODO: any gate can be controlled
    //     case qc::OpType::X: {
    //       // check if swap
    //       if (check_swap(it, end, ctrl, target)) {
    //         addSwap(ctrl, target, qubit_vertices);
    //         return it + 3;
    //         // std::cout << "SWAP" << "\n";
    //       } else {
    //         addCnot(ctrl, target, qubit_vertices);
    //       }

    //       break;
    //     }
    //     case qc::OpType::Z: {
    //       auto phase = PiRational(1, 1);
    //       addZSpider(target, qubit_vertices, Expression(phase / 2));
    //       addCnot(ctrl, target, qubit_vertices);
    //       addZSpider(target, qubit_vertices, Expression(-phase / 2));
    //       addCnot(ctrl, target, qubit_vertices);
    //       break;
    //     }

    //     case qc::OpType::I: {
    //       break;
    //     }

    //     case qc::OpType::Phase: {
    //       auto phase = PiRational(op->getParameter()[0]);
    //       addCphase(phase, ctrl, target, qubit_vertices);
    //       break;
    //     }

    //     case qc::OpType::T: {
    //       addCphase(PiRational(1, 4), ctrl, target, qubit_vertices);
    //       break;
    //     }

    //     case qc::OpType::S: {
    //       addCphase(PiRational(1, 2), ctrl, target, qubit_vertices);
    //       break;
    //     }

    //     case qc::OpType::Tdag: {
    //       addCphase(PiRational(-1, 4), ctrl, target, qubit_vertices);
    //       break;
    //     }

    //     case qc::OpType::Sdag: {
    //       addCphase(PiRational(-1, 2), ctrl, target, qubit_vertices);
    //       break;
    //     }

    //     default: {
    //       throw ZXException("Unsupported Controlled Operation: " +
    //                         qc::toString(op->getType()));
    //     }
    //     }
    //   } else if (op->getNcontrols() == 2) {
    //     Qubit ctrl_0 = 0;
    //     Qubit ctrl_1 = 0;
    //     Qubit target = op->getTargets()[0];
    //     int i = 0;
    //     for (auto &ctrl : op->getControls()) {
    //       if (i++ == 0)
    //         ctrl_0 = ctrl.qubit;
    //       else
    //         ctrl_1 = ctrl.qubit;
    //     }
    //     switch (op->getType()) {
    //     case qc::OpType::X: {
    //       addCcx(ctrl_0, ctrl_1, target, qubit_vertices);
    //       break;
    //     }

    //     case qc::OpType::Z: {
    //       addZSpider(target, qubit_vertices, Expression(),
    //                    EdgeType::Hadamard);
    //       addCcx(ctrl_0, ctrl_1, target, qubit_vertices);
    //       addZSpider(target, qubit_vertices, Expression(),
    //                    EdgeType::Hadamard);
    //       break;
    //     }
    //     default: {
    //       throw ZXException("Unsupported Multi-control operation: " +
    //                         qc::toString(op->getType()));
    //       break;
    //     }
    //     }
    //   } else {
    //     throw ZXException("Unsupported Multi-control operation (>2 ctrls)" +
    //                       qc::toString(op->getType()));
    //   }
    //   return it + 1;
    // }
} // namespace zx
