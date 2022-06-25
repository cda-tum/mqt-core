#include "ZXDiagram.hpp"

#include "Definitions.hpp"
#include "Expression.hpp"
#include "Rational.hpp"
#include "Utils.hpp"

#include <algorithm>
#include <cstddef>
#include <unordered_map>
#include <utility>
#include <vector>

namespace zx {

    ZXDiagram::ZXDiagram(std::size_t nqubits) {
        auto qubit_vertices = initGraph(nqubits);
        closeGraph(qubit_vertices);
    }

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

    void ZXDiagram::removeEdge(Vertex from, Vertex to) {
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
            vertices.emplace_back(data);
            edges.emplace_back();
        }
        return nvertices - 1;
    }

    Vertex ZXDiagram::addVertex(Qubit qubit, Col col, const Expression& phase,
                                VertexType type) {
        return addVertex({col, qubit, phase, type});
    }

    void ZXDiagram::addQubit() {
        auto in  = addVertex(static_cast<zx::Qubit>(getNQubits()) + 1, 0, Expression(), VertexType::Boundary);
        auto out = addVertex(static_cast<zx::Qubit>(getNQubits()) + 1, 0, Expression(), VertexType::Boundary);
        inputs.emplace_back(in);
        outputs.emplace_back(out);
    }
    void ZXDiagram::addQubits(zx::Qubit n) {
        for (zx::Qubit i = 0; i < n; ++i) {
            addQubit();
        }
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
        for (std::size_t i = 0; i < rhs.vertices.size(); i++) {
            if (!rhs.vertices[i].has_value() || rhs.isInput(i))
                continue;

            auto new_v = addVertex(rhs.vertices[i].value());
            new_vs[i]  = new_v;
        }

        for (std::size_t i = 0; i < rhs.vertices.size(); i++) { // add new edges
            if (!rhs.vertices[i].has_value() || rhs.isInput(i))
                continue;

            for (auto& [to, type]: rhs.edges[i]) {
                if (!rhs.isInput(to)) {
                    if (i < to) { // make sure not to add edge twice
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

    std::vector<Vertex> ZXDiagram::initGraph(std::size_t nqubits) {
        std::vector<Vertex> qubit_vertices(nqubits, 0);

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
        inputs.erase(inputs.begin() + qubit);
        outputs.erase(outputs.begin() + qubit);

        setType(in_v, VertexType::X);
        setType(in_v, VertexType::X);
        removeVertex(in_v);
        removeVertex(out_v);
    }

    void ZXDiagram::approximateCliffords(fp tolerance) {
        for (auto& v: vertices) {
            if (v.has_value()) {
                v.value().phase.roundToClifford(tolerance);
            }
        }
    }
} // namespace zx
