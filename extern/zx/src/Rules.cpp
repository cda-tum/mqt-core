#include "Rules.hpp"

#include "Definitions.hpp"
#include "Expression.hpp"
#include "Rational.hpp"
#include "Utils.hpp"
#include "ZXDiagram.hpp"

#include <algorithm>
#include <cstddef>
#include <optional>

namespace zx {
    bool checkIdSimp(ZXDiagram& diag, Vertex v) {
        return diag.degree(v) == 2 && diag.phase(v).isZero() &&
               !diag.isBoundaryVertex(v);
    }

    void removeId(ZXDiagram& diag, Vertex v) {
        auto   edges = diag.incidentEdges(v);
        Vertex v0    = edges[0].to;
        Vertex v1    = edges[1].to;

        EdgeType type = EdgeType::Simple;
        if (edges[0].type != edges[1].type) {
            type = EdgeType::Hadamard;
        }
        //  diag.addEdge(v0, v1,type);
        diag.addEdgeParallelAware(v0, v1, type);
        diag.removeVertex(v);
    }

    bool checkSpiderFusion(ZXDiagram& diag, Vertex v0, Vertex v1) {
        auto edge_opt = diag.getEdge(v0, v1);
        return v0 != v1 && diag.type(v0) == diag.type(v1) &&
               edge_opt.value_or(Edge{0, EdgeType::Hadamard}).type ==
                       EdgeType::Simple &&
               diag.type(v0) != VertexType::Boundary;
    }

    void fuseSpiders(ZXDiagram& diag, Vertex v0, Vertex v1) {
        diag.addPhase(v0, diag.phase(v1));
        for (auto& [to, type]: diag.incidentEdges(v1)) {
            if (v0 != to)
                diag.addEdgeParallelAware(v0, to, type);
        }
        diag.removeVertex(v1);
    }

    bool checkLocalComp(ZXDiagram& diag, Vertex v) {
        auto v_data =
                diag.getVData(v).value_or(VertexData{0, 0, Expression(), VertexType::X});
        if (v_data.type != VertexType::Z || !v_data.phase.isProperClifford())
            return false;

        auto& edges = diag.incidentEdges(v);
        return std::all_of(edges.begin(), edges.end(), [&](auto& edge) {
            return edge.type == EdgeType::Hadamard &&
                   diag.type(edge.to) == VertexType::Z;
        });
    }

    void localComp(ZXDiagram& diag, Vertex v) { // TODO:scalars
        auto  phase = -diag.phase(v);
        auto& edges = diag.incidentEdges(v);

        for (size_t i = 0; i < edges.size(); i++) {
            auto& [n0, _] = edges[i];
            diag.addPhase(n0, phase);
            for (size_t j = i + 1; j < edges.size(); j++) {
                auto& [n1, _] = edges[j];
                diag.addEdgeParallelAware(n0, n1, EdgeType::Hadamard);
            }
        }
        diag.addGlobalPhase(PiRational{diag.phase(v).getConst().getNum(), 4});
        diag.removeVertex(v);
    }

    static bool isPauli(ZXDiagram& diag, Vertex v) {
        return diag.phase(v).isPauli();
    }

    bool checkPivotPauli(ZXDiagram& diag, Vertex v0, Vertex v1) {
        auto v0_data = diag.getVData(v0).value_or(
                VertexData{0, 0, Expression(), VertexType::X});
        auto v1_data = diag.getVData(v0).value_or(
                VertexData{0, 0, Expression(), VertexType::X});

        if (v0_data.type != VertexType::Z || // maybe problem if there is a self-loop?
            v1_data.type != VertexType::Z || !isPauli(diag, v0) ||
            !isPauli(diag, v1)) {
            return false;
        }

        auto edge_opt = diag.getEdge(v0, v1);

        if (!edge_opt.has_value() || edge_opt.value().type != EdgeType::Hadamard) {
            return false;
        }

        auto& edges_v0      = diag.incidentEdges(v0);
        auto  is_valid_edge = [&](const Edge& e) {
            return diag.type(e.to) == VertexType::Z && e.type == EdgeType::Hadamard;
        };

        if (!std::all_of(edges_v0.begin(), edges_v0.end(), is_valid_edge))
            return false;

        auto& edges_v1 = diag.incidentEdges(v1);

        return std::all_of(edges_v1.begin(), edges_v1.end(), is_valid_edge);
    }

    void pivotPauli(ZXDiagram& diag, Vertex v0, Vertex v1) { // TODO: phases

        auto phase_v0 = diag.phase(v0);
        auto phase_v1 = diag.phase(v1);

        if (!phase_v0.isZero() && !phase_v1.isZero())
            diag.addGlobalPhase(PiRational(1, 1));

        auto& edges_v0 = diag.incidentEdges(v0);
        auto& edges_v1 = diag.incidentEdges(v1);

        for (auto& [neighbor_v0, _]: edges_v0) {
            if (neighbor_v0 == v1) {
                continue;
            }

            diag.addPhase(neighbor_v0, phase_v1);
            for (auto& [neighbor_v1, _]: edges_v1) {
                if (neighbor_v1 != v0)
                    diag.addEdgeParallelAware(neighbor_v0, neighbor_v1,
                                              EdgeType::Hadamard);
            }
        }

        for (auto& [neighbor_v1, _]: edges_v1) {
            diag.addPhase(neighbor_v1, phase_v0);
        }

        diag.removeVertex(v0);
        diag.removeVertex(v1);
    }

    bool is_interior(ZXDiagram& diag, Vertex v) {
        auto& edges = diag.incidentEdges(v);
        return std::all_of(edges.begin(), edges.end(), [&](auto& edge) {
            return diag.degree(edge.to) > 1 && diag.type(edge.to) == VertexType::Z;
        });
    }

    bool checkPivot(ZXDiagram& diag, Vertex v0, Vertex v1) {
        auto v0_type = diag.type(v0);
        auto v1_type = diag.type(v1);

        if (v0 == v1 || v0_type != VertexType::Z || v1_type != VertexType::Z) {
            return false;
        }

        auto edge_opt = diag.getEdge(v0, v1);
        if (!edge_opt.has_value() || edge_opt.value().type != EdgeType::Hadamard) {
            return false;
        }

        auto& edges_v0        = diag.incidentEdges(v0);
        auto  is_invalid_edge = [&](const Edge& e) {
            auto to_type = diag.type(e.to);
            return !((to_type == VertexType::Z && e.type == EdgeType::Hadamard) ||
                     to_type == VertexType::Boundary);
        };

        if (std::any_of(edges_v0.begin(), edges_v0.end(), is_invalid_edge))
            return false;

        auto& edges_v1 = diag.incidentEdges(v1);

        if (std::any_of(edges_v1.begin(), edges_v1.end(), is_invalid_edge))
            return false;

        // auto is_interior = [&](Vertex v) {
        //   auto &edges = diag.incidentEdges(v);
        //   return std::all_of(edges.begin(), edges.end(), [&](auto &edge) {
        //     return diag.degree(edge.to) > 1 && diag.type(edge.to) == VertexType::Z;
        //   });
        // };

        auto is_interior_pauli = [&](Vertex v) {
            return is_interior(diag, v) && isPauli(diag, v);
        };

        return (is_interior_pauli(v0) || is_interior_pauli(v1));
    }

    static void extract_gadget(ZXDiagram& diag, Vertex v) {
        auto   v_data     = diag.getVData(v).value();
        Vertex phase_vert = diag.addVertex(v_data.qubit, -2, v_data.phase);
        Vertex id_vert    = diag.addVertex(v_data.qubit, -1);
        diag.setPhase(v, Expression(PiRational(0, 1)));
        diag.addHadamardEdge(v, id_vert);
        diag.addHadamardEdge(id_vert, phase_vert);
    }

    static void extract_pauli_gadget(ZXDiagram& diag, Vertex v) {
        if (diag.phase(v).isPauli())
            return;

        extract_gadget(diag, v);
    }

    static void ensure_interior(ZXDiagram& diag, Vertex v) {
        // auto &edges = diag.incidentEdges(v);
        // auto v_data = diag.getVData(v).value();
        // for (auto &[to, type] : edges) {
        //   if (diag.isBoundaryVertex(to)) {
        //     Vertex new_v = diag.addVertex(v_data.qubit, v_data.col, PiRational(0,
        //     1));

        //     auto other_dir = diag.getEdge(to, v);
        //     auto boundary_edge_type = type == zx::EdgeType::Simple
        //                                   ? zx::EdgeType::Hadamard
        //                                   : zx::EdgeType::Simple;

        //     auto& new_edges = diag.incidentEdges(new_v);
        //     new_edges.emplace_back(v, EdgeType::Hadamard);
        //     new_edges.emplace_back(to, boundary_edge_type);

        //     to = new_v;
        //     type = zx::EdgeType::Hadamard;

        //     other_dir.value().to = new_v;
        //     other_dir.value().type = boundary_edge_type;
        //   }
        // }

        auto edges  = diag.incidentEdges(v);
        auto v_data = diag.getVData(v).value();

        for (auto& [to, type]: edges) {
            if (!diag.isBoundaryVertex(to))
                continue;

            Vertex new_v =
                    diag.addVertex(v_data.qubit, v_data.col, Expression(PiRational(0, 1)));
            auto boundary_edge_type = type == zx::EdgeType::Simple ? zx::EdgeType::Hadamard : zx::EdgeType::Simple;

            diag.addEdge(v, new_v, EdgeType::Hadamard);
            diag.addEdge(to, new_v, boundary_edge_type);
            diag.removeEdge(v, to);
        }
    }

    static void ensure_pauli_vertex(ZXDiagram& diag, Vertex v) {
        extract_pauli_gadget(diag, v);
        ensure_interior(diag, v);
    }

    void pivot(ZXDiagram& diag, Vertex v0, Vertex v1) {
        ensure_pauli_vertex(diag, v0);
        ensure_pauli_vertex(diag, v1);

        pivotPauli(diag, v0, v1);
    }

    bool checkPivotGadget(ZXDiagram& diag, Vertex v0, Vertex v1) {
        auto p0 = diag.phase(v0);
        auto p1 = diag.phase(v1);
        if (!p0.isPauli()) {
            if (!p1.isPauli()) {
                return false;
            }
        } else if (p1.isPauli()) {
            return false;
        }
        if (!is_interior(diag, v0) || !is_interior(diag, v1))
            return false;

        return checkPivot(diag, v0, v1);
    }

    void pivotGadget(ZXDiagram& diag, Vertex v0, Vertex v1) {
        if (isPauli(diag, v0)) {
            extract_gadget(diag, v1);
        } else {
            extract_gadget(diag, v0);
        }
        pivotPauli(diag, v0, v1);
    }

    bool checkAndFuseGadget(ZXDiagram& diag, Vertex v) {
        if (diag.degree(v) != 1 || diag.isBoundaryVertex(v))
            return false;

        auto id0       = diag.incidentEdges(v)[0].to;
        auto id0_etype = diag.incidentEdges(v)[0].type;
        if (!isPauli(diag, id0) || diag.degree(id0) < 2 ||
            id0_etype != zx::EdgeType::Hadamard)
            return false;

        if (diag.degree(id0) == 2) {
            auto& [v0, v0_etype] = diag.incidentEdges(id0)[0].to == v ? diag.incidentEdges(id0)[1] : diag.incidentEdges(id0)[0];
            if (v0_etype != EdgeType::Hadamard)
                return false;

            if (diag.phase(id0).isZero())
                diag.addPhase(v0, diag.phase(v));
            else
                diag.addPhase(v0, -diag.phase(v));
            diag.removeVertex(v);
            diag.removeVertex(id0);
            return true;
        }

        std::optional<Vertex> n0;
        for (auto& [n, etype]: diag.incidentEdges(id0)) {
            if (n == v)
                continue;

            if (etype != zx::EdgeType::Hadamard)
                return false;
            n0 = n;
            // n0_etype = etype;
        }

        std::optional<Vertex> id1;
        std::optional<Vertex> phase_spider;

        bool found_gadget = false;
        for (auto& [n, etype]: diag.incidentEdges(n0.value())) {
            if (n == id0)
                continue;

            if (etype != zx::EdgeType::Hadamard || diag.isDeleted(n) ||
                !diag.phase(n).isPauli() || diag.degree(n) != diag.degree(id0) ||
                diag.connected(n, id0)) {
                continue;
            }

            found_gadget = true;
            id1          = n;

            for (auto& [nn, nn_etype]:
                 diag.incidentEdges(id1.value())) { // Todo: maybe problem with parallel edge?
                                                    // There shouldnt be any
                if (nn_etype != zx::EdgeType::Hadamard || diag.isDeleted(nn)) {
                    found_gadget = false;
                    break; // not a phase gadget
                }

                if (diag.degree(nn) == 1 && !diag.isBoundaryVertex(nn)) {
                    found_gadget = true;
                    phase_spider = nn;
                    continue;
                }

                if (std::find_if(diag.incidentEdges(nn).begin(),
                                 diag.incidentEdges(nn).end(), [&](Edge e) {
                                     return e.to == id0;
                                 }) == diag.incidentEdges(nn).end()) {
                    found_gadget = false;
                    break;
                }
            }

            if (found_gadget)
                break;
        }

        if (!found_gadget || !phase_spider.has_value())
            return false;

        if (!diag.phase(id0).isZero()) {
            diag.setPhase(v, -diag.phase(v));
            diag.setPhase(id0, Expression(PiRational(0, 1)));
        }
        if (diag.phase(id1.value()).isZero())
            diag.addPhase(v, diag.phase(phase_spider.value()));
        else
            diag.addPhase(v, -diag.phase(phase_spider.value()));
        diag.removeVertex(phase_spider.value());
        diag.removeVertex(id1.value());
        return true;
    }

} // namespace zx
