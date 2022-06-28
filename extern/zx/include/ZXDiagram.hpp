#pragma once

#include "Definitions.hpp"
#include "Expression.hpp"
#include "Rational.hpp"
#include "Utils.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace zx {
    class ZXDiagram {
    public:
        ZXDiagram() = default;
        explicit ZXDiagram(std::size_t nqubits); // create n_qubit identity_diagram
        explicit ZXDiagram(std::string filename);
        // explicit ZXDiagram(const qc::QuantumComputation &circuit);

        void addEdge(Vertex from, Vertex to, EdgeType type = EdgeType::Simple);
        void addHadamardEdge(Vertex from, Vertex to) {
            addEdge(from, to, EdgeType::Hadamard);
        };
        void addEdgeParallelAware(Vertex from, Vertex to,
                                  EdgeType type = EdgeType::Simple);
        void removeEdge(Vertex from, Vertex to);

        Vertex addVertex(const VertexData& data);
        Vertex addVertex(Qubit qubit, Col col = 0,
                         const Expression& phase = Expression(),
                         VertexType        type  = VertexType::Z);
        void   addQubit();
        void   addQubits(zx::Qubit n);
        void   removeVertex(Vertex to_remove);

        [[nodiscard]] std::size_t getNdeleted() const { return deleted.size(); }
        [[nodiscard]] std::size_t getNVertices() const { return nvertices; }
        [[nodiscard]] std::size_t getNEdges() const { return nedges; }
        [[nodiscard]] std::size_t getNQubits() const { return inputs.size(); }

        [[nodiscard]] bool                connected(Vertex from, Vertex to) const;
        [[nodiscard]] std::optional<Edge> getEdge(Vertex from, Vertex to) const;
        [[nodiscard]] std::vector<Edge>&  incidentEdges(Vertex v) {
             return edges[v];
        }

        [[nodiscard]] std::size_t degree(Vertex v) const { return edges[v].size(); }

        [[nodiscard]] const Expression& phase(Vertex v) const {
            return vertices[v].value().phase;
        }

        [[nodiscard]] Qubit qubit(Vertex v) const {
            return vertices[v].value().qubit;
        }

        [[nodiscard]] VertexType type(Vertex v) const {
            return vertices[v].value().type;
        }

        [[nodiscard]] std::optional<VertexData> getVData(Vertex v) const {
            return vertices[v];
        }

        [[nodiscard]] std::vector<std::pair<Vertex, VertexData&>> getVertices();
        [[nodiscard]] std::vector<std::pair<Vertex, Vertex>>      getEdges();

        [[nodiscard]] const std::vector<Vertex>& getInputs() const {
            return inputs;
        }

        [[nodiscard]] const std::vector<Vertex>& getOutputs() const {
            return outputs;
        }

        [[nodiscard]] bool isDeleted(Vertex v) const {
            return !vertices[v].has_value();
        }

        [[nodiscard]] bool isBoundaryVertex(Vertex v) const {
            return vertices[v].value().type == VertexType::Boundary;
        }

        [[nodiscard]] bool isInput(Vertex v) const;
        [[nodiscard]] bool isOutput(Vertex v) const;

        void addPhase(Vertex v, const Expression& phase) {
            vertices[v].value().phase += phase;
        }

        void setPhase(Vertex v, const Expression& phase) {
            vertices[v].value().phase = phase;
        }

        void setType(Vertex v, VertexType type) {
            vertices[v].value().type = type;
        }

        void toGraphlike();

        [[nodiscard]] bool isIdentity() const;

        [[nodiscard]] ZXDiagram adjoint() const;

        ZXDiagram& invert();

        ZXDiagram& concat(const ZXDiagram& rhs);
        ZXDiagram& operator+=(const ZXDiagram& rhs) { return this->concat(rhs); }

        void makeAncilla(Qubit qubit);

        void approximateCliffords(fp tolerance);

        void removeDisconnectedSpiders();

    private:
        std::vector<std::vector<Edge>>         edges;
        std::vector<std::optional<VertexData>> vertices;
        std::vector<Vertex>                    deleted;
        std::vector<Vertex>                    inputs;
        std::vector<Vertex>                    outputs;
        std::size_t                            nvertices = 0;
        std::size_t                            nedges    = 0;

        std::vector<Vertex> initGraph(std::size_t nqubits);
        void                closeGraph(std::vector<Vertex>& qubit_vertices);

        void removeHalfEdge(Vertex from, Vertex to);

        std::vector<Edge>::iterator getEdgePtr(Vertex from, Vertex to);
    };
} // namespace zx
