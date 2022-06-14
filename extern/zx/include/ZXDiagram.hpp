#ifndef ZX_INCLUDE_GRAPH_HPP_
#define ZX_INCLUDE_GRAPH_HPP_

#include "Definitions.hpp"
#include "Expression.hpp"
#include "Rational.hpp"
#include "Utils.hpp"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <numeric>
#include <optional>
#include <stdint.h>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace zx {
    class ZXDiagram {
    public:
        ZXDiagram() = default;
        ZXDiagram(int32_t nqubits); // create n_qubit identity_diagram
        explicit ZXDiagram(std::string filename);
        // explicit ZXDiagram(const qc::QuantumComputation &circuit);

        void addEdge(Vertex from, Vertex to, EdgeType type = EdgeType::Simple);
        void addHadamardEdge(Vertex from, Vertex to) {
            addEdge(from, to, EdgeType::Hadamard);
        };
        void addEdgeParallelAware(Vertex from, Vertex to,
                                  EdgeType type = EdgeType::Simple);
        void remove_edge(Vertex from, Vertex to);

        Vertex addVertex(const VertexData& data);
        Vertex addVertex(Qubit qubit, Col col = 0,
                         const Expression& phase = Expression(),
                         VertexType        type  = VertexType::Z);
        void   removeVertex(Vertex to_remove);

        int32_t               get_ndeleted() const { return deleted.size(); }
        [[nodiscard]] int32_t getNVertices() const { return nvertices; }
        [[nodiscard]] int32_t getNEdges() const { return nedges; }
        [[nodiscard]] int32_t getNQubits() const { return inputs.size(); }

        [[nodiscard]] bool                connected(Vertex from, Vertex to) const;
        [[nodiscard]] std::optional<Edge> getEdge(Vertex from, Vertex to) const;
        [[nodiscard]] std::vector<Edge>&  incidentEdges(Vertex v) {
             return edges[v];
        }

        [[nodiscard]] int32_t degree(Vertex v) const { return edges[v].size(); }

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

        // What about Swaps?

        void makeAncilla(Qubit qubit);

    private:
        std::vector<std::vector<Edge>>         edges;
        std::vector<std::optional<VertexData>> vertices;
        std::vector<Vertex>                    deleted;
        std::vector<Vertex>                    inputs;
        std::vector<Vertex>                    outputs;
        int32_t                                nvertices = 0;
        int32_t                                nedges    = 0;

        void addZSpider(Qubit qubit, std::vector<Vertex>& qubit_vertices,
                        const Expression& phase = Expression(),
                        EdgeType          type  = EdgeType::Simple);
        void addXSpider(Qubit qubit, std::vector<Vertex>& qubit_vertices,

                        const Expression& phase = Expression(),
                        EdgeType          type  = EdgeType::Simple);
        void addCnot(Qubit ctrl, Qubit target,
                     std::vector<Vertex>& qubit_vertices);
      
        void addCphase(PiRational phase, Qubit ctrl, Qubit target,
                       std::vector<Vertex>& qubit_vertices);
        void addSwap(Qubit ctrl, Qubit target,
                     std::vector<Vertex>& qubit_vertices);
        void addCcx(Qubit ctrl_0, Qubit ctrl_1, Qubit target,
                    std::vector<Vertex>& qubit_vertices);

        std::vector<Vertex> initGraph(int nqubits);
        void                closeGraph(std::vector<Vertex>& qubit_vertices);

        void removeHalfEdge(Vertex from, Vertex to);

        std::vector<Edge>::iterator getEdgePtr(Vertex from, Vertex to);

    };
} // namespace zx
#endif /* ZX_INCLUDE_GRAPH_HPP_ */
