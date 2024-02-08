#pragma once

#include "operations/Expression.hpp"
#include "zx/Utils.hpp"
#include "zx/ZXDefinitions.hpp"

#include <cstddef>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace zx {
class ZXDiagram {
public:
  ZXDiagram() = default;
  // create n_qubit identity_diagram
  explicit ZXDiagram(std::size_t nqubits);

  void addEdge(Vertex from, Vertex to, EdgeType type = EdgeType::Simple);
  void addHadamardEdge(const Vertex from, const Vertex to) {
    addEdge(from, to, EdgeType::Hadamard);
  };
  void addEdgeParallelAware(Vertex from, Vertex to,
                            EdgeType eType = EdgeType::Simple);
  void removeEdge(Vertex from, Vertex to);

  Vertex addVertex(const VertexData& data);
  Vertex addVertex(Qubit qubit, Col col = 0,
                   const PiExpression& phase = PiExpression(),
                   VertexType type = VertexType::Z);
  void addQubit();
  void addQubits(zx::Qubit n);
  void removeVertex(Vertex toRemove);

  [[nodiscard]] std::size_t getNdeleted() const { return deleted.size(); }
  [[nodiscard]] std::size_t getNVertices() const { return nvertices; }
  [[nodiscard]] std::size_t getNEdges() const { return nedges; }
  [[nodiscard]] std::size_t getNQubits() const { return inputs.size(); }

  [[nodiscard]] bool connected(Vertex from, Vertex to) const;
  [[nodiscard]] std::optional<Edge> getEdge(Vertex from, Vertex to) const;
  [[nodiscard]] const std::vector<Edge>& incidentEdges(const Vertex v) const {
    return edges[v];
  }
  [[nodiscard]] const Edge& incidentEdge(const Vertex v, const std::size_t n) {
    return edges[v][n];
  }

  [[nodiscard]] std::size_t degree(const Vertex v) const {
    return edges[v].size();
  }

  [[nodiscard]] const PiExpression& phase(const Vertex v) const {
    const auto& vertex = vertices[v];
    assert(vertex.has_value());
    return vertex->phase; // NOLINT(bugprone-unchecked-optional-access)
  }

  [[nodiscard]] Qubit qubit(const Vertex v) const {
    const auto& vertex = vertices[v];
    assert(vertex.has_value());
    return vertex->qubit; // NOLINT(bugprone-unchecked-optional-access)
  }

  [[nodiscard]] VertexType type(const Vertex v) const {
    const auto& vertex = vertices[v];
    assert(vertex.has_value());
    return vertex->type; // NOLINT(bugprone-unchecked-optional-access)
  }

  [[nodiscard]] std::optional<VertexData> getVData(const Vertex v) const {
    return vertices[v];
  }

  [[nodiscard]] std::vector<std::pair<Vertex, const VertexData&>>
  getVertices() const;
  [[nodiscard]] std::vector<std::pair<Vertex, Vertex>> getEdges() const;

  [[nodiscard]] const std::vector<Vertex>& getInputs() const { return inputs; }
  [[nodiscard]] Vertex getInput(const std::size_t i) const { return inputs[i]; }

  [[nodiscard]] const std::vector<Vertex>& getOutputs() const {
    return outputs;
  }

  [[nodiscard]] Vertex getOutput(const std::size_t i) const {
    return outputs[i];
  }

  [[nodiscard]] bool isDeleted(const Vertex v) const {
    return !vertices[v].has_value();
  }

  [[nodiscard]] bool isBoundaryVertex(const Vertex v) const {
    return type(v) == VertexType::Boundary;
  }

  [[nodiscard]] bool isInput(Vertex v) const;
  [[nodiscard]] bool isOutput(Vertex v) const;

  void addPhase(const Vertex v, const PiExpression& phase) {
    auto& vertex = vertices[v];
    if (vertex.has_value()) {
      vertex->phase += phase;
    }
  }

  void setPhase(const Vertex v, const PiExpression& phase) {
    auto& vertex = vertices[v];
    if (vertex.has_value()) {
      vertex->phase = phase;
    }
  }

  void setType(const Vertex v, const VertexType type) {
    auto& vertex = vertices[v];
    if (vertex.has_value()) {
      vertex->type = type;
    }
  }

  void toGraphlike();

  [[nodiscard]] bool isIdentity() const;

  [[nodiscard]] ZXDiagram adjoint() const;

  ZXDiagram& invert();

  ZXDiagram& concat(const ZXDiagram& rhs);
  ZXDiagram& operator+=(const ZXDiagram& rhs) { return this->concat(rhs); }

  void makeAncilla(Qubit qubit);
  void makeAncilla(Qubit in, Qubit out);

  void approximateCliffords(fp tolerance);

  void removeDisconnectedSpiders();

  void addGlobalPhase(const PiExpression& phase);
  [[nodiscard]] PiExpression getGlobalPhase() const { return globalPhase; }
  [[nodiscard]] bool globalPhaseIsZero() const { return globalPhase.isZero(); }
  [[nodiscard]] gf2Mat getAdjMat() const;
  [[nodiscard]] std::vector<Vertex>
  getConnectedSet(const std::vector<Vertex>& s,
                  const std::vector<Vertex>& exclude = {}) const;
  static bool isIn(const Vertex& v, const std::vector<Vertex>& vertices);

private:
  std::vector<std::vector<Edge>> edges;
  std::vector<std::optional<VertexData>> vertices;
  std::vector<Vertex> deleted;
  std::vector<Vertex> inputs;
  std::vector<Vertex> outputs;
  std::size_t nvertices = 0;
  std::size_t nedges = 0;
  PiExpression globalPhase = {};

  std::vector<Vertex> initGraph(std::size_t nqubits);
  void closeGraph(const std::vector<Vertex>& qubitVertices);

  void removeHalfEdge(Vertex from, Vertex to);

  std::vector<Edge>::iterator getEdgePtr(Vertex from, Vertex to);
};
} // namespace zx
