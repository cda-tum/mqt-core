#ifndef JKQZX_INCLUDE_GRAPH_HPP_
#define JKQZX_INCLUDE_GRAPH_HPP_

#include <algorithm>
#include <numeric>
#include <optional>
#include <stdint.h>
#include <string>
#include <unordered_map>
#include <vector>

#include "Rational.hpp"
#include "dd/Definitions.hpp"

namespace zx {

enum class EdgeType { Simple, Hadamard };
enum class VertexType { Boundary, Z, X };
using Vertex = int32_t;
using Col = int32_t;

struct Edge {
  int32_t to;
  EdgeType type;

  void toggle() {
    type = (type == EdgeType::Simple) ? EdgeType::Hadamard : EdgeType::Simple;
  }
};

struct VertexData {
  Col col;
  dd::Qubit qubit;
  Rational phase;
  VertexType type;
};

class ZXDiagram {
public:
  ZXDiagram() = default;
  explicit ZXDiagram(std::string filename);

  void add_edge(Vertex from, Vertex to, EdgeType type = EdgeType::Simple);
  void remove_edge(Vertex from, Vertex to);

  Vertex add_vertex(const VertexData &data);
  void remove_vertex(Vertex to_remove);

  [[nodiscard]] int32_t get_nvertices() const { return nvertices; }
  [[nodiscard]] int32_t get_nedges() const { return nedges; }

  [[nodiscard]] std::optional<Edge> get_edge(Vertex from, Vertex to) const;

  [[nodiscard]] const std::vector<Edge> &incident_edges(Vertex v) const {
    return edges[v];
  }

  [[nodiscard]] std::optional<VertexData> get_vdata(Vertex v) const {
    return vertices[v];
  }

  [[nodiscard]] const std::vector<Vertex> &get_inputs() const { return inputs; }

  [[nodiscard]] const std::vector<Vertex> &get_outputs() const {
    return outputs;
  }

  [[nodiscard]] bool is_deleted(Vertex v) const {
    if (std::find(deleted.begin(), deleted.end(), v) != deleted.end())
      return true;
    return false;
  }

private:
  std::vector<std::vector<Edge>> edges;
  std::vector<std::optional<VertexData>> vertices;
  std::vector<Vertex> deleted;
  std::vector<Vertex> inputs;
  std::vector<Vertex> outputs;
  int32_t nvertices = 0;
  int32_t nedges = 0;

  void add_z_spider(dd::Qubit qubit, std::vector<Vertex> &qubit_vertices,
                    Rational phase = 0, EdgeType type = EdgeType::Simple);
  void add_x_spider(dd::Qubit qubit, std::vector<Vertex> &qubit_vertices,
                    Rational phase = 0, EdgeType type = EdgeType::Simple);
  void add_cnot(dd::Qubit ctrl, dd::Qubit target,
                std::vector<Vertex> &qubit_vertices);

  std::vector<Vertex> init_graph(int nqubits);
  void close_graph(std::vector<Vertex> &qubit_vertices);

  void remove_half_edge(Vertex from, Vertex to);
};
} // namespace zx
#endif /* JKQZX_INCLUDE_GRAPH_HPP_ */
