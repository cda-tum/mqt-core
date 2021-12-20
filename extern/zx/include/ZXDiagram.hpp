#ifndef JKQZX_INCLUDE_GRAPH_HPP_
#define JKQZX_INCLUDE_GRAPH_HPP_

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
  explicit ZXDiagram(std::string &filename);

  void add_edge(Vertex from, Vertex to, EdgeType type);
  Vertex add_vertex(Rational phase);
  Vertex add_vertex(const VertexData &data);
  void remove_vertex(Vertex to_remove);

private:
  std::vector<std::vector<Edge>> edges;
  std::vector<std::optional<VertexData>> vertices;
  std::vector<Vertex> deleted;
  std::vector<Vertex> inputs;
  std::vector<Vertex> outputs;
  int32_t nvertices;

  void add_z_spider(dd::Qubit qubit, std::vector<Vertex> qubit_vertices,
                    Rational phase = 0, EdgeType type = EdgeType::Simple);
  void add_x_spider(dd::Qubit qubit, std::vector<Vertex> qubit_vertices,
                    Rational phase = 0, EdgeType type = EdgeType::Simple);
  void add_cnot(dd::Qubit ctrl, dd::Qubit target,
                std::vector<Vertex> qubit_vertices);

  std::vector<Vertex> init_graph(int nqubits);
  void close_graph(std::vector<Vertex> qubit_vertices);
  };
} // namespace zx
#endif /* JKQZX_INCLUDE_GRAPH_HPP_ */
