#ifndef JKQZX_INCLUDE_GRAPH_HPP_
#define JKQZX_INCLUDE_GRAPH_HPP_

#include <stdint.h>
#include <unordered_map>
#include <vector>
#include <optional>

#include "Rational.hpp"

enum class EdgeType { Simple, Hadamard };
enum class VertexType {Boundary, Z, X};
using Vertex = int32_t;

struct Edge {
  int32_t to;
  EdgeType type;

  void toggle() {
    type = (type == EdgeType::Simple) ? EdgeType::Hadamard : EdgeType::Simple;
  }
};

struct VertexData {
  int32_t row;
  int32_t qubit;
  Rational phase;
  VertexType type;
};

class Graph {
public:
  void add_edge(int32_t from, int32_t to, EdgeType type);
  void add_vertex(Rational phase);
  void add_vertex(const VertexData &data);
  void remove_vertex(Vertex to_remove);
  
private:
  std::vector<std::vector<Edge>> adj_lst;
  std::vector<std::optional<VertexData>> vertices;
  std::vector<Vertex> deleted;
  std::vector<Vertex> inputs;
  std::vector<Vertex> outputs;
  int32_t nvertices;
};

#endif /* JKQZX_INCLUDE_GRAPH_HPP_ */
