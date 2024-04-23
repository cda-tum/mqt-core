//
// This file is part of the MQT QMAP library released under the MIT license.
// See README.md or go to https://github.com/cda-tum/mqt-qmap for more
// information.
//

#pragma once

#include "Definitions.hpp"

#include <numeric>
#include <sstream>
#include <unordered_set>

namespace qc {

template <class V> class DirectedGraph {
protected:
  // the adjecency matrix works with indices
  std::vector<std::vector<bool>> adjacencyMatrix;
  // the mapping of vertices to indices in the graph are stored in a map
  std::unordered_map<V, std::size_t> mapping;
  // the inverse mapping is used to get the vertex from the index
  std::unordered_map<std::size_t, V> invMapping;
  // the number of vertices in the graph
  std::size_t nVertices = 0;
  // the number of edges in the graph
  std::size_t nEdges = 0;
  // the in-degrees of the vertices in the graph
  std::vector<std::size_t> inDegrees;
  // the out-degrees of the vertices in the graph
  std::vector<std::size_t> outDegrees;

public:
  virtual ~DirectedGraph() = default;
  virtual auto addVertex(V v) -> void {
    // check whether the vertex is already in the graph, if so do nothing
    if (mapping.find(v) == mapping.end()) {
      mapping[v] = nVertices;
      invMapping[nVertices] = v;
      ++nVertices;
      for (auto& row : adjacencyMatrix) {
        row.emplace_back(false);
      }
      adjacencyMatrix.emplace_back(nVertices, false);
      inDegrees.emplace_back(0);
      outDegrees.emplace_back(0);
    } else {
      std::stringstream ss;
      ss << "The vertex " << v << " is already in the graph.";
      throw std::invalid_argument(ss.str());
    }
  }
  virtual auto addEdge(V u, V v) -> void {
    if (mapping.find(u) == mapping.end()) {
      addVertex(u);
    }
    if (mapping.find(v) == mapping.end()) {
      addVertex(v);
    }
    const auto i = mapping.at(u);
    const auto j = mapping.at(v);
    if (!adjacencyMatrix[i][j]) {
      adjacencyMatrix[i][j] = true;
      ++outDegrees[i];
      ++inDegrees[j];
      ++nEdges;
    }
  }
  [[nodiscard]] auto getNVertices() const -> std::size_t { return nVertices; }
  [[nodiscard]] auto getNEdges() const -> std::size_t { return nEdges; }
  [[nodiscard]] auto getInDegree(V v) const -> std::size_t {
    if (mapping.find(v) == mapping.end()) {
      std::stringstream ss;
      ss << "The vertex " << v << " is not in the graph.";
      throw std::invalid_argument(ss.str());
    }
    const auto i = mapping.at(v);
    return inDegrees[i];
  }
  [[nodiscard]] auto getOutDegree(V v) const -> std::size_t {
    if (mapping.find(v) == mapping.end()) {
      std::stringstream ss;
      ss << "The vertex " << v << " is not in the graph.";
      throw std::invalid_argument(ss.str());
    }
    const auto i = mapping.at(v);
    return outDegrees[i];
  }
  [[nodiscard]] auto getVertices() const -> std::unordered_set<V> {
    return std::accumulate(mapping.cbegin(), mapping.cend(),
                           std::unordered_set<V>(),
                           [](auto& acc, const auto& v) {
                             acc.emplace(v.first);
                             return acc;
                           });
  }
  [[nodiscard]] auto isEdge(const V u, const V v) const -> bool {
    const auto i = mapping.at(u);
    const auto j = mapping.at(v);
    return adjacencyMatrix[i][j];
  }
  /// Outputs a string representation of the graph in the DOT format
  [[nodiscard]] auto toString() const -> std::string {
    std::stringstream ss;
    ss << "digraph {\n";
    for (const auto& [v, i] : mapping) {
      ss << "  " << i << " [label=\"" << v << "\"];\n";
    }
    for (std::size_t i = 0; i < nVertices; ++i) {
      for (std::size_t j = 0; j < nVertices; ++j) {
        if (adjacencyMatrix[i][j]) {
          ss << "  " << i << " -> " << j << ";\n";
        }
      }
    }
    ss << "}\n";
    return ss.str();
  }
  friend auto operator<<(std::ostream& os,
                         const DirectedGraph& g) -> std::ostream& {
    os << g.toString(); // Using toString() method
    return os;
  }
};
} // namespace qc
