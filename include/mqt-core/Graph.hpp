//
// This file is part of the MQT QMAP library released under the MIT license.
// See README.md or go to https://github.com/cda-tum/mqt-qmap for more
// information.
//

#pragma once

#include "Definitions.hpp"

#include <algorithm>
#include <numeric>
#include <sstream>
#include <unordered_set>

namespace qc {

template <class V, class E> class Graph {
protected:
  // the adjecency matrix works with indices
  std::vector<std::vector<E>> adjacencyMatrix{};
  // the mapping of vertices to indices in the graph are stored in a map
  std::unordered_map<V, std::size_t> mapping;
  // the inverse mapping is used to get the vertex from the index
  std::unordered_map<std::size_t, V> invMapping;
  std::size_t nVertices = 0;

public:
  auto addVertex(V v) -> void {
    // check whether the vertex is already in the graph, if so do nothing
    if (mapping.find(v) == mapping.end()) {
      mapping[v] = nVertices;
      invMapping[nVertices] = v;
      ++nVertices;
      for (auto& row : adjacencyMatrix) {
        row.emplace_back(nullptr);
      }
      adjacencyMatrix.emplace_back(1, nullptr);
    }
  }
  auto addEdge(V u, V v, E e) -> void {
    if (mapping.find(u) == mapping.end()) {
      addVertex(u);
    }
    if (mapping.find(v) == mapping.end()) {
      addVertex(v);
    }
    std::size_t const i = mapping.at(u);
    std::size_t const j = mapping.at(v);
    if (i < j) {
      adjacencyMatrix[i][j - i] = e;
    } else {
      adjacencyMatrix[j][i - j] = e;
    }
  }
  [[nodiscard]] auto getNVertices() const -> std::size_t { return nVertices; }
  [[nodiscard]] auto getNEdges() const -> std::size_t {
    return std::accumulate(
        adjacencyMatrix.cbegin(), adjacencyMatrix.cend(), 0UL,
        [](const std::size_t acc, const auto& row) {
          return acc + static_cast<std::size_t>(std::count_if(
                           row.cbegin(), row.cend(),
                           [](const auto& edge) { return edge != nullptr; }));
        });
  }
  [[nodiscard]] auto getEdge(V v, V u) const -> E {
    std::size_t const i = mapping.at(v);
    std::size_t const j = mapping.at(u);
    if (i < j ? adjacencyMatrix[i][j - i] != nullptr
              : adjacencyMatrix[j][i - j] != nullptr) {
      return i < j ? adjacencyMatrix[i][j - i] : adjacencyMatrix[j][i - j];
    }
    std::stringstream ss;
    ss << "The edge (" << v << ", " << u << ") does not exist.";
    throw std::invalid_argument(ss.str());
  }
  [[nodiscard]] auto getDegree(V v) const -> std::size_t {
    if (mapping.find(v) == mapping.end()) {
      std::stringstream ss;
      ss << "The vertex " << v << " is not in the graph.";
      throw std::invalid_argument(ss.str());
    }
    std::size_t const i = mapping.at(v);
    std::size_t degree = 0;
    for (std::size_t j = 0; j < nVertices; ++j) {
      if ((i <= j and adjacencyMatrix[i][j - i] != nullptr) or
          (j < i and adjacencyMatrix[j][i - j] != nullptr)) {
        ++degree;
      }
    }
    return degree;
  }
  [[nodiscard]] auto getVertices() const -> std::unordered_set<V> {
    return std::accumulate(mapping.cbegin(), mapping.cend(),
                           std::unordered_set<V>(),
                           [](auto& acc, const auto& v) {
                             acc.emplace(v.first);
                             return acc;
                           });
  }
  [[nodiscard]] auto isAdjacent(const V u, const V v) const -> bool {
    std::size_t const i = mapping.at(u);
    std::size_t const j = mapping.at(v);
    return (i < j and adjacencyMatrix[i][j - i] != nullptr) or
           (j < i and adjacencyMatrix[j][i - j] != nullptr);
  }
  [[nodiscard]] auto
  // NOLINTNEXTLINE(readability-convert-member-functions-to-static)
  isAdjacentEdge(const std::pair<V, V>& e, const std::pair<V, V>& f) const
      -> bool {
    return e.first == f.first or e.first == f.second or e.second == f.first or
           e.second == f.second;
  }
  /// Outputs a string representation of the graph in the DOT format
  [[nodiscard]] auto toString() const -> std::string {
    std::stringstream ss;
    ss << "graph {\n";
    for (const auto& [v, i] : mapping) {
      ss << "  " << i << " [label=\"" << v << "\"];\n";
    }
    for (std::size_t i = 0; i < nVertices; ++i) {
      for (std::size_t j = i + 1; j < nVertices; ++j) {
        if (adjacencyMatrix[i][j - i] != nullptr) {
          ss << "  " << i << " -- " << j << ";\n";
        }
      }
    }
    ss << "}\n";
    return ss.str();
  }
  friend auto operator<<(std::ostream& os, const Graph& g) -> std::ostream& {
    os << g.toString(); // Using toString() method
    return os;
  }
};
} // namespace qc
