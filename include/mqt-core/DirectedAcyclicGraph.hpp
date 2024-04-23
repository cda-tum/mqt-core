//
// This file is part of the MQT QMAP library released under the MIT license.
// See README.md or go to https://github.com/cda-tum/mqt-qmap for more
// information.
//

#pragma once

#include "Definitions.hpp"
#include "DirectedGraph.hpp"

#include <algorithm>
#include <cassert>
#include <sstream>
#include <stack>
#include <stdexcept>

namespace qc {

template <class V>
class DirectedAcyclicGraph : public DirectedGraph<V> {
protected:
  // transitive closure matrix to detect cycles
  std::vector<std::vector<bool>> closureMatrix;

public:
  auto addVertex(V v) -> void override {
    DirectedGraph<V>::addVertex(v);
    for (auto& row : closureMatrix) {
      row.emplace_back(false);
    }
    closureMatrix.emplace_back(DirectedGraph<V>::nVertices, false);
    const auto i = DirectedGraph<V>::mapping.at(v);
    closureMatrix[i][i] = true;
  }
  auto addEdge(V u, V v) -> void override {
    if (DirectedGraph<V>::mapping.find(u) ==
        DirectedGraph<V>::mapping.end()) {
      addVertex(u);
    }
    if (DirectedGraph<V>::mapping.find(v) ==
        DirectedGraph<V>::mapping.end()) {
      addVertex(v);
    }
    std::size_t const i = DirectedGraph<V>::mapping.at(u);
    std::size_t const j = DirectedGraph<V>::mapping.at(v);
    if (closureMatrix[j][i]) {
      std::ostringstream oss;
      oss << "Adding edge (" << u << ", " << v << ") would create a cycle.";
      throw std::logic_error(oss.str());
    }
    DirectedGraph<V>::addEdge(u, v);
    for (std::size_t k = 0; k < DirectedGraph<V>::nVertices; ++k) {
      if (closureMatrix[k][i]) {
        closureMatrix[k][j] = true;
      }
      if (closureMatrix[j][k]) {
        closureMatrix[i][k] = true;
      }
    }
  }
  [[nodiscard]] auto isReachable(V u, V v) const -> bool {
    if (DirectedGraph<V>::mapping.find(u) ==
        DirectedGraph<V>::mapping.end()) {
      throw std::invalid_argument("Vertex u not in graph.");
    }
    if (DirectedGraph<V>::mapping.find(v) ==
        DirectedGraph<V>::mapping.end()) {
      throw std::invalid_argument("Vertex v not in graph.");
    }
    return closureMatrix[DirectedGraph<V>::mapping.at(u)]
                        [DirectedGraph<V>::mapping.at(v)];
  }
  /// Perform a depth-first search on the graph and return the nodes in a
  /// topological order
  [[nodiscard]] auto orderTopologically() const -> std::vector<V> {
    std::stack<std::size_t> stack{};
    std::vector<std::size_t> result{};
    std::vector visited(DirectedGraph<V>::nVertices, false);
    // visitedInDegree is used to count the incoming edges that have been
    // visited already such that the resulting order of the nodes is one that
    // satisfies a topological ordering
    std::vector<std::size_t> visitedInDegree(DirectedGraph<V>::nVertices, 0);
    // Push nodes with 0 indegree onto the stack
    for (std::size_t k = 0; k < DirectedGraph<V>::nVertices; ++k) {
      if (DirectedGraph<V>::inDegrees[k] == 0) {
        stack.push(k);
        visited[k] = true;
      }
    }
    // Perform DFS
    while (!stack.empty()) {
      const auto u = stack.top();
      stack.pop();
      result.push_back(u);

      for (std::size_t k = 0; k < DirectedGraph<V>::nVertices; ++k) {
        if (DirectedGraph<V>::adjacencyMatrix[u][k]) {
          if (not visited[k]) {
            if (++visitedInDegree[k] == DirectedGraph<V>::inDegrees[k]) {
              stack.push(k);
              visited[k] = true;
            }
          }
        }
      }
    }
    // Otherwise graph has a cycle
    assert(result.size() == (DirectedGraph<V>::nVertices));
    std::vector<V> vertices;
    std::transform(
        result.cbegin(), result.cend(), std::back_inserter(vertices),
        [&](const auto i) { return DirectedGraph<V>::invMapping.at(i); });
    return vertices;
  }
};
} // namespace qc
