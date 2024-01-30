#pragma once

#include "operations/Expression.hpp"
#include "zx/ZXDefinitions.hpp"

#include <cstdint>
#include <iterator>
#include <optional>
#include <utility>
#include <vector>

namespace zx {

struct Edge {
  Vertex to;
  EdgeType type;

  Edge() = default;
  Edge(const Vertex t, const EdgeType typ) : to(t), type(typ){};
  void toggle() {
    if (type == EdgeType::Simple) {
      type = EdgeType::Hadamard;
    } else {
      type = EdgeType::Simple;
    }
  }
};

struct VertexData {
  Col col;
  Qubit qubit;
  PiExpression phase;
  VertexType type;
};

class Vertices {
public:
  explicit Vertices(const std::vector<std::optional<VertexData>>& verts)
      : vertices(verts){};

  class VertexIterator {
  public:
    using iterator_category = std::forward_iterator_tag;
    using difference_type = std::int32_t;
    using value_type = std::pair<Vertex, const VertexData&>;
    using pointer = value_type*;
    using reference = value_type&;

    explicit VertexIterator(const std::vector<std::optional<VertexData>>& verts)
        : currentPos(verts.begin()), vertices(verts) {
      nextValidVertex();
    }
    VertexIterator(const std::vector<std::optional<VertexData>>& verts,
                   Vertex vertex);

    value_type operator*() const {
      assert(currentPos->has_value());
      // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
      return {v, currentPos->value()};
    }

    // Prefix increment
    VertexIterator operator++();

    // Postfix increment
    VertexIterator operator++(int);

    friend bool operator==(const VertexIterator& a, const VertexIterator& b);
    friend bool operator!=(const VertexIterator& a, const VertexIterator& b);

  private:
    Vertex v = 0;
    std::vector<std::optional<VertexData>>::const_iterator currentPos;
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
    const std::vector<std::optional<VertexData>>& vertices;

    void nextValidVertex();
  };

  using iterator = VertexIterator;

  iterator begin() { return VertexIterator(vertices); }
  iterator end() { return {vertices, vertices.size()}; }

private:
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  const std::vector<std::optional<VertexData>>& vertices;
};

class Edges {
public:
  Edges(const std::vector<std::vector<Edge>>& edgs,
        const std::vector<std::optional<VertexData>>& verts)
      : edges(edgs), vertices(verts){};

  class EdgeIterator {
  public:
    using iterator_category = std::forward_iterator_tag;
    using difference_type = std::int32_t;
    using value_type = std::pair<Vertex, Vertex>;
    using pointer = value_type*;
    using reference = value_type&;

    EdgeIterator(const std::vector<std::vector<Edge>>& es,
                 const std::vector<std::optional<VertexData>>& verts);

    EdgeIterator(const std::vector<std::vector<Edge>>& es,
                 const std::vector<std::optional<VertexData>>& verts,
                 Vertex vertex);

    value_type operator*() const { return {v, currentPos->to}; }

    // Prefix increment
    EdgeIterator operator++();

    // Postfix increment
    EdgeIterator operator++(int);

    friend bool operator==(const EdgeIterator& a, const EdgeIterator& b);
    friend bool operator!=(const EdgeIterator& a, const EdgeIterator& b);

  private:
    Vertex v;
    std::vector<Edge>::const_iterator currentPos;
    std::vector<std::vector<Edge>>::const_iterator edgesPos;
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
    const std::vector<std::vector<Edge>>& edges;
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
    const std::vector<std::optional<VertexData>>& vertices;

    void checkNextVertex();
  };

  using iterator = EdgeIterator;

  iterator begin() { return {edges, vertices}; }
  iterator end() { return {edges, vertices, edges.size()}; }

private:
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  const std::vector<std::vector<Edge>>& edges;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  const std::vector<std::optional<VertexData>>& vertices;
};

bool isPauli(const PiExpression& expr);
bool isClifford(const PiExpression& expr);
bool isProperClifford(const PiExpression& expr);

void roundToClifford(PiExpression& expr, fp tolerance);
} // namespace zx
