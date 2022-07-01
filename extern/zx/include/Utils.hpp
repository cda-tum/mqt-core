#pragma once

#include "Definitions.hpp"
#include "Expression.hpp"

#include <iterator>
#include <optional>
#include <vector>

namespace zx {

    struct Edge {
        Vertex   to;
        EdgeType type;

        Edge() = default;
        Edge(Vertex to, EdgeType type):
            to(to), type(type){};
        void toggle() {
            type = (type == EdgeType::Simple) ? EdgeType::Hadamard : EdgeType::Simple;
        }
    };

    struct VertexData {
        Col        col;
        Qubit      qubit;
        Expression phase;
        VertexType type;
    };

    class Vertices {
    public:
        explicit Vertices(
                std::vector<std::optional<VertexData>>& vertices):
            vertices(vertices){};

        class VertexIterator {
        public:
            using iterator_category = std::forward_iterator_tag;
            using difference_type   = std::int32_t;
            using value_type        = std::pair<Vertex, VertexData&>;
            using pointer           = value_type*;
            using reference         = value_type&;

            explicit VertexIterator(std::vector<std::optional<VertexData>>& vertices):
                currentPos(vertices.begin()), vertices(vertices) {
                next_valid_vertex();
            }
            VertexIterator(std::vector<std::optional<VertexData>>& vertices,
                           Vertex                                  v);

            value_type operator*() const { return {v, currentPos->value()}; }
            // pointer operator->() { return ptr; }

            // Prefix increment
            VertexIterator operator++();

            // Postfix increment
            const VertexIterator operator++(int);

            friend bool operator==(const VertexIterator& a,
                                   const VertexIterator& b);
            friend bool operator!=(const VertexIterator& a,
                                   const VertexIterator& b);

        private:
            Vertex                                           v = 0;
            std::vector<std::optional<VertexData>>::iterator currentPos;
            std::vector<std::optional<VertexData>>&          vertices;

            void next_valid_vertex();
        };

        using iterator = VertexIterator;

        iterator begin() { return VertexIterator(vertices); }
        iterator end() { return {vertices, vertices.size()}; }

    private:
        std::vector<std::optional<VertexData>>& vertices;
    };

    class Edges {
    public:
        Edges(std::vector<std::vector<Edge>>&         edges,
              std::vector<std::optional<VertexData>>& vertices):
            edges(edges),
            vertices(vertices){};

        class EdgeIterator {
        public:
            using iterator_category = std::forward_iterator_tag;
            using difference_type   = std::int32_t;
            using value_type        = std::pair<Vertex, Vertex>;
            using pointer           = value_type*;
            using reference         = value_type&;

            EdgeIterator(std::vector<std::vector<Edge>>&         edges,
                         std::vector<std::optional<VertexData>>& vertices);

            EdgeIterator(std::vector<std::vector<Edge>>&         edges,
                         std::vector<std::optional<VertexData>>& vertices, Vertex v);

            value_type operator*() const { return {v, currentPos->to}; }
            // pointer operator->() { return ptr; }

            // Prefix increment
            EdgeIterator operator++();

            // Postfix increment
            const EdgeIterator operator++(int);

            friend bool operator==(const EdgeIterator& a, const EdgeIterator& b);
            friend bool operator!=(const EdgeIterator& a, const EdgeIterator& b);

        private:
            Vertex                                   v;
            std::vector<Edge>::iterator              currentPos;
            std::vector<std::vector<Edge>>::iterator edgesPos;
            std::vector<std::vector<Edge>>&          edges;
            std::vector<std::optional<VertexData>>&  vertices;

            void checkNextVertex();
        };

        using iterator = EdgeIterator;

        iterator begin() { return {edges, vertices}; }
        iterator end() { return {edges, vertices, edges.size()}; }

    private:
        std::vector<std::vector<Edge>>&         edges;
        std::vector<std::optional<VertexData>>& vertices;
    };
} // namespace zx
