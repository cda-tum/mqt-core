#include "Utils.hpp"

namespace zx {
    Vertices::VertexIterator::VertexIterator(
            std::vector<std::optional<VertexData>>& vertices, Vertex v):
        v(v),
        current_pos(vertices.begin()), vertices(vertices) {
        if ((size_t)v >= vertices.size()) {
            current_pos = vertices.end();
            v           = vertices.size();
        } else {
            current_pos = vertices.begin() + v;
            next_valid_vertex();
        }
    }
    // Prefix increment
    Vertices::VertexIterator Vertices::VertexIterator::operator++() {
        Vertices::VertexIterator it = *this;
        current_pos++;
        v++;
        next_valid_vertex();
        return it;
    }

    // Postfix increment
    Vertices::VertexIterator Vertices::VertexIterator::operator++(int) {
        current_pos++;
        v++;
        next_valid_vertex();
        return *this;
    }

    bool operator==(const Vertices::VertexIterator& a,
                    const Vertices::VertexIterator& b) {
        return a.current_pos == b.current_pos;
    }
    bool operator!=(const Vertices::VertexIterator& a,
                    const Vertices::VertexIterator& b) {
        return !(a == b);
    }

    void Vertices::VertexIterator::next_valid_vertex() {
        while (current_pos != vertices.end() && !current_pos->has_value()) {
            v++;
            current_pos++;
        }
    }

    Edges::EdgeIterator::EdgeIterator(
            std::vector<std::vector<Edge>>&         edges,
            std::vector<std::optional<VertexData>>& vertices):
        v(0),
        current_pos(edges[0].begin()), edges(edges), vertices(vertices) {
        if (vertices.size() != 0) {
            while ((size_t)v < edges.size() && !vertices[v].has_value())
                v++;
            current_pos = edges[v].begin();
            check_next_vertex();
        } else {
            current_pos = edges.back().end();
            v           = edges.size();
        }
    }

    Edges::EdgeIterator::EdgeIterator(
            std::vector<std::vector<Edge>>&         edges,
            std::vector<std::optional<VertexData>>& vertices, Vertex v):
        v(v),
        edges(edges), vertices(vertices) {
        if ((size_t)v >= edges.size()) {
            current_pos = edges.back().end();
            v           = edges.size();
        } else {
            current_pos = edges[v].begin();
        }
    }

    // Prefix increment
    Edges::EdgeIterator Edges::EdgeIterator::operator++() {
        Edges::EdgeIterator it = *this;
        current_pos++;
        check_next_vertex();
        return it;
    }

    void Edges::EdgeIterator::check_next_vertex() {
        while (current_pos != edges[v].end() &&
               current_pos->to < v) // make sure to not iterate over an edge twice
            current_pos++;

        while (current_pos == edges[v].end() && (size_t)v < edges.size()) {
            v++;
            while ((size_t)v < edges.size() && !vertices[v].has_value())
                v++;

            if ((size_t)v == edges.size()) {
                current_pos = edges.back().end();
                v--;
                return;
            }
            current_pos = edges[v].begin();
            while (current_pos != edges[v].end() &&
                   current_pos->to < v) // make sure to not iterate over an edge twice
                current_pos++;
        }
    }
    // Postfix increment
    Edges::EdgeIterator Edges::EdgeIterator::operator++(int) {
        current_pos++;
        check_next_vertex();
        return *this;
    }

    bool operator==(const Edges::EdgeIterator& a, const Edges::EdgeIterator& b) {
        return a.current_pos == b.current_pos;
    }
    bool operator!=(const Edges::EdgeIterator& a, const Edges::EdgeIterator& b) {
        return !(a == b);
    }
} // namespace zx
