#include "Utils.hpp"

namespace zx {
    Vertices::VertexIterator::VertexIterator(
            std::vector<std::optional<VertexData>>& vertices, Vertex v):
        v(v),
        currentPos(vertices.begin()), vertices(vertices) {
        if ((size_t)v >= vertices.size()) {
            currentPos = vertices.end();
            this->v    = vertices.size();
        } else {
            currentPos = vertices.begin() + static_cast<int>(v);
            next_valid_vertex();
        }
    }
    // Prefix increment
    Vertices::VertexIterator Vertices::VertexIterator::operator++() {
        Vertices::VertexIterator it = *this;
        currentPos++;
        v++;
        next_valid_vertex();
        return it;
    }

    // Postfix increment
    const Vertices::VertexIterator Vertices::VertexIterator::operator++(int) {
        currentPos++;
        v++;
        next_valid_vertex();
        return *this;
    }

    bool operator==(const Vertices::VertexIterator& a,
                    const Vertices::VertexIterator& b) {
        return a.currentPos == b.currentPos;
    }
    bool operator!=(const Vertices::VertexIterator& a,
                    const Vertices::VertexIterator& b) {
        return !(a == b);
    }

    void Vertices::VertexIterator::next_valid_vertex() {
        while (currentPos != vertices.end() && !currentPos->has_value()) {
            v++;
            currentPos++;
        }
    }

    Edges::EdgeIterator::EdgeIterator(
            std::vector<std::vector<Edge>>&         edges,
            std::vector<std::optional<VertexData>>& vertices):
        v(0),
        currentPos(edges[0].begin()), edgesPos(edges.begin()), edges(edges), vertices(vertices) {
        if (!vertices.empty()) {
            while ((size_t)v < edges.size() && !vertices[v].has_value())
                v++;
            currentPos = edges[v].begin();
            edgesPos   = edges.begin() + static_cast<int>(v);
            checkNextVertex();
        } else {
            currentPos = edges.back().end();
            edgesPos   = edges.end();
            v          = edges.size();
        }
    }

    Edges::EdgeIterator::EdgeIterator(
            std::vector<std::vector<Edge>>&         edges,
            std::vector<std::optional<VertexData>>& vertices, Vertex v):
        v(v),
        edges(edges), vertices(vertices) {
        if ((size_t)v >= edges.size()) {
            currentPos = edges.back().end();
            edgesPos   = edges.end();
            this->v    = edges.size();
        } else {
            currentPos = edges[v].begin();
            edgesPos   = edges.begin() + static_cast<int>(v);
        }
    }

    // Prefix increment
    Edges::EdgeIterator Edges::EdgeIterator::operator++() {
        Edges::EdgeIterator it = *this;
        currentPos++;
        checkNextVertex();
        return it;
    }

    void Edges::EdgeIterator::checkNextVertex() {
        while (currentPos != edges[v].end() &&
               currentPos->to < v) // make sure to not iterate over an edge twice
            currentPos++;

        while (currentPos == edges[v].end() && (size_t)v < edges.size()) {
            v++;
            while ((size_t)v < edges.size() && !vertices[v].has_value())
                v++;

            if ((size_t)v == edges.size()) {
                currentPos = edges.back().end();
                edgesPos   = edges.end();
                v--;
                return;
            }
            currentPos = edges[v].begin();
            edgesPos   = edges.begin() + static_cast<int>(v);
            while (currentPos != edges[v].end() &&
                   currentPos->to < v) // make sure to not iterate over an edge twice
                currentPos++;
        }
    }
    // Postfix increment
    const Edges::EdgeIterator Edges::EdgeIterator::operator++(int) {
        currentPos++;
        checkNextVertex();
        return *this;
    }

    bool operator==(const Edges::EdgeIterator& a, const Edges::EdgeIterator& b) {
        return a.edgesPos == b.edgesPos && a.currentPos == b.currentPos;
    }
    bool operator!=(const Edges::EdgeIterator& a, const Edges::EdgeIterator& b) {
        return !(a == b);
    }
} // namespace zx
