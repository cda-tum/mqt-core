#include "Graph.hpp"

void Graph::add_edge(int32_t from, int32_t to, EdgeType type) {
  adj_lst[from].push_back({to, type});
  adj_lst[to].push_back({from, type});
}

// void Graph::add_vertex() {}

void Graph::add_vertex(const VertexData& data) {
  vertices.push_back(data);
  nvertices += 1;
}


