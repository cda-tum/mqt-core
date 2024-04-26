#include "datastructures/UndirectedGraph.hpp"

#include "gtest/gtest.h"

namespace qc {
TEST(UndirectedGraph, Numbered) {
  UndirectedGraph<int, int> g;
  g.addVertex(0);
  g.addEdge(1, 1, 0);
  g.addEdge(1, 2, 1);
  g.addEdge(2, 3, 2);
  g.addEdge(3, 1, 3);
  EXPECT_EQ(g.getNVertices(), 4);
  EXPECT_EQ(g.getNEdges(), 4);
  EXPECT_EQ(g.getEdge(1, 2), 1);
  EXPECT_EQ(g.getEdge(1, 1), 0);
  EXPECT_EQ(g.getDegree(2), 2);
  EXPECT_EQ(g.getDegree(1), 3);
  EXPECT_TRUE(g.isAdjacent(1, 2));
  EXPECT_FALSE(g.isAdjacent(1, 0));
  EXPECT_TRUE(g.isAdjacentEdge({1, 2}, {2, 3}));

  EXPECT_ANY_THROW(g.addVertex(1));
  EXPECT_ANY_THROW(std::ignore = g.getDegree(4));
  EXPECT_ANY_THROW(std::ignore = g.getEdge(0, 1));
}
} // namespace qc
