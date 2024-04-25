//
// This file is part of the MQT QMAP library released under the MIT license.
// See README.md or go to https://github.com/cda-tum/mqt-core for more
// information.
//

#include "UndirectedGraph.hpp"

#include "gtest/gtest.h"

TEST(UndirectedGraph, Numbered) {
  qc::UndirectedGraph<int, int> g;
  g.addVertex(0);
  g.addEdge(1, 1, std::make_shared<int>(0));
  g.addEdge(1, 2, std::make_shared<int>(1));
  g.addEdge(2, 3, std::make_shared<int>(2));
  g.addEdge(3, 1, std::make_shared<int>(3));
  EXPECT_EQ(g.getNVertices(), 4);
  EXPECT_EQ(g.getNEdges(), 4);
  EXPECT_EQ(*g.getEdge(1, 2), 1);
  EXPECT_EQ(*g.getEdge(1, 1), 0);
  EXPECT_EQ(g.getDegree(2), 2);
  EXPECT_EQ(g.getDegree(1), 3);
  EXPECT_TRUE(g.isAdjacent(1, 2));
  EXPECT_FALSE(g.isAdjacent(1, 0));
  EXPECT_TRUE(g.isAdjacentEdge({1, 2}, {2, 3}));

  EXPECT_ANY_THROW(g.addVertex(1));
  EXPECT_ANY_THROW(std::ignore = g.getDegree(4));
  EXPECT_ANY_THROW(std::ignore = g.getEdge(0, 1));
}
