//
// This file is part of the MQT QMAP library released under the MIT license.
// See README.md or go to https://github.com/cda-tum/mqt-core for more
// information.
//

#include "DirectedGraph.hpp"

#include "gtest/gtest.h"

TEST(DirectedGraph, Numbered) {
  qc::DirectedGraph<int> g;
  g.addVertex(0);
  g.addEdge(1, 1);
  g.addEdge(1, 2);
  g.addEdge(2, 3);
  g.addEdge(3, 1);
  //      ┌────┐
  //  0   └──> 1 ———> 2 ———> 3
  //           ^             |
  //           └─────────────┘
  EXPECT_EQ(g.getNVertices(), 4);
  EXPECT_EQ(g.getNEdges(), 4);
  EXPECT_EQ(g.getInDegree(2), 1);
  EXPECT_EQ(g.getInDegree(1), 2);
  EXPECT_EQ(g.getOutDegree(2), 1);
  EXPECT_EQ(g.getOutDegree(1), 2);
  EXPECT_TRUE(g.isEdge(1, 2));
  EXPECT_FALSE(g.isEdge(2, 1));
  EXPECT_FALSE(g.isEdge(1, 0));

  EXPECT_ANY_THROW(g.addVertex(1));
  EXPECT_ANY_THROW(std::ignore = g.getInDegree(4));
  EXPECT_ANY_THROW(std::ignore = g.getOutDegree(4));
}
