//
// This file is part of the MQT QMAP library released under the MIT license.
// See README.md or go to https://github.com/cda-tum/mqt-core for more
// information.
//

#include "DirectedGraph.hpp"

#include "gtest/gtest.h"

TEST(DirectedGraph, Numbered) {
  qc::DirectedGraph<int, int*> g;
  int a = 0;
  int b = 1;
  int c = 2;
  int d = 3;
  g.addVertex(0);
  g.addEdge(1, 1, &a);
  g.addEdge(1, 2, &b);
  g.addEdge(2, 3, &c);
  g.addEdge(3, 1, &d);
  //      ┌────┐
  //  0   └──> 1 ———> 2 ———> 3
  //           ^             |
  //           └─────────────┘
  EXPECT_EQ(g.getNVertices(), 4);
  EXPECT_EQ(g.getNEdges(), 4);
  EXPECT_EQ(*g.getEdge(1, 2), 1);
  EXPECT_EQ(*g.getEdge(1, 1), 0);
  EXPECT_EQ(g.getInDegree(2), 1);
  EXPECT_EQ(g.getInDegree(1), 2);
  EXPECT_EQ(g.getOutDegree(2), 1);
  EXPECT_EQ(g.getOutDegree(1), 2);
  EXPECT_TRUE(g.isEdge(1, 2));
  EXPECT_FALSE(g.isEdge(2, 1));
  EXPECT_FALSE(g.isEdge(1, 0));
}
