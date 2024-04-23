//
// This file is part of the MQT QMAP library released under the MIT license.
// See README.md or go to https://github.com/cda-tum/mqt-core for more
// information.
//

#include "DirectedAcyclicGraph.hpp"

#include "gtest/gtest.h"

TEST(DirectedAcyclicGraph, Reachable) {
  qc::DirectedAcyclicGraph<int> g;
  g.addVertex(0);
  g.addEdge(0, 1);
  g.addEdge(1, 2);
  g.addEdge(2, 3);
  EXPECT_ANY_THROW(g.addEdge(3, 1));
  //  0 ────> 1 ———> 2 ———> 3
  //          ^             |
  //          └──────X──────┘
  EXPECT_TRUE(g.isReachable(1, 3));
  EXPECT_FALSE(g.isReachable(3, 1));
}

TEST(DirectedAcyclicGraph, TopologicalOrder) {
  qc::DirectedAcyclicGraph<int> g;
  g.addVertex(0);
  g.addEdge(0, 1);
  g.addEdge(1, 2);
  g.addEdge(2, 3);
  //  0 ───> 1 ———> 2 ———> 3
  const auto& actual = g.orderTopologically();
  const std::vector<int> expected = {0, 1, 2, 3};
  EXPECT_EQ(actual, expected);
}