//
// This file is part of the MQT QMAP library released under the MIT license.
// See README.md or go to https://github.com/cda-tum/mqt-core for more
// information.
//

#include "DisjointSet.hpp"

#include <gtest/gtest.h>

TEST(DisjointSet, FindSet) {
  std::vector elements = {1, 2, 3};
  qc::DisjointSet<int> ds(elements.begin(), elements.end());
  EXPECT_EQ(ds.findSet(1), ds.findSet(1));
  EXPECT_NE(ds.findSet(1), ds.findSet(2));
}

TEST(DisjointSet, UnionSet) {
  std::vector elements = {1, 2, 3};
  qc::DisjointSet<int> ds(elements.begin(), elements.end());
  ds.unionSet(1, 2);
  EXPECT_EQ(ds.findSet(1), ds.findSet(2));
  EXPECT_NE(ds.findSet(1), ds.findSet(3));
  EXPECT_NE(ds.findSet(2), ds.findSet(3));
  ds.unionSet(1, 3);
  EXPECT_EQ(ds.findSet(1), ds.findSet(2));
  EXPECT_EQ(ds.findSet(1), ds.findSet(3));
  EXPECT_EQ(ds.findSet(2), ds.findSet(3));
}
