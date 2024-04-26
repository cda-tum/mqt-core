#include "QuantumComputation.hpp"
#include "na/NADefinitions.hpp"
#include "operations/OpType.hpp"

#include "gtest/gtest.h"
#include <sstream>
#include <string>
#include <unordered_map>

TEST(NADefinitions, Point) {
  const na::Point p(-1, 2);
  EXPECT_EQ(p.x, -1);
  EXPECT_EQ(p.y, 2);
  EXPECT_EQ(p.length(), 2);
  EXPECT_EQ(p.toString(), "(-1, 2)");
  std::stringstream ss;
  ss << p;
  EXPECT_EQ(ss.str(), "(-1, 2)");
  EXPECT_EQ(p, na::Point(-1, 2));
  EXPECT_FALSE(p == na::Point(1, 2));
  EXPECT_EQ(p - na::Point(1, 2), na::Point(-2, 0));
  EXPECT_EQ(na::Point(1, 2) + p, na::Point(0, 4));
}

TEST(NADefinitions, OpType) {
  const na::OpType t{qc::OpType::X, 1};
  EXPECT_EQ(t.type, qc::OpType::X);
  EXPECT_EQ(t.nControls, 1);
  EXPECT_EQ(t.toString(), "cx");
  std::stringstream ss;
  ss << t;
  EXPECT_EQ(ss.str(), "cx");
  EXPECT_EQ(t, (na::OpType{qc::OpType::X, 1}));
  EXPECT_FALSE(t == (na::OpType{qc::OpType::X, 2}));
}

TEST(NADefinitions, IsGlobal) {
  const std::string testfile = "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit[3] q;\n"
                               "rz(pi/4) q[0];\n"
                               "ry(pi/2) q;\n";
  const auto qc = qc::QuantumComputation::fromQASM(testfile);
  EXPECT_EQ(qc.getHighestLogicalQubitIndex(), 2);
  EXPECT_FALSE(na::isGlobal(*qc.at(0), 3));
  EXPECT_TRUE(na::isGlobal(*qc.at(1), 3));
}

TEST(NADefinitions, OpTypeHash) {
  std::unordered_map<na::OpType, int> map;
  map[na::OpType{qc::OpType::X, 1}] = 1;
  map[na::OpType{qc::OpType::X, 2}] = 2;
  map[na::OpType{qc::OpType::Y, 1}] = 3;
  map[na::OpType{qc::OpType::Y, 2}] = 4;
  EXPECT_EQ((map[na::OpType{qc::OpType::X, 1}]), 1);
  EXPECT_EQ((map[na::OpType{qc::OpType::X, 2}]), 2);
  EXPECT_EQ((map[na::OpType{qc::OpType::Y, 1}]), 3);
  EXPECT_EQ((map[na::OpType{qc::OpType::Y, 2}]), 4);
}
