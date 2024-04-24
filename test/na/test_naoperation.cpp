//
// This file is part of the MQT QMAP library released under the MIT license.
// See README.md or go to https://github.com/cda-tum/mqt-core for more
// information.
//

#include "na/operations/NAGlobalOperation.hpp"
#include "na/operations/NALocalOperation.hpp"
#include "na/operations/NAOperation.hpp"
#include "na/operations/NAShuttlingOperation.hpp"

#include "gtest/gtest.h"

TEST(NAOperation, ShuttlingOperation) {
  const na::NAShuttlingOperation shuttling(
      na::ShuttleType::LOAD,
      std::vector{std::make_shared<na::Point>(0, 0),
                  std::make_shared<na::Point>(1, 0)},
      std::vector{std::make_shared<na::Point>(0, 1),
                  std::make_shared<na::Point>(1, 1)});
  EXPECT_TRUE(shuttling.isShuttlingOperation());
  EXPECT_FALSE(shuttling.isLocalOperation());
  EXPECT_EQ(shuttling.getStart()[1]->x, 1);
  EXPECT_EQ(shuttling.getEnd()[0]->x, 0);
  EXPECT_ANY_THROW(na::NAShuttlingOperation(
      na::ShuttleType::STORE, std::vector{std::make_shared<na::Point>(0, 0)},
      std::vector{std::make_shared<na::Point>(0, 1),
                  std::make_shared<na::Point>(1, 1)}));
}

TEST(NAOperation, GlobalOperation) {
  const na::NAGlobalOperation op(na::OpType{qc::RY, 0},
                                 std::vector<qc::fp>{qc::PI_2});
  EXPECT_FALSE(op.isShuttlingOperation());
  EXPECT_FALSE(op.isLocalOperation());
  EXPECT_TRUE(op.isGlobalOperation());
  EXPECT_DOUBLE_EQ(op.getParams()[0], qc::PI_2);
  EXPECT_ANY_THROW(na::NAGlobalOperation(na::OpType{qc::ECR, 0}));
}

TEST(NAOperation, LocalOperation) {
  const na::NALocalOperation op(na::OpType{qc::RY, 0},
                                std::vector<qc::fp>{qc::PI_2},
                                std::make_shared<na::Point>(0, 0));
  EXPECT_FALSE(op.isShuttlingOperation());
  EXPECT_FALSE(op.isGlobalOperation());
  EXPECT_TRUE(op.isLocalOperation());
  EXPECT_EQ(op.getType(), (na::OpType{qc::RY, 0}));
  EXPECT_DOUBLE_EQ(op.getParams()[0], qc::PI_2);
  EXPECT_EQ(op.getPositions()[0]->x, 0);
  EXPECT_ANY_THROW(na::NALocalOperation(na::OpType{qc::ECR, 0},
                                        std::make_shared<na::Point>(0, 0)));
  EXPECT_ANY_THROW(na::NALocalOperation(na::OpType{qc::RY, 1},
                                        std::make_shared<na::Point>(0, 0)));
}