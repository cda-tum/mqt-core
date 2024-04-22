//
// This file is part of the MQT QMAP library released under the MIT license.
// See README.md or go to https://github.com/cda-tum/mqt-core for more information.
//

#include "na/NAComputation.hpp"
#include "na/NADefinitions.hpp"
#include "na/operations/NAGlobalOperation.hpp"
#include "na/operations/NALocalOperation.hpp"
#include "na/operations/NAShuttlingOperation.hpp"
#include "operations/OpType.hpp"

#include "gtest/gtest.h"
#include <sstream>
#include <string>

TEST(NAComputation, General) {
  auto qc = na::NAComputation();
  qc.emplaceInitialPosition(std::make_shared<na::Point>(0, 0));
  qc.emplaceInitialPosition(std::make_shared<na::Point>(1, 0));
  qc.emplaceInitialPosition(std::make_shared<na::Point>(2, 0));
  qc.emplaceBack(std::make_unique<na::NALocalOperation>(
      na::OpType{qc::RZ, 0}, std::vector<qc::fp>{qc::PI_2},
      std::make_shared<na::Point>(0, 0)));
  qc.emplaceBack(std::make_unique<na::NAGlobalOperation>(
      na::OpType{qc::RY, 0}, std::vector<qc::fp>{qc::PI_2}));
  qc.emplaceBack(std::make_unique<na::NAShuttlingOperation>(
      na::ShuttleType::LOAD,
      std::vector{std::make_shared<na::Point>(0, 0),
                  std::make_shared<na::Point>(1, 0)},
      std::vector{std::make_shared<na::Point>(0, 1),
                  std::make_shared<na::Point>(1, 1)}));
  qc.emplaceBack(std::make_unique<na::NAShuttlingOperation>(
      na::ShuttleType::MOVE,
      std::vector{std::make_shared<na::Point>(0, 1),
                  std::make_shared<na::Point>(1, 1)},
      std::vector{std::make_shared<na::Point>(4, 1),
                  std::make_shared<na::Point>(5, 1)}));
  qc.emplaceBack(std::make_unique<na::NAShuttlingOperation>(
      na::ShuttleType::STORE,
      std::vector{std::make_shared<na::Point>(4, 1),
                  std::make_shared<na::Point>(5, 1)},
      std::vector{std::make_shared<na::Point>(4, 0),
                  std::make_shared<na::Point>(5, 0)}));
  std::stringstream ss;
  ss << qc;
  EXPECT_EQ(ss.str(), "init at (0, 0), (1, 0), (2, 0);\n"
                      "rz(1.5708) at (0, 0);\n"
                      "ry(1.5708);\n"
                      "load (0, 0), (1, 0) to (0, 1), (1, 1);\n"
                      "move (0, 1), (1, 1) to (4, 1), (5, 1);\n"
                      "store (4, 1), (5, 1) to (4, 0), (5, 0);\n");
}
