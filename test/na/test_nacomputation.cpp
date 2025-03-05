/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "Definitions.hpp"
#include "na/NAComputation.hpp"
#include "na/entities/Atom.hpp"
#include "na/entities/Location.hpp"
#include "na/entities/Zone.hpp"
#include "na/operations/GlobalCZOp.hpp"
#include "na/operations/GlobalRYOp.hpp"
#include "na/operations/LoadOp.hpp"
#include "na/operations/LocalRZOp.hpp"
#include "na/operations/MoveOp.hpp"
#include "na/operations/StoreOp.hpp"

#include <gtest/gtest.h>
#include <sstream>
#include <vector>

namespace na {
TEST(NAComputation, Atom) {
  const auto atom = Atom("atom");
  EXPECT_EQ(atom.getName(), "atom");
  std::stringstream ss;
  ss << atom;
  EXPECT_EQ(ss.str(), "atom");
}

TEST(NAComputation, Zone) {
  const auto zone = Zone("zone");
  EXPECT_EQ(zone.getName(), "zone");
  std::stringstream ss;
  ss << zone;
  EXPECT_EQ(ss.str(), "zone");
}

TEST(NAComputation, Location) {
  const Location loc{3, 4};
  EXPECT_EQ(loc, (Location{3, 4}));
  std::stringstream ss;
  ss << loc;
  EXPECT_EQ(ss.str(), "(3.000, 4.000)");
  EXPECT_DOUBLE_EQ((Location{0, 0}).getEuclideanDistance(loc), 5.0);
  EXPECT_DOUBLE_EQ((Location{0, 0}).getManhattanDistanceX(loc), 3);
  EXPECT_DOUBLE_EQ((Location{0, 0}).getManhattanDistanceY(loc), 4);
}

TEST(NAComputation, General) {
  auto qc = NAComputation();
  const auto& atom0 = qc.emplaceBackAtom("atom0");
  const auto& atom1 = qc.emplaceBackAtom("atom1");
  const auto& atom2 = qc.emplaceBackAtom("atom2");
  const auto& globalZone = qc.emplaceBackZone("global");
  qc.emplaceInitialLocation(atom0, 0, 0);
  qc.emplaceInitialLocation(atom1, 1, 0);
  qc.emplaceInitialLocation(atom2, 2, 0);
  qc.emplaceBack<LocalRZOp>(atom0, qc::PI_2);
  qc.emplaceBack<LocalRZOp>(std::vector{&atom1, &atom2}, qc::PI_2);
  qc.emplaceBack<GlobalRYOp>(globalZone, qc::PI_2);
  qc.emplaceBack<LoadOp>(std::vector{&atom0, &atom1},
                         std::vector{Location{0, 1}, Location{1, 1}});
  qc.emplaceBack<MoveOp>(std::vector{&atom0, &atom1},
                         std::vector{Location{4, 1}, Location{5, 1}});
  qc.emplaceBack<StoreOp>(std::vector{&atom0, &atom1},
                          std::vector{Location{4, 0}, Location{5, 0}});
  qc.emplaceBack(GlobalCZOp(globalZone));
  std::stringstream ss;
  ss << qc;
  EXPECT_EQ(ss.str(), "atom (0.000, 0.000) atom0\n"
                      "atom (1.000, 0.000) atom1\n"
                      "atom (2.000, 0.000) atom2\n"
                      "@+ rz 1.57080 atom0\n"
                      "@+ rz [\n"
                      "    1.57080 atom1\n"
                      "    1.57080 atom2\n"
                      "]\n"
                      "@+ ry 1.57080 global\n"
                      "@+ load [\n"
                      "    (0.000, 1.000) atom0\n"
                      "    (1.000, 1.000) atom1\n"
                      "]\n"
                      "@+ move [\n"
                      "    (4.000, 1.000) atom0\n"
                      "    (5.000, 1.000) atom1\n"
                      "]\n"
                      "@+ store [\n"
                      "    (4.000, 0.000) atom0\n"
                      "    (5.000, 0.000) atom1\n"
                      "]\n"
                      "@+ cz global\n");
}

TEST(NAComputation, EmptyPrint) {
  const NAComputation qc;
  std::stringstream ss;
  ss << qc;
  EXPECT_EQ(ss.str(), "");
}

TEST(NAComputation, ValidateAODConstraints) {
  auto qc = NAComputation();
  const auto& atom0 = qc.emplaceBackAtom("atom0");
  const auto& atom1 = qc.emplaceBackAtom("atom1");
  const auto& atom2 = qc.emplaceBackAtom("atom2");
  const auto& atom3 = qc.emplaceBackAtom("atom3");
  qc.emplaceInitialLocation(atom0, 0, 0);
  qc.emplaceInitialLocation(atom1, 1, 0);
  qc.emplaceInitialLocation(atom2, 0, 2);
  qc.emplaceInitialLocation(atom3, 1, 2);
  qc.emplaceBack<LoadOp>(std::vector{&atom0, &atom1},
                         std::vector{Location{0, 1}, Location{1, 1}});
  EXPECT_TRUE(qc.validate().first);
  // atom already loaded
  qc.emplaceBack<LoadOp>(atom0, Location{0, 1});
  EXPECT_FALSE(qc.validate().first);
  qc.clear();
  // atom not loaded
  qc.emplaceBack<MoveOp>(atom0, Location{0, 1});
  EXPECT_FALSE(qc.validate().first);
  qc.clear();
  // two atoms identical
  qc.emplaceBack<LoadOp>(std::vector{&atom0, &atom0});
  qc.emplaceBack<MoveOp>(std::vector{&atom0, &atom0},
                         std::vector{Location{0, 1}, Location{1, 1}});
  EXPECT_FALSE(qc.validate().first);
  qc.clear();
  // two end points identical
  qc.emplaceBack<LoadOp>(std::vector{&atom0, &atom1});
  qc.emplaceBack<MoveOp>(std::vector{&atom0, &atom1},
                         std::vector{Location{0, 1}, Location{0, 1}});
  EXPECT_FALSE(qc.validate().first);
  qc.clear();
  // columns not preserved
  qc.emplaceBack<LoadOp>(std::vector{&atom1, &atom3});
  qc.emplaceBack<MoveOp>(std::vector{&atom1, &atom3},
                         std::vector{Location{0, 1}, Location{2, 2}});
  EXPECT_FALSE(qc.validate().first);
  qc.clear();
  // rows not preserved
  qc.emplaceBack<LoadOp>(std::vector{&atom0, &atom1});
  qc.emplaceBack<MoveOp>(std::vector{&atom0, &atom1},
                         std::vector{Location{0, 1}, Location{1, -1}});
  EXPECT_FALSE(qc.validate().first);
  qc.clear();
  // column order not preserved
  qc.emplaceBack<LoadOp>(std::vector{&atom0, &atom3});
  qc.emplaceBack<MoveOp>(std::vector{&atom0, &atom3},
                         std::vector{Location{1, 1}, Location{0, 1}});
  EXPECT_FALSE(qc.validate().first);
  qc.clear();
  // row order not preserved
  qc.emplaceBack<LoadOp>(std::vector{&atom0, &atom3});
  qc.emplaceBack<MoveOp>(std::vector{&atom0, &atom3},
                         std::vector{Location{0, 1}, Location{2, 0}});
  EXPECT_FALSE(qc.validate().first);
  qc.clear();
  // column order not preserved
  qc.emplaceBack<LoadOp>(std::vector{&atom2, &atom1});
  qc.emplaceBack<MoveOp>(std::vector{&atom1, &atom2},
                         std::vector{Location{0, 1}, Location{1, 3}});
  EXPECT_FALSE(qc.validate().first);
  qc.clear();
  // row order not preserved
  qc.emplaceBack<LoadOp>(std::vector{&atom2, &atom1});
  qc.emplaceBack<MoveOp>(std::vector{&atom2, &atom1},
                         std::vector{Location{0, 1}, Location{2, 2}});
  EXPECT_FALSE(qc.validate().first);
  qc.clear();
  // two atoms identical
  qc.emplaceBack<LocalRZOp>(std::vector{&atom0, &atom0}, qc::PI_2);
  EXPECT_FALSE(qc.validate().first);
  qc.clear();
  // store unloaded atom
  qc.emplaceBack<StoreOp>(atom0);
  EXPECT_FALSE(qc.validate().first);
  qc.clear();
  qc.emplaceBack<LoadOp>(std::vector{&atom2, &atom1});
  // row order not preserved
  qc.emplaceBack<StoreOp>(std::vector{&atom2, &atom1},
                          std::vector{Location{0, 1}, Location{2, 2}});
  EXPECT_FALSE(qc.validate().first);
  qc.clear();
  qc.emplaceBack<LoadOp>(atom1);
  qc.emplaceBack<StoreOp>(atom1);
  qc.emplaceBack<StoreOp>(atom1);
  EXPECT_FALSE(qc.validate().first);
}

TEST(NAComputation, GetPositionOfAtomAfterOperation) {
  auto qc = NAComputation();
  const auto& atom0 = qc.emplaceBackAtom("atom0");
  qc.emplaceInitialLocation(atom0, 0, 0);
  qc.emplaceBack<LoadOp>(atom0);
  qc.emplaceBack<MoveOp>(atom0, Location{1, 1});
  qc.emplaceBack<StoreOp>(atom0);
  EXPECT_EQ(qc.getLocationOfAtomAfterOperation(atom0, qc[0]), (Location{0, 0}));
  EXPECT_EQ(qc.getLocationOfAtomAfterOperation(atom0, qc[2]), (Location{1, 1}));
}
} // namespace na
