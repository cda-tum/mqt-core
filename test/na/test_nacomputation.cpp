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
#include "na/operations/LocalCZOp.hpp"
#include "na/operations/LocalHOp.hpp"
#include "na/operations/LocalRXOp.hpp"
#include "na/operations/LocalRYOp.hpp"
#include "na/operations/LocalRZOp.hpp"
#include "na/operations/LocalXOp.hpp"
#include "na/operations/LocalYOp.hpp"
#include "na/operations/LocalZOp.hpp"
#include "na/operations/MoveOp.hpp"
#include "na/operations/StoreOp.hpp"

#include <gtest/gtest.h>
#include <sstream>
#include <stdexcept>
#include <tuple>
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
TEST(NAComputation, ZonesExtent) {
  const auto zone = Zone("zone", {0, 0, 2, 2});
  EXPECT_TRUE(zone.contains({1., 1.}));
  EXPECT_FALSE(zone.contains({1., 3.}));
  EXPECT_THROW(std::ignore = Zone("zone").contains({0., 0.}),
               std::runtime_error);
}

TEST(NAComputation, Location) {
  constexpr Location loc{3, 4};
  EXPECT_EQ(loc, (Location{3, 4}));
  std::stringstream ss;
  ss << loc;
  EXPECT_EQ(ss.str(), "(3.000, 4.000)");
  EXPECT_DOUBLE_EQ((Location{0, 0}).getEuclideanDistance(loc), 5.0);
  EXPECT_DOUBLE_EQ((Location{0, 0}).getManhattanDistanceX(loc), 3);
  EXPECT_DOUBLE_EQ((Location{0, 0}).getManhattanDistanceY(loc), 4);
}

TEST(NAComputation, LocalRXOp) {
  const Atom atom("atom");
  const LocalRXOp op(atom, 0.0);
  EXPECT_EQ(op.toString(), "@+ rx 0.00000 atom");
}

TEST(NAComputation, LocalRYOp) {
  const Atom atom("atom");
  const LocalRYOp op(atom, 0.0);
  EXPECT_EQ(op.toString(), "@+ ry 0.00000 atom");
}

TEST(NAComputation, LocalRZOp) {
  const Atom atom("atom");
  const LocalRZOp op(atom, 0.0);
  EXPECT_EQ(op.toString(), "@+ rz 0.00000 atom");
}

TEST(NAComputation, LocalXOp) {
  const Atom atom("atom");
  const LocalXOp op(atom);
  EXPECT_EQ(op.toString(), "@+ x atom");
}

TEST(NAComputation, LocalYOp) {
  const Atom atom("atom");
  const LocalYOp op(atom);
  EXPECT_EQ(op.toString(), "@+ y atom");
}

TEST(NAComputation, LocalZOp) {
  const Atom atom("atom");
  const LocalZOp op(atom);
  EXPECT_EQ(op.toString(), "@+ z atom");
}

TEST(NAComputation, LocalHOp) {
  const Atom atom("atom");
  const LocalHOp op(atom);
  EXPECT_EQ(op.toString(), "@+ h atom");
}

TEST(NAComputation, LocalCZOp) {
  const Atom atom1("atom1");
  const Atom atom2("atom2");
  const LocalCZOp op(atom1, atom2);
  EXPECT_EQ(op.toString(), "@+ cz {atom1, atom2}");
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

class NAComputationValidateAODConstraints : public ::testing::Test {
protected:
  NAComputation qc;
  const Atom* atom0 = nullptr;
  const Atom* atom1 = nullptr;
  const Atom* atom2 = nullptr;
  const Atom* atom3 = nullptr;

  auto SetUp() -> void override {
    atom0 = &qc.emplaceBackAtom("atom0");
    atom1 = &qc.emplaceBackAtom("atom1");
    atom2 = &qc.emplaceBackAtom("atom2");
    atom3 = &qc.emplaceBackAtom("atom3");
    qc.emplaceInitialLocation(*atom0, 0, 0);
    qc.emplaceInitialLocation(*atom1, 1, 0);
    qc.emplaceInitialLocation(*atom2, 0, 2);
    qc.emplaceInitialLocation(*atom3, 1, 2);
  }
};

TEST_F(NAComputationValidateAODConstraints, AtomAlreadyLoaded) {
  qc.emplaceBack<LoadOp>(std::vector{atom0, atom1},
                         std::vector{Location{0, 1}, Location{1, 1}});
  EXPECT_TRUE(qc.validate().first);
  qc.emplaceBack<LoadOp>(*atom0, Location{0, 1});
  EXPECT_FALSE(qc.validate().first);
}
TEST_F(NAComputationValidateAODConstraints, AtomNotLoaded) {
  qc.emplaceBack<MoveOp>(*atom0, Location{0, 1});
  EXPECT_FALSE(qc.validate().first);
}
TEST_F(NAComputationValidateAODConstraints, DuplicateAtomsInShuttle) {
  qc.emplaceBack<LoadOp>(std::vector{atom0, atom0});
  qc.emplaceBack<MoveOp>(std::vector{atom0, atom0},
                         std::vector{Location{0, 1}, Location{1, 1}});
  EXPECT_FALSE(qc.validate().first);
}
TEST_F(NAComputationValidateAODConstraints, DuplicateEndPoints) {
  qc.emplaceBack<LoadOp>(std::vector{atom0, atom1});
  qc.emplaceBack<MoveOp>(std::vector{atom0, atom1},
                         std::vector{Location{0, 1}, Location{0, 1}});
  EXPECT_FALSE(qc.validate().first);
}
TEST_F(NAComputationValidateAODConstraints, ColumnPreserving1) {
  qc.emplaceBack<LoadOp>(std::vector{atom1, atom3});
  qc.emplaceBack<MoveOp>(std::vector{atom1, atom3},
                         std::vector{Location{0, 1}, Location{2, 2}});
  EXPECT_FALSE(qc.validate().first);
}
TEST_F(NAComputationValidateAODConstraints, RowPreserving1) {
  qc.emplaceBack<LoadOp>(std::vector{atom0, atom1});
  qc.emplaceBack<MoveOp>(std::vector{atom0, atom1},
                         std::vector{Location{0, 1}, Location{1, -1}});
  EXPECT_FALSE(qc.validate().first);
}
TEST_F(NAComputationValidateAODConstraints, ColumnPreserving2) {
  qc.emplaceBack<LoadOp>(std::vector{atom0, atom3});
  qc.emplaceBack<MoveOp>(std::vector{atom0, atom3},
                         std::vector{Location{1, 1}, Location{0, 1}});
  EXPECT_FALSE(qc.validate().first);
}
TEST_F(NAComputationValidateAODConstraints, RowPreserving2) {
  // row order not preserved
  qc.emplaceBack<LoadOp>(std::vector{atom0, atom3});
  qc.emplaceBack<MoveOp>(std::vector{atom0, atom3},
                         std::vector{Location{0, 1}, Location{2, 0}});
  EXPECT_FALSE(qc.validate().first);
}
TEST_F(NAComputationValidateAODConstraints, ColumnPreserving3) {
  qc.emplaceBack<LoadOp>(std::vector{atom2, atom1});
  qc.emplaceBack<MoveOp>(std::vector{atom1, atom2},
                         std::vector{Location{0, 1}, Location{1, 3}});
  EXPECT_FALSE(qc.validate().first);
}
TEST_F(NAComputationValidateAODConstraints, RowPreserving3) {
  qc.emplaceBack<LoadOp>(std::vector{atom2, atom1});
  qc.emplaceBack<MoveOp>(std::vector{atom2, atom1},
                         std::vector{Location{0, 1}, Location{2, 2}});
  EXPECT_FALSE(qc.validate().first);
}
TEST_F(NAComputationValidateAODConstraints, DuplicateAtomsInRz) {
  qc.emplaceBack<LocalRZOp>(std::vector{atom0, atom0}, qc::PI_2);
  EXPECT_FALSE(qc.validate().first);
}
TEST_F(NAComputationValidateAODConstraints, DuplicateAtoms) {
  // store unloaded atom
  qc.emplaceBack<StoreOp>(*atom0);
  EXPECT_FALSE(qc.validate().first);
}
TEST_F(NAComputationValidateAODConstraints, RowPreserving4) {
  qc.emplaceBack<LoadOp>(std::vector{atom2, atom1});
  qc.emplaceBack<StoreOp>(std::vector{atom2, atom1},
                          std::vector{Location{0, 1}, Location{2, 2}});
  EXPECT_FALSE(qc.validate().first);
}
TEST_F(NAComputationValidateAODConstraints, StoreStoredAtom) {
  qc.emplaceBack<LoadOp>(*atom1);
  qc.emplaceBack<StoreOp>(*atom1);
  qc.emplaceBack<StoreOp>(*atom1);
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

TEST(NAComputation, ConvertToLocalGates) {
  auto qc = NAComputation();
  const auto& atom0 = qc.emplaceBackAtom("atom0");
  const auto& atom1 = qc.emplaceBackAtom("atom1");
  const auto& atom2 = qc.emplaceBackAtom("atom2");
  const auto& atom3 = qc.emplaceBackAtom("atom3");
  qc.emplaceInitialLocation(atom0, 1, 0);
  qc.emplaceInitialLocation(atom1, 2, 0);
  qc.emplaceInitialLocation(atom2, 1, 4);
  qc.emplaceInitialLocation(atom3, 2, 4);
  const auto& global = qc.emplaceBackZone("global", Zone::Extent{0, 0, 3, 5});
  const auto& czZone = qc.emplaceBackZone("zone_cz0", Zone::Extent{0, 0, 3, 2});
  qc.emplaceBack<GlobalRYOp>(global, 0.1);
  qc.emplaceBack<GlobalCZOp>(czZone);
  EXPECT_EQ(qc.toString(), "atom (1.000, 0.000) atom0\n"
                           "atom (1.000, 4.000) atom2\n"
                           "atom (2.000, 0.000) atom1\n"
                           "atom (2.000, 4.000) atom3\n"
                           "@+ ry 0.10000 global\n"
                           "@+ cz zone_cz0\n");
  qc.convertToLocalGates(3.);
  EXPECT_EQ(qc.toString(), "atom (1.000, 0.000) atom0\n"
                           "atom (1.000, 4.000) atom2\n"
                           "atom (2.000, 0.000) atom1\n"
                           "atom (2.000, 4.000) atom3\n"
                           "@+ ry [\n"
                           "    0.10000 atom0\n"
                           "    0.10000 atom2\n"
                           "    0.10000 atom1\n"
                           "    0.10000 atom3\n"
                           "]\n"
                           "@+ cz {atom0, atom1}\n");
}
} // namespace na
