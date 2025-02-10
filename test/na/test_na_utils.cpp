/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "ir/QuantumComputation.hpp"
#include "na/NAUtils.hpp"
#include "qasm3/Importer.hpp"

#include <gtest/gtest.h>
#include <string>

namespace na {
TEST(NADefinitions, IsGlobal) {
  const std::string testfile = "OPENQASM 3.0;\n"
                               "include \"stdgates.inc\";\n"
                               "qubit[3] q;\n"
                               "rz(pi/4) q[0];\n"
                               "ry(pi/2) q;\n";
  const auto qc = qasm3::Importer::imports(testfile);
  EXPECT_EQ(qc.getHighestLogicalQubitIndex(), 2);
  EXPECT_FALSE(isGlobal(*qc.at(0), 3));
  EXPECT_TRUE(isGlobal(*qc.at(1), 3));
}
} // namespace na
