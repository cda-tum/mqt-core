/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "na/entities/Atom.hpp"
#include "na/operations/LocalRZOp.hpp"

#include <gtest/gtest.h>

namespace na {
TEST(NAOps, Clone) {
  const auto* atom0 = new Atom("atom0");
  const auto op = std::make_unique<LocalRZOp>(1.57080, atom0);
  auto clone = op->clone();
  EXPECT_NE(clone.get(), op.get());
  EXPECT_EQ(op->toString(), clone->toString());
}
} // namespace na
