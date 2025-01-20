/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once
#include "dd/DDpackageConfig.hpp"
#include "dd/Node.hpp"

namespace qc {
using VectorDD = dd::vEdge;
using MatrixDD = dd::mEdge;
using DensityMatrixDD = dd::dEdge;
} // namespace qc

namespace dd {
template <class Config = DDPackageConfig> class Package;
} // namespace dd
