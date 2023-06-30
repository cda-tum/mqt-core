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
