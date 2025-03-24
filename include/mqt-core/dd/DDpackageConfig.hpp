/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include "ir/operations/OpType.hpp"

#include <cstddef>

namespace dd {
struct DDPackageConfig {
  std::size_t UT_VEC_NBUCKET = 32768U;
  std::size_t UT_VEC_INITIAL_ALLOCATION_SIZE = 2048U;
  std::size_t UT_MAT_NBUCKET = 32768U;
  std::size_t UT_MAT_INITIAL_ALLOCATION_SIZE = 2048U;
  std::size_t CT_VEC_ADD_NBUCKET = 16384U;
  std::size_t CT_MAT_ADD_NBUCKET = 16384U;
  std::size_t CT_VEC_ADD_MAG_NBUCKET = 16384U;
  std::size_t CT_MAT_ADD_MAG_NBUCKET = 16384U;
  std::size_t CT_VEC_CONJ_NBUCKET = 4096U;
  std::size_t CT_MAT_CONJ_TRANS_NBUCKET = 4096U;
  std::size_t CT_MAT_VEC_MULT_NBUCKET = 16384U;
  std::size_t CT_MAT_MAT_MULT_NBUCKET = 16384U;
  std::size_t CT_VEC_KRON_NBUCKET = 4096U;
  std::size_t CT_MAT_KRON_NBUCKET = 4096U;
  std::size_t CT_DM_TRACE_NBUCKET = 1U;
  std::size_t CT_MAT_TRACE_NBUCKET = 4096U;
  std::size_t CT_VEC_INNER_PROD_NBUCKET = 4096U;
  std::size_t CT_DM_NOISE_NBUCKET = 1U;
  std::size_t UT_DM_NBUCKET = 1U;
  std::size_t UT_DM_INITIAL_ALLOCATION_SIZE = 1U;
  std::size_t CT_DM_DM_MULT_NBUCKET = 1U;
  std::size_t CT_DM_ADD_NBUCKET = 1U;

  // The number of different quantum operations. i.e., the number of operations
  // defined in OpType.hpp. This parameter is required to initialize the
  // StochasticNoiseOperationTable.hpp
  std::size_t STOCHASTIC_CACHE_OPS = 1;
};

constexpr auto StochasticNoiseSimulatorDDPackageConfig = DDPackageConfig{
    .STOCHASTIC_CACHE_OPS = qc::OpType::OpTypeEnd,
    .CT_VEC_ADD_MAG_NBUCKET = 1U,
    .CT_MAT_ADD_MAG_NBUCKET = 1U,
    .CT_VEC_CONJ_NBUCKET = 1U,
};

constexpr auto DensityMatrixSimulatorDDPackageConfig = DDPackageConfig{
    .UT_DM_NBUCKET = 65536U,
    .UT_DM_INITIAL_ALLOCATION_SIZE = 4096U,

    .CT_DM_DM_MULT_NBUCKET = 16384U,
    .CT_DM_ADD_NBUCKET = 16384U,
    .CT_DM_NOISE_NBUCKET = 4096U,

    .UT_MAT_NBUCKET = 16384U,
    .CT_MAT_ADD_NBUCKET = 4096U,
    .CT_VEC_ADD_NBUCKET = 4096U,
    .CT_MAT_CONJ_TRANS_NBUCKET = 4096U,

    .CT_MAT_MAT_MULT_NBUCKET = 1U,
    .CT_MAT_VEC_MULT_NBUCKET = 1U,
    .UT_VEC_NBUCKET = 1U,
    .UT_VEC_INITIAL_ALLOCATION_SIZE = 1U,
    .UT_MAT_INITIAL_ALLOCATION_SIZE = 1U,
    .CT_VEC_KRON_NBUCKET = 1U,
    .CT_MAT_KRON_NBUCKET = 1U,
    .CT_DM_TRACE_NBUCKET = 4096U,
    .CT_MAT_TRACE_NBUCKET = 1U,
    .CT_VEC_INNER_PROD_NBUCKET = 1U,
    .STOCHASTIC_CACHE_OPS = 1U,
    .CT_VEC_ADD_MAG_NBUCKET = 1U,
    .CT_MAT_ADD_MAG_NBUCKET = 1U,
    .CT_VEC_CONJ_NBUCKET = 1U,
};

constexpr auto UnitarySimulatorDDPackageConfig = DDPackageConfig{
    // unitary simulation requires more resources for matrices.
    .UT_MAT_NBUCKET = 65'536U,
    .CT_MAT_ADD_NBUCKET = 65'536U,
    .CT_MAT_MAT_MULT_NBUCKET = 65'536U,

    // unitary simulation does not need any vector nodes
    .UT_VEC_NBUCKET = 1U,
    .UT_VEC_INITIAL_ALLOCATION_SIZE = 1U,

    // unitary simulation needs no vector functionality
    .CT_VEC_ADD_NBUCKET = 1U,
    .CT_VEC_ADD_MAG_NBUCKET = 1U,
    .CT_MAT_ADD_MAG_NBUCKET = 1U,
    .CT_VEC_CONJ_NBUCKET = 1U,
    .CT_MAT_CONJ_TRANS_NBUCKET = 1U,
    .CT_MAT_VEC_MULT_NBUCKET = 1U,
    .CT_VEC_KRON_NBUCKET = 1U,
    .CT_MAT_KRON_NBUCKET = 1U,
    .CT_MAT_TRACE_NBUCKET = 1U,
    .CT_VEC_INNER_PROD_NBUCKET = 1U,
};
} // namespace dd
