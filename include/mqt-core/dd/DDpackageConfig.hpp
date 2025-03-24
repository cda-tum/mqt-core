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

constexpr auto StochasticNoiseSimulatorDDPackageConfig = []() {
  DDPackageConfig config{};
  config.STOCHASTIC_CACHE_OPS = qc::OpType::OpTypeEnd;
  config.CT_VEC_ADD_MAG_NBUCKET = 1U;
  config.CT_MAT_ADD_MAG_NBUCKET = 1U;
  config.CT_VEC_CONJ_NBUCKET = 1U;
  return config;
}();

constexpr auto DensityMatrixSimulatorDDPackageConfig = []() {
  DDPackageConfig config{};
  config.UT_DM_NBUCKET = 65536U;
  config.UT_DM_INITIAL_ALLOCATION_SIZE = 4096U;
  config.CT_DM_DM_MULT_NBUCKET = 16384U;
  config.CT_DM_ADD_NBUCKET = 16384U;
  config.CT_DM_NOISE_NBUCKET = 4096U;
  config.UT_MAT_NBUCKET = 16384U;
  config.CT_MAT_ADD_NBUCKET = 4096U;
  config.CT_VEC_ADD_NBUCKET = 4096U;
  config.CT_MAT_CONJ_TRANS_NBUCKET = 4096U;
  config.CT_MAT_MAT_MULT_NBUCKET = 1U;
  config.CT_MAT_VEC_MULT_NBUCKET = 1U;
  config.UT_VEC_NBUCKET = 1U;
  config.UT_VEC_INITIAL_ALLOCATION_SIZE = 1U;
  config.UT_MAT_INITIAL_ALLOCATION_SIZE = 1U;
  config.CT_VEC_KRON_NBUCKET = 1U;
  config.CT_MAT_KRON_NBUCKET = 1U;
  config.CT_DM_TRACE_NBUCKET = 4096U;
  config.CT_MAT_TRACE_NBUCKET = 1U;
  config.CT_VEC_INNER_PROD_NBUCKET = 1U;
  config.STOCHASTIC_CACHE_OPS = 1U;
  config.CT_VEC_ADD_MAG_NBUCKET = 1U;
  config.CT_MAT_ADD_MAG_NBUCKET = 1U;
  config.CT_VEC_CONJ_NBUCKET = 1U;
  return config;
}();

constexpr auto UnitarySimulatorDDPackageConfig = []() {
  DDPackageConfig config{};
  config.UT_MAT_NBUCKET = 65'536U;
  config.CT_MAT_ADD_NBUCKET = 65'536U;
  config.CT_MAT_MAT_MULT_NBUCKET = 65'536U;
  config.UT_VEC_NBUCKET = 1U;
  config.UT_VEC_INITIAL_ALLOCATION_SIZE = 1U;
  config.CT_VEC_ADD_NBUCKET = 1U;
  config.CT_VEC_ADD_MAG_NBUCKET = 1U;
  config.CT_MAT_ADD_MAG_NBUCKET = 1U;
  config.CT_VEC_CONJ_NBUCKET = 1U;
  config.CT_MAT_CONJ_TRANS_NBUCKET = 1U;
  config.CT_MAT_VEC_MULT_NBUCKET = 1U;
  config.CT_VEC_KRON_NBUCKET = 1U;
  config.CT_MAT_KRON_NBUCKET = 1U;
  config.CT_MAT_TRACE_NBUCKET = 1U;
  config.CT_VEC_INNER_PROD_NBUCKET = 1U;
  return config;
}();
} // namespace dd
