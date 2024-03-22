#pragma once

#include "operations/OpType.hpp"

#include <cstddef>

namespace dd {
struct DDPackageConfig {
  // Note the order of parameters here must be the *same* as in the template
  // definition.
  static constexpr std::size_t UT_VEC_NBUCKET = 32768U;
  static constexpr std::size_t UT_VEC_INITIAL_ALLOCATION_SIZE = 2048U;
  static constexpr std::size_t UT_MAT_NBUCKET = 32768U;
  static constexpr std::size_t UT_MAT_INITIAL_ALLOCATION_SIZE = 2048U;
  static constexpr std::size_t CT_VEC_ADD_NBUCKET = 16384U;
  static constexpr std::size_t CT_MAT_ADD_NBUCKET = 16384U;
  static constexpr std::size_t CT_VEC_ADD_MAG_NBUCKET = 16384U;
  static constexpr std::size_t CT_MAT_ADD_MAG_NBUCKET = 16384U;
  static constexpr std::size_t CT_VEC_CONJ_NBUCKET = 4096U;
  static constexpr std::size_t CT_MAT_CONJ_TRANS_NBUCKET = 4096U;
  static constexpr std::size_t CT_MAT_VEC_MULT_NBUCKET = 16384U;
  static constexpr std::size_t CT_MAT_MAT_MULT_NBUCKET = 16384U;
  static constexpr std::size_t CT_VEC_KRON_NBUCKET = 4096U;
  static constexpr std::size_t CT_MAT_KRON_NBUCKET = 4096U;
  static constexpr std::size_t CT_VEC_INNER_PROD_NBUCKET = 4096U;
  static constexpr std::size_t CT_DM_NOISE_NBUCKET = 1U;
  static constexpr std::size_t UT_DM_NBUCKET = 1U;
  static constexpr std::size_t UT_DM_INITIAL_ALLOCATION_SIZE = 1U;
  static constexpr std::size_t CT_DM_DM_MULT_NBUCKET = 1U;
  static constexpr std::size_t CT_DM_ADD_NBUCKET = 1U;

  // The number of different quantum operations. I.e., the number of operations
  // defined in OpType.hpp. This parameter is required to initialize the
  // StochasticNoiseOperationTable.hpp
  static constexpr std::size_t STOCHASTIC_CACHE_OPS = 1;
};

struct StochasticNoiseSimulatorDDPackageConfig : public dd::DDPackageConfig {
  static constexpr std::size_t STOCHASTIC_CACHE_OPS = qc::OpType::OpCount;

  static constexpr std::size_t CT_VEC_ADD_MAG_NBUCKET = 1U;
  static constexpr std::size_t CT_MAT_ADD_MAG_NBUCKET = 1U;
  static constexpr std::size_t CT_VEC_CONJ_NBUCKET = 1U;
};

struct DensityMatrixSimulatorDDPackageConfig : public dd::DDPackageConfig {
  static constexpr std::size_t UT_DM_NBUCKET = 65536U;
  static constexpr std::size_t UT_DM_INITIAL_ALLOCATION_SIZE = 4096U;

  static constexpr std::size_t CT_DM_DM_MULT_NBUCKET = 16384U;
  static constexpr std::size_t CT_DM_ADD_NBUCKET = 16384U;
  static constexpr std::size_t CT_DM_NOISE_NBUCKET = 4096U;

  static constexpr std::size_t UT_MAT_NBUCKET = 16384U;
  static constexpr std::size_t CT_MAT_ADD_NBUCKET = 4096U;
  static constexpr std::size_t CT_VEC_ADD_NBUCKET = 4096U;
  static constexpr std::size_t CT_MAT_CONJ_TRANS_NBUCKET = 4096U;

  static constexpr std::size_t CT_MAT_MAT_MULT_NBUCKET = 1U;
  static constexpr std::size_t CT_MAT_VEC_MULT_NBUCKET = 1U;
  static constexpr std::size_t UT_VEC_NBUCKET = 1U;
  static constexpr std::size_t UT_VEC_INITIAL_ALLOCATION_SIZE = 1U;
  static constexpr std::size_t UT_MAT_INITIAL_ALLOCATION_SIZE = 1U;
  static constexpr std::size_t CT_VEC_KRON_NBUCKET = 1U;
  static constexpr std::size_t CT_MAT_KRON_NBUCKET = 1U;
  static constexpr std::size_t CT_VEC_INNER_PROD_NBUCKET = 1U;
  static constexpr std::size_t STOCHASTIC_CACHE_OPS = 1U;
  static constexpr std::size_t CT_VEC_ADD_MAG_NBUCKET = 1U;
  static constexpr std::size_t CT_MAT_ADD_MAG_NBUCKET = 1U;
  static constexpr std::size_t CT_VEC_CONJ_NBUCKET = 1U;
};
} // namespace dd
