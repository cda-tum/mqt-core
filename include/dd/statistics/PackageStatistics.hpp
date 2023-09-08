#pragma once

#include "dd/Package.hpp"

#include <nlohmann/json.hpp>

namespace dd {

template <class Config = DDPackageConfig>
[[nodiscard]] static nlohmann::json getStatistics(Package<Config>* package) {
  nlohmann::json j;

  auto& vector = j["vector"];
  vector["unique_table"] = package->vUniqueTable.getStatsJson();
  vector["memory_manager"] = package->vMemoryManager.getStats().json();

  auto& matrix = j["matrix"];
  matrix["unique_table"] = package->mUniqueTable.getStatsJson();
  matrix["memory_manager"] = package->mMemoryManager.getStats().json();

  auto& densityMatrix = j["density_matrix"];
  densityMatrix["unique_table"] = package->dUniqueTable.getStatsJson();
  densityMatrix["memory_manager"] = package->dMemoryManager.getStats().json();

  auto& realNumbers = j["real_numbers"];
  realNumbers["unique_table"] = package->cUniqueTable.getStats().json();
  realNumbers["memory_manager"] = package->cMemoryManager.getStats().json();
  realNumbers["cache_manager"] = package->cCacheManager.getStats().json();

  auto& computeTables = j["compute_tables"];
  computeTables["vector_add"] = package->vectorAdd.getStats().json();
  computeTables["matrix_add"] = package->matrixAdd.getStats().json();
  computeTables["density_matrix_add"] = package->densityAdd.getStats().json();
  computeTables["matrix_transpose"] =
      package->matrixTranspose.getStats().json();
  computeTables["matrix_conjugate_transpose"] =
      package->conjugateMatrixTranspose.getStats().json();
  computeTables["matrix_vector_mult"] =
      package->matrixVectorMultiplication.getStats().json();
  computeTables["matrix_matrix_mult"] =
      package->matrixMatrixMultiplication.getStats().json();
  computeTables["density_density_mult"] =
      package->densityDensityMultiplication.getStats().json();
  computeTables["vector_kronecker"] =
      package->vectorKronecker.getStats().json();
  computeTables["matrix_kronecker"] =
      package->matrixKronecker.getStats().json();
  computeTables["vector_inner_product"] =
      package->vectorInnerProduct.getStats().json();
  computeTables["stochastic_noise_operations"] =
      package->stochasticNoiseOperationCache.getStats().json();
  computeTables["density_noise_operations"] =
      package->densityNoise.getStats().json();

  return j;
}

template <class Config = DDPackageConfig>
[[nodiscard]] static std::string getStatisticsString(Package<Config>* package) {
  return getStatistics(package).dump(2U);
}

template <class Config = DDPackageConfig>
static void printStatistics(Package<Config>* package,
                            std::ostream& os = std::cout) {
  os << getStatisticsString(package);
}

} // namespace dd
