/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "bench/QAOA.hpp"

#include "bench/Evaluation.hpp"

#include "EvaluationUtils.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace mqt::bench {

QAOA::QAOA(QAOAOptions options)
    : options_(std::move(options)),
      output_{.name = "result", .width = options_.qubits} {
  if (options_.qubits < 3 ||
      options_.qubits >
          static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
    throw std::invalid_argument(
        "QAOA qubits must be at least 3 and fit circuit dimensions");
  }
  if (options_.gammas.empty() ||
      options_.gammas.size() != options_.betas.size()) {
    throw std::invalid_argument(
        "QAOA gamma and beta vectors must have the same nonzero length");
  }
  for (auto& gamma : options_.gammas) {
    if (!std::isfinite(gamma) || gamma < 0. || gamma > 2. * std::numbers::pi) {
      throw std::invalid_argument(
          "QAOA gamma angles must be finite and in [0, 2*pi]");
    }
    if (gamma == 0.) {
      gamma = 0.;
    }
  }
  for (auto& beta : options_.betas) {
    if (!std::isfinite(beta) || beta < 0. || beta > std::numbers::pi) {
      throw std::invalid_argument(
          "QAOA beta angles must be finite and in [0, pi]");
    }
    if (beta == 0.) {
      beta = 0.;
    }
  }
}

const QAOAOptions& QAOA::options() const noexcept { return options_; }

const Output& QAOA::output() const noexcept { return output_; }

size_t QAOA::cutSize(const std::string_view outcome) const {
  detail::validateOutcome(outcome, output_.width);
  auto cut = size_t{outcome.front() != outcome.back()};
  for (size_t index = 1; index < outcome.size(); ++index) {
    if (outcome[index - 1] != outcome[index]) {
      ++cut;
    }
  }
  return cut;
}

QAOAEvaluation QAOA::evaluate(const Counts& counts) const {
  const auto shots = detail::countShots(counts);
  // Extended precision preserves weighted scores without integer overflow.
  // NOLINTBEGIN(google-runtime-float)
  long double weightedCut = 0.L;
  for (const auto& [outcome, count] : counts) {
    weightedCut += static_cast<long double>(cutSize(outcome)) *
                   static_cast<long double>(count);
  }
  const auto optimum = options_.qubits - (options_.qubits % 2);
  const auto mean = std::min(weightedCut / static_cast<long double>(shots),
                             static_cast<long double>(optimum));
  return {
      .meanCut = static_cast<double>(mean),
      .approximationRatio =
          static_cast<double>(mean / static_cast<long double>(optimum)),
  };
  // NOLINTEND(google-runtime-float)
}

} // namespace mqt::bench
