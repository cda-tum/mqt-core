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
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string_view>

namespace mqt::bench {

QAOA::QAOA(const QAOAOptions options)
    : options_(options), output_{.name = "result", .width = options_.qubits} {
  if (options_.qubits < 3 ||
      options_.qubits >
          static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
    throw std::invalid_argument(
        "QAOA qubits must be at least 3 and fit circuit dimensions");
  }
  if (options_.depth == 0 || options_.depth > QAOAOptions::MAX_DEPTH) {
    throw std::invalid_argument(
        "QAOA depth must be positive and fit circuit input dimensions");
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
