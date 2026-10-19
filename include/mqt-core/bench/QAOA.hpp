/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include "bench/Evaluation.hpp"
#include "bench/mqt_core_bench_export.h"

#include <cstddef>
#include <numbers>
#include <string_view>
#include <vector>

namespace mqt::bench {

/// Parameters for QAOA MaxCut on a cycle graph.
struct QAOAOptions {
  /// Number of vertices and qubits, at least three and within signed 64-bit
  /// circuit dimensions.
  size_t qubits;
  /// Cost angles in radians, each finite and in \f$[0, 2\pi]\f$.
  ///
  /// Both angle vectors must have the same nonzero length, which is the depth.
  std::vector<double> gammas = {std::numbers::pi / 4.};
  /// Mixer angles in radians, each finite and in \f$[0, \pi]\f$.
  std::vector<double> betas = {std::numbers::pi / 8.};
};

/// Classical scores of sampled QAOA cuts.
struct QAOAEvaluation {
  /// Mean number of edges whose endpoints have different measured bits.
  double meanCut = 0.;
  /// Mean cut size divided by the cycle graph's maximum cut size.
  double approximationRatio = 0.;
};

/// A validated QAOA MaxCut benchmark.
class MQT_CORE_BENCH_EXPORT QAOA final {
public:
  explicit QAOA(QAOAOptions options);

  [[nodiscard]] const QAOAOptions& options() const noexcept;
  [[nodiscard]] const Output& output() const noexcept;
  /// Return the cut size of a big-endian logical outcome on the cycle.
  [[nodiscard]] size_t cutSize(std::string_view outcome) const;
  /// Score sampled cuts without computing an ideal output distribution.
  [[nodiscard]] QAOAEvaluation evaluate(const Counts& counts) const;

private:
  QAOAOptions options_;
  Output output_;
};

} // namespace mqt::bench
