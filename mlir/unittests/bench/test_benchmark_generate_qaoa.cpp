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
#include "dd/DDDefinitions.hpp"
#include "dd/Package.hpp"
#include "dd/StateGeneration.hpp"
#include "mqt/Dialect/MQT/IR/MQTDialect.h"
#include "mqt/Dialect/QCO/Utils/DDFunctionality.h"
#include "mqt/bench/Generate.h"

#include "TestUtils.h"

#include "gtest/gtest.h"

#include "llvm/Support/LogicalResult.h"

#include <cmath>
#include <complex>
#include <cstddef>
#include <utility>
#include <vector>

namespace mqt::bench {

// Evaluate the paper's cost eigenvalues, independently of RZZ decomposition.
static dd::CVec qaoaState(const QAOAOptions& options) {
  const auto size = size_t{1} << options.qubits;
  dd::CVec state(size, 1. / std::sqrt(static_cast<double>(size)));
  for (size_t layer = 0; layer < options.gammas.size(); ++layer) {
    for (size_t index = 0; index < size; ++index) {
      size_t cut = 0;
      for (size_t bit = 0; bit < options.qubits; ++bit) {
        const auto neighbor = (bit + 1) % options.qubits;
        if (((index >> bit) & 1U) != ((index >> neighbor) & 1U)) {
          ++cut;
        }
      }
      state[index] *=
          std::polar(1., -options.gammas[layer] * static_cast<double>(cut));
    }
    const auto diagonal = std::cos(options.betas[layer]);
    const std::complex<double> offDiagonal{0., -std::sin(options.betas[layer])};
    for (size_t bit = 0; bit < options.qubits; ++bit) {
      const auto mask = size_t{1} << bit;
      for (size_t index = 0; index < size; ++index) {
        if ((index & mask) != 0) {
          continue;
        }
        const auto zero = state[index];
        const auto one = state[index | mask];
        state[index] = (diagonal * zero) + (offDiagonal * one);
        state[index | mask] = (offDiagonal * zero) + (diagonal * one);
      }
    }
  }
  return state;
}

TEST(GenerateProgramTest, QAOAMatchesCostAndMixerEvolution) {
  for (const size_t qubits : {3U, 4U, 5U, 7U}) {
    for (const auto& options : {
             QAOAOptions{.qubits = qubits},
             QAOAOptions{
                 .qubits = qubits,
                 .gammas = {0.31, 1.17},
                 .betas = {0.23, 0.49},
             },
             QAOAOptions{
                 .qubits = qubits,
                 .gammas = {0., 0.73, 1.29},
                 .betas = {0.41, 0., 0.27},
             },
         }) {
      SCOPED_TRACE(qubits);
      SCOPED_TRACE(options.gammas.size());
      auto program = test::generateQCO(QAOA{options});
      ASSERT_TRUE(mlir::succeeded(program));
      dd::Package package(0);
      auto state = mlir::qco::simulateStatevector(
          mlir::mqt::getEntryPoint(program->module()), package);
      ASSERT_TRUE(mlir::succeeded(state));
      ASSERT_EQ(package.qubits(), qubits);
      auto expected = dd::makeStateFromVector(qaoaState(options), package);
      EXPECT_NEAR(package.fidelity(*state, expected), 1., 1e-10);
      package.decRef(expected);
      package.decRef(*state);
    }
  }
}

TEST(GenerateProgramTest, SamplesQAOACutQuality) {
  for (const size_t qubits : {3U, 4U, 5U}) {
    const QAOA benchmark{{.qubits = qubits}};
    auto program = test::generateQCO(benchmark);
    ASSERT_TRUE(mlir::succeeded(program));
    auto counts = mlir::qco::sample(mlir::mqt::getEntryPoint(program->module()),
                                    16'384, 17);
    ASSERT_TRUE(mlir::succeeded(counts));
    // The triangle has a different light cone than the longer cycles.
    const auto cutFraction = qubits == 3 ? 5. / 8. : 3. / 4.;
    const auto evaluation = benchmark.evaluate(*counts);
    EXPECT_NEAR(evaluation.meanCut / static_cast<double>(qubits), cutFraction,
                0.03);
    EXPECT_NEAR(evaluation.approximationRatio,
                evaluation.meanCut / static_cast<double>(qubits - qubits % 2),
                1e-12);
  }
}

TEST(GenerateProgramTest, SerializesWideQAOAWithCompactLoops) {
  for (const size_t layers : {2U, 128U}) {
    SCOPED_TRACE(layers);
    QAOAOptions options{
        .qubits = 1'024,
        .gammas = std::vector<double>(layers, 0.31),
        .betas = std::vector<double>(layers, 0.23),
    };
    options.gammas.back() = 1.17;
    options.betas.back() = 0.49;
    auto program = generate(QAOA{options});
    ASSERT_TRUE(mlir::succeeded(program));
    // Generation must not expand a gate list per vertex or layer.
    EXPECT_LT(test::countOperations(program->module()), 150U);
    test::expectJeffRoundTrip(std::move(*program));
  }
}

} // namespace mqt::bench
