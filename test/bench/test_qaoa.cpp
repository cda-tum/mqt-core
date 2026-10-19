/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "bench/Evaluation.hpp"
#include "bench/JSON.hpp"
#include "bench/QAOA.hpp"

#include "JSONTestUtils.hpp"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>

namespace mqt::bench {

using test::expectInvalidJSON;

TEST(QAOA, UsesDocumentedDefaults) {
  const QAOA benchmark{{.qubits = 5}};
  EXPECT_EQ(benchmark.options().qubits, 5);
  EXPECT_EQ(benchmark.options().gammas,
            (std::vector<double>{std::numbers::pi / 4.}));
  EXPECT_EQ(benchmark.options().betas,
            (std::vector<double>{std::numbers::pi / 8.}));
  EXPECT_EQ(benchmark.output(), (Output{"result", 5}));
}

TEST(QAOA, StoresAllPrescribedLayers) {
  const QAOA benchmark{{
      .qubits = 7,
      .gammas = {0.25, 1.5, 2. * std::numbers::pi},
      .betas = {0.125, 0., std::numbers::pi},
  }};
  EXPECT_EQ(benchmark.options().gammas,
            (std::vector<double>{0.25, 1.5, 2. * std::numbers::pi}));
  EXPECT_EQ(benchmark.options().betas,
            (std::vector<double>{0.125, 0., std::numbers::pi}));
}

TEST(QAOA, ValidatesTheConfiguredInstance) {
  EXPECT_THROW(static_cast<void>(QAOA{{.qubits = 2}}), std::invalid_argument);
  EXPECT_NO_THROW(static_cast<void>(QAOA{{.qubits = 3}}));
  const auto maximum = static_cast<size_t>(std::numeric_limits<int64_t>::max());
  EXPECT_NO_THROW(static_cast<void>(QAOA{{.qubits = maximum}}));
  if (maximum < std::numeric_limits<size_t>::max()) {
    EXPECT_THROW(static_cast<void>(QAOA{{.qubits = maximum + 1}}),
                 std::invalid_argument);
  }
  EXPECT_THROW(
      static_cast<void>(QAOA{{.qubits = 3, .gammas = {}, .betas = {}}}),
      std::invalid_argument);
  EXPECT_THROW(static_cast<void>(QAOA{{.qubits = 3, .gammas = {0., 1.}}}),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(QAOA{{.qubits = 3, .betas = {}}}),
               std::invalid_argument);
}

TEST(QAOA, RejectsAnglesOutsideThePaperDomain) {
  for (const double angle : {
           -0.1,
           7.,
           std::numeric_limits<double>::infinity(),
           std::numeric_limits<double>::quiet_NaN(),
       }) {
    EXPECT_THROW(static_cast<void>(QAOA{{.qubits = 3, .gammas = {angle}}}),
                 std::invalid_argument);
    EXPECT_THROW(static_cast<void>(QAOA{{.qubits = 3, .betas = {angle}}}),
                 std::invalid_argument);
  }
  EXPECT_THROW(static_cast<void>(QAOA{{.qubits = 3, .betas = {4.}}}),
               std::invalid_argument);
}

TEST(QAOA, CountsEveryCycleEdge) {
  const QAOA triangle{{.qubits = 3}};
  EXPECT_EQ(triangle.cutSize("000"), 0);
  EXPECT_EQ(triangle.cutSize("001"), 2);
  EXPECT_EQ(triangle.cutSize("010"), 2);
  const QAOA square{{.qubits = 4}};
  EXPECT_EQ(square.cutSize("0000"), 0);
  EXPECT_EQ(square.cutSize("0001"), 2);
  EXPECT_EQ(square.cutSize("0011"), 2);
  EXPECT_EQ(square.cutSize("0101"), 4);
  EXPECT_EQ(square.cutSize("1010"), 4);
  EXPECT_THROW(static_cast<void>(square.cutSize("010")), std::invalid_argument);
  EXPECT_THROW(static_cast<void>(square.cutSize("010x")),
               std::invalid_argument);
}

TEST(QAOA, ScoresCutsBeyondMachineWordWidths) {
  constexpr size_t qubits = 513;
  const QAOA benchmark{{.qubits = qubits}};
  auto outcome = std::string(qubits, '0');
  for (size_t index = 1; index < qubits; index += 2) {
    outcome[index] = '1';
  }
  EXPECT_EQ(benchmark.cutSize(outcome), 512);
  const auto evaluation = benchmark.evaluate({{outcome, 1}});
  EXPECT_DOUBLE_EQ(evaluation.meanCut, 512.);
  EXPECT_DOUBLE_EQ(evaluation.approximationRatio, 1.);
}

TEST(QAOA, EvaluatesCutsAgainstTheCycleOptimum) {
  const auto triangle = QAOA{{.qubits = 3}}.evaluate({
      {"000", 1},
      {"010", 3},
  });
  EXPECT_DOUBLE_EQ(triangle.meanCut, 1.5);
  EXPECT_DOUBLE_EQ(triangle.approximationRatio, 0.75);
  const auto square = QAOA{{.qubits = 4}}.evaluate({
      {"0000", 1},
      {"0101", 3},
  });
  EXPECT_DOUBLE_EQ(square.meanCut, 3.);
  EXPECT_DOUBLE_EQ(square.approximationRatio, 0.75);
}

TEST(QAOA, AvoidsOverflowWhileWeightingCuts) {
  const auto evaluation = QAOA{{.qubits = 4}}.evaluate({
      {"0101", std::numeric_limits<size_t>::max()},
  });
  EXPECT_DOUBLE_EQ(evaluation.meanCut, 4.);
  EXPECT_DOUBLE_EQ(evaluation.approximationRatio, 1.);
}

TEST(QAOA, KeepsRoundedOptimalScoresWithinTheirBounds) {
  const QAOA benchmark{{.qubits = 3}};
  const auto maximum = std::numeric_limits<size_t>::max();
  const auto evaluation = benchmark.evaluate({
      {"001", maximum / 3},
      {"010", maximum / 5},
      {"100", maximum / 5},
  });
  EXPECT_DOUBLE_EQ(evaluation.meanCut, 2.);
  EXPECT_LE(evaluation.meanCut, 2.);
  EXPECT_DOUBLE_EQ(evaluation.approximationRatio, 1.);
  EXPECT_LE(evaluation.approximationRatio, 1.);
  EXPECT_NO_THROW(static_cast<void>(evaluationToJSON(
      caseId(benchmark), (maximum / 3) + (2 * (maximum / 5)), evaluation)));
}

TEST(QAOA, ValidatesEveryCountKey) {
  const QAOA benchmark{{.qubits = 4}};
  EXPECT_THROW(static_cast<void>(benchmark.evaluate({})),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(benchmark.evaluate({{"0000", 0}})),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(benchmark.evaluate({
                   {"0000", 1},
                   {"00x0", 0},
               })),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(benchmark.evaluate({
                   {"0000", std::numeric_limits<size_t>::max()},
                   {"0101", 1},
               })),
               std::overflow_error);
}

TEST(QAOA, RoundTripsJSON) {
  const auto parsed = qaoaFromInstanceSpecificationJSON(
      R"({"schema_version":1,"benchmark":"qaoa","parameters":{"qubits":5,"gammas":[0.5,1.5],"betas":[0.25,0.75]}})");
  EXPECT_EQ(parsed.options().qubits, 5);
  EXPECT_EQ(parsed.options().gammas, (std::vector<double>{0.5, 1.5}));
  EXPECT_EQ(parsed.options().betas, (std::vector<double>{0.25, 0.75}));
  EXPECT_EQ(
      toInstanceSpecificationJSON(parsed),
      R"({"benchmark":"qaoa","parameters":{"betas":[0.25,0.75],"gammas":[0.5,1.5],"qubits":5},"schema_version":1})");
  const auto manifest = toManifestJSON(parsed);
  EXPECT_EQ(toManifestJSON(qaoaFromManifestJSON(manifest)), manifest);
  EXPECT_EQ(benchmarkIdFromManifestJSON(manifest), "qaoa");
  EXPECT_NE(manifest.find(R"("kind":"verification")"), std::string::npos);
  EXPECT_NE(manifest.find(R"("model":"qaoa_max_cut_cycle")"),
            std::string::npos);
  EXPECT_NE(caseId(parsed), caseId(QAOA{{.qubits = 5}}));
  auto changed = parsed.options();
  changed.gammas.back() = 1.;
  EXPECT_NE(caseId(parsed), caseId(QAOA(changed)));
  changed = parsed.options();
  changed.betas.back() = 1.;
  EXPECT_NE(caseId(parsed), caseId(QAOA(changed)));
  changed = parsed.options();
  changed.qubits = 7;
  EXPECT_NE(caseId(parsed), caseId(QAOA(changed)));
}

TEST(QAOA, ResolvesDefaultAnglesAndCanonicalZero) {
  const auto parsed = qaoaFromInstanceSpecificationJSON(
      R"({"schema_version":1,"benchmark":"qaoa","parameters":{"qubits":3}})");
  const QAOA defaults{{.qubits = 3}};
  EXPECT_EQ(toInstanceSpecificationJSON(parsed),
            toInstanceSpecificationJSON(defaults));
  EXPECT_EQ(caseId(parsed), caseId(defaults));
  const QAOA zero{{.qubits = 3, .gammas = {0.}, .betas = {0.}}};
  const QAOA negativeZero{{.qubits = 3, .gammas = {-0.}, .betas = {-0.}}};
  EXPECT_EQ(caseId(zero), caseId(negativeZero));
}

TEST(QAOA, DescribesJSONSchema) {
  const auto schema = describeBenchmarkJSON("qaoa");
  EXPECT_NE(schema.find(R"("minimum":3)"), std::string::npos);
  EXPECT_NE(schema.find(R"("maximum":9223372036854775807)"), std::string::npos);
  EXPECT_NE(schema.find(R"("minItems":1)"), std::string::npos);
  EXPECT_NE(schema.find(R"("required":["qubits"])"), std::string::npos);
}

TEST(QAOA, RejectsInvalidJSONParameters) {
  for (const auto* parameters : {
           R"({"qubits":2})",
           R"({"qubits":3,"gammas":[],"betas":[]})",
           R"({"qubits":3,"gammas":[1,2],"betas":[1]})",
           R"({"qubits":3,"gammas":[7]})",
           R"({"qubits":3,"betas":[4]})",
       }) {
    const auto specification =
        std::string(R"({"schema_version":1,"benchmark":"qaoa","parameters":)") +
        parameters + "}";
    EXPECT_THROW(
        static_cast<void>(qaoaFromInstanceSpecificationJSON(specification)),
        std::invalid_argument);
  }
  expectInvalidJSON(
      [] {
        static_cast<void>(qaoaFromInstanceSpecificationJSON(
            R"({"schema_version":1,"benchmark":"qaoa","parameters":{"qubits":3,"depth":2}})"));
      },
      "unknown key 'depth'");
  expectInvalidJSON(
      [] {
        static_cast<void>(qaoaFromInstanceSpecificationJSON(
            R"({"schema_version":1,"benchmark":"qaoa","parameters":{"qubits":3,"gammas":1}})"));
      },
      "$/parameters/gammas must be an array");
}

TEST(QAOA, ReportsTheParameterLocationOnce) {
  EXPECT_THAT(
      [] {
        static_cast<void>(qaoaFromInstanceSpecificationJSON(
            R"({"schema_version":1,"benchmark":"qaoa","parameters":{}})",
            "case.json"));
      },
      testing::ThrowsMessage<std::invalid_argument>(
          testing::StartsWith("case.json:$/parameters/qubits ")));
  EXPECT_THAT(
      [] {
        static_cast<void>(qaoaFromInstanceSpecificationJSON(
            R"({"schema_version":1,"benchmark":"qaoa","parameters":{"qubits":3,"betas":[0,"x"]}})",
            "case.json"));
      },
      testing::ThrowsMessage<std::invalid_argument>(
          testing::StartsWith("case.json:$/parameters/betas/1 ")));
}

TEST(QAOA, EvaluatesCountsFromJSON) {
  const auto evaluation =
      evaluateJSON(toManifestJSON(QAOA{{.qubits = 4}}),
                   R"({"schema_version":1,"counts":{"0000":1,"0101":3}})");
  EXPECT_NE(evaluation.find(R"("mean_cut":3.0)"), std::string::npos);
  EXPECT_NE(evaluation.find(R"("approximation_ratio":0.75)"),
            std::string::npos);
  EXPECT_EQ(evaluation.find("total_variation_distance"), std::string::npos);
  EXPECT_EQ(evaluation.find("squared_hellinger_fidelity"), std::string::npos);
}

TEST(QAOA, RejectsInvalidEvaluationMetrics) {
  const auto id = caseId(QAOA{{.qubits = 3}});
  EXPECT_THROW(static_cast<void>(evaluationToJSON(id, 0, QAOAEvaluation{})),
               std::invalid_argument);
  EXPECT_THROW(
      static_cast<void>(evaluationToJSON("invalid", 1, QAOAEvaluation{})),
      std::invalid_argument);
  EXPECT_THROW(static_cast<void>(
                   evaluationToJSON(id, 1, QAOAEvaluation{.meanCut = -1.})),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(evaluationToJSON(
                   id, 1, QAOAEvaluation{.approximationRatio = 1.1})),
               std::invalid_argument);
  EXPECT_THROW(static_cast<void>(evaluationToJSON(
                   id, 1,
                   QAOAEvaluation{
                       .meanCut = std::numeric_limits<double>::quiet_NaN(),
                   })),
               std::invalid_argument);
}

} // namespace mqt::bench
