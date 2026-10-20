/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Compiler/Programs.h"
#include "mqt/Compiler/Target.h"
#include "mqt/Compiler/TargetEnvironment.h"

#include "gtest/gtest.h"
#include "nlohmann/json.hpp"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/LogicalResult.h"

#include <cstddef>
#include <string>
#include <utility>
#include <variant>

using namespace mlir;

static TargetEnvironment makeIQMEnvironment() {
  using Capability = CompilerTarget::OperationCapability;
  using Site = CompilerTarget::Site;
  const SmallVector capabilities{
      llvm::cantFail(Capability::create("r", 1, 2)),
      llvm::cantFail(Capability::create("cz", 2, 0)),
      llvm::cantFail(Capability::create("measure", 1, 0)),
  };
  auto target = llvm::cantFail(CompilerTarget::create(
      "sparse-iqm",
      {
          llvm::cantFail(Site::create(5, "readout-b")),
          llvm::cantFail(Site::create(17, "readout-a")),
          llvm::cantFail(Site::create(42, "spare")),
      },
      CompilerTarget::Connectivity::allToAll(),
      CompilerTarget::NativeOperations::fromOperations(capabilities)));
  auto payload = llvm::cantFail(
      PayloadSpecification::create({.id = "iqm-json"}, {}, true));
  return {target, std::move(payload)};
}

TEST(IQMJSONPipelineTest, CompilesInputFormatsForAnExplicitTarget) {
  constexpr llvm::StringLiteral source = R"qasm(
    OPENQASM 3.0;
    qubit[2] q;
    bit[2] result;
    r(1.25, -0.75) q[1];
    cz q[1], q[0];
    result[1] = measure q[1];
    result[0] = measure q[0];
  )qasm";
  auto qc = QCProgram::fromOpenQASMString(source.str());
  ASSERT_TRUE(succeeded(qc));
  EXPECT_TRUE(failed(
      runDefaultPipeline(CompilerInput(qc->copy()), ProgramFormat::IQMJSON)));
  auto qco = qc->copy().intoQCO();
  ASSERT_TRUE(succeeded(qco));
  auto jeff = qco->copy().intoJeff();
  ASSERT_TRUE(succeeded(jeff));
  SmallVector<CompilerInput> inputs;
  inputs.emplace_back(std::move(*qc));
  inputs.emplace_back(std::move(*qco));
  inputs.emplace_back(std::move(*jeff));
  inputs.emplace_back(OpenQASMProgram(source.str()));
  for (auto& input : inputs) {
    SCOPED_TRACE(input.index());
    auto output = runDefaultPipeline(std::move(input), makeIQMEnvironment());
    ASSERT_TRUE(succeeded(output));
    ASSERT_TRUE(std::holds_alternative<IQMJSONProgram>(*output));
    const auto& exported = std::get<IQMJSONProgram>(*output);
    const auto instructions =
        nlohmann::json::parse(exported.source()).at("instructions");
    EXPECT_EQ(instructions.front().at("name"), "prx");
    EXPECT_EQ(instructions.back().at("locus").size(), 2U);
    EXPECT_EQ(instructions.front().at("locus")[0],
              instructions.back().at("locus")[1]);
  }
}

TEST(IQMJSONPipelineTest, UnrollsLoopsAndSynthesizesIQMBasis) {
  auto output = runDefaultPipeline(CompilerInput(OpenQASMProgram(R"qasm(
    OPENQASM 3.0;
    include "stdgates.inc";
    qubit[2] q;
    bit[2] result;
    for int i in [0:1] { h q[i]; }
    cx q[0], q[1];
    result = measure q;
  )qasm")),
                                   makeIQMEnvironment());
  ASSERT_TRUE(succeeded(output));
  const auto instructions =
      nlohmann::json::parse(std::get<IQMJSONProgram>(*output).source())
          .at("instructions");
  size_t rotations = 0;
  size_t entanglers = 0;
  for (const auto& instruction : instructions) {
    const auto name = instruction.at("name").get<std::string>();
    EXPECT_TRUE(name == "prx" || name == "cz" || name == "measure") << name;
    rotations += static_cast<size_t>(name == "prx");
    entanglers += static_cast<size_t>(name == "cz");
  }
  EXPECT_GT(rotations, 0U);
  EXPECT_GT(entanglers, 0U);
  EXPECT_EQ(instructions.back().at("name"), "measure");
  EXPECT_EQ(instructions.back().at("locus").size(), 2U);
}

TEST(IQMJSONPipelineTest, RejectsReconstructedOutputsAndNonterminalGates) {
  for (const auto* source : {
           "qubit q; bit[2] result = 0; result[0] = measure q;",
           "qubit q; bit untouched = 0; bit result; result = measure q;",
           R"(qubit[2] q; bit result;
              result = measure q[0]; result = measure q[1];)",
           "qubit q; bit result; result = measure q; result = 0;",
           "qubit q; bit result; int value = 1; result = measure q;",
           R"(qubit[2] q; bit[2] result; h q[0]; h q[1];
              result[0] = measure q[0]; r(1, 0) q[0]; cz q[0], q[1];
              result[1] = measure q[1];)",
       }) {
    SCOPED_TRACE(source);
    auto input =
        QCProgram::fromOpenQASMString(std::string("OPENQASM 3.0; ") + source);
    ASSERT_TRUE(succeeded(input));
    ScopedDiagnosticHandler diagnostics(input->module().getContext(),
                                        [](Diagnostic&) { return success(); });
    EXPECT_TRUE(failed(runDefaultPipeline(CompilerInput(input->copy()),
                                          makeIQMEnvironment())));
  }
}

TEST(IQMJSONPipelineTest, DelaysIndependentReadoutUntilRoutedGatesFinish) {
  using Capability = CompilerTarget::OperationCapability;
  const auto target = llvm::cantFail(CompilerTarget::create(
      {
          llvm::cantFail(CompilerTarget::Site::create(10, "QB7")),
          llvm::cantFail(CompilerTarget::Site::create(30, "QB2")),
          llvm::cantFail(CompilerTarget::Site::create(90, "QB11")),
      },
      CompilerTarget::Connectivity::fromCouplings({{10, 30}, {30, 90}}),
      CompilerTarget::NativeOperations::fromOperations({
          llvm::cantFail(Capability::create("r", 1, 2)),
          llvm::cantFail(Capability::create("cz", 2, 0)),
          llvm::cantFail(Capability::create("measure", 1, 0)),
      })));
  const TargetEnvironment environment(
      target, makeIQMEnvironment().payloadSpecification());
  auto output = runDefaultPipeline(CompilerInput(OpenQASMProgram(R"qasm(
    OPENQASM 3.0;
    include "stdgates.inc";
    qubit[3] q;
    bit[2] zeta;
    bit alpha;
    x q[0];
    cx q[0], q[2];
    zeta[1] = measure q[0];
    alpha = measure q[1];
    zeta[0] = measure q[2];
  )qasm")),
                                   environment);
  ASSERT_TRUE(succeeded(output));
  const auto instructions =
      nlohmann::json::parse(std::get<IQMJSONProgram>(*output).source())
          .at("instructions");
  ASSERT_GT(instructions.size(), 1U);
  EXPECT_EQ(instructions.back().at("name"), "measure");
  EXPECT_EQ(instructions.back().at("locus").size(), 3U);
  for (size_t index = 0; index + 1 < instructions.size(); ++index) {
    EXPECT_NE(instructions[index].at("name"), "measure");
  }
}
