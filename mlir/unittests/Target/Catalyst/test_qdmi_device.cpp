/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Compiler/QDMIAdapter.h"
#include "qdmi/QDMI.hpp"

#include "Driver.hpp"
#include "QDMIDevice.h"

#include "gtest/gtest.h"
#include "qdmi/device.h"

#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/Error.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using mqt::catalyst::Context;
using mqt::catalyst::createDevice;
using mqt::catalyst::decodeSamples;
using mqt::catalyst::JobParameters;
using mqt::catalyst::QDMIDevice;

thread_local std::optional<std::string> returnedShots;
thread_local std::string returnedKeys;
thread_local std::vector<size_t> returnedCounts;
thread_local std::vector<std::byte> receivedParameter;
thread_local decltype(QDMI_device_job_set_parameter)* setParameter;

} // namespace

static int copyResult(std::span<const std::byte> bytes, size_t size,
                      void* value, size_t* sizeRet) {
  if (sizeRet != nullptr) {
    *sizeRet = bytes.size();
  }
  if (value != nullptr) {
    if (size < bytes.size()) {
      return QDMI_ERROR_INVALIDARGUMENT;
    }
    std::memcpy(value, bytes.data(), bytes.size());
  }
  return QDMI_SUCCESS;
}

static int getResults(QDMI_Device_Job, size_t, QDMI_Job_Result result,
                      size_t size, void* value, size_t* sizeRet) {
  if (result == QDMI_JOB_RESULT_SHOTS && returnedShots) {
    return copyResult(std::as_bytes(std::span(returnedShots->c_str(),
                                              returnedShots->size() + 1)),
                      size, value, sizeRet);
  }
  if (result == QDMI_JOB_RESULT_HIST_KEYS && !returnedKeys.empty()) {
    return copyResult(
        std::as_bytes(std::span(returnedKeys.c_str(), returnedKeys.size() + 1)),
        size, value, sizeRet);
  }
  if (result == QDMI_JOB_RESULT_HIST_VALUES && !returnedKeys.empty()) {
    return copyResult(std::as_bytes(std::span(returnedCounts)), size, value,
                      sizeRet);
  }
  return QDMI_ERROR_NOTSUPPORTED;
}

TEST(CatalystQDMI, ValidatesProviderResultsAndRetainsBinaryParameters) {
  auto device = qdmi::builtin_driver::openDevice("mqt.ddsim.default");
  auto& api =
      const_cast<qdmi::DeviceAPI&>(static_cast<QDMI_Device>(device)->api());
  const auto originalGetResults = api.device_job_get_results;
  setParameter = api.device_job_set_parameter;
  const auto restore = llvm::make_scope_exit([&] {
    api.device_job_get_results = originalGetResults;
    api.device_job_set_parameter = setParameter;
  });
  api.device_job_set_parameter = [](QDMI_Device_Job job,
                                    QDMI_Device_Job_Parameter parameter,
                                    size_t size, const void* value) -> int {
    if (parameter == QDMI_DEVICE_JOB_PARAMETER_CUSTOM5) {
      const auto* bytes = static_cast<const std::byte*>(value);
      receivedParameter.assign(bytes, bytes + size);
      return QDMI_SUCCESS;
    }
    return setParameter(job, parameter, size, value);
  };
  const std::vector<std::byte> expected{
      std::byte{0},
      std::byte{42},
      std::byte{255},
  };
  auto context = [&] {
    JobParameters parameters;
    parameters[0] = 42;
    parameters[4] = expected;
    return Context::create(device, std::move(parameters));
  }();
  QDMIDevice recorder(context);
  const auto wires = recorder.AllocateQubits(3);
  recorder.NamedOperation("PauliX", {}, {wires[0]});
  recorder.SetDeviceShots(2);
  std::vector<double> probabilities(8);
  DataView<double, 1> view(probabilities);
  recorder.Probs(view);
  EXPECT_EQ(receivedParameter, expected);
  EXPECT_EQ(probabilities, (std::vector<double>{0, 0, 0, 0, 1, 0, 0, 0}));
  QDMIDevice importer(3);
  importer.NamedOperation("PauliX", {}, {0});
  const auto program =
      llvm::cantFail(mlir::compileProgram(importer.takeProgram(), device));
  auto job = llvm::cantFail(mlir::submitProgram(device, program, 2));
  ASSERT_TRUE(job.wait());
  EXPECT_EQ(decodeSamples(job, 3, 2), (std::vector<int8_t>{1, 0, 0, 1, 0, 0}));
  api.device_job_get_results = getResults;
  returnedKeys.clear();
  returnedShots = "10 0,001";
  EXPECT_EQ(decodeSamples(job, 3, 2), (std::vector<int8_t>{0, 0, 1, 1, 0, 0}));
  for (const auto* malformed :
       {"001", "0000,001", "0x1,001", "00\xFF,001", "000,001,001"}) {
    returnedShots = malformed;
    EXPECT_THROW(decodeSamples(job, 3, 2), std::runtime_error);
  }
  returnedShots.reset();
  returnedKeys = "100,001";
  returnedCounts = {2, 3};
  const auto samples = decodeSamples(job, 3, 5);
  EXPECT_EQ(samples,
            (std::vector<int8_t>{1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1}));
  EXPECT_THROW(decodeSamples(job, 3, 4), std::runtime_error);
  EXPECT_THROW(decodeSamples(job, 3, 6), std::runtime_error);
  returnedKeys = "10x,001";
  EXPECT_THROW(decodeSamples(job, 3, 5), std::runtime_error);
}

TEST(CatalystQDMI, InvalidBasisStateLeavesRecordingUsable) {
  QDMIDevice runtime(
      Context::create(qdmi::builtin_driver::openDevice("mqt.ddsim.default")));
  const auto wires = runtime.AllocateQubits(3);
  runtime.SetDeviceShots(4);
  std::vector<int8_t> bits{1, 2};
  DataView<int8_t, 1> state(bits);
  std::vector<QubitIdType> prepared{wires[2], wires[0]};
  EXPECT_THROW(runtime.SetBasisState(state, prepared), std::runtime_error);
  bits[1] = 0;
  runtime.SetBasisState(state, prepared);
  std::vector<double> probabilities(8);
  DataView<double, 1> probabilityView(probabilities);
  runtime.Probs(probabilityView);
  EXPECT_EQ(probabilities, (std::vector<double>{0, 1, 0, 0, 0, 0, 0, 0}));
}

TEST(CatalystQDMI, FactoryRetainsContextAndRejectsInvalidLifetimes) {
  auto context =
      Context::create(qdmi::builtin_driver::openDevice("mqt.ddsim.default"));
  const auto kwargs = "{'context': " + std::to_string(context->id()) + '}';
  auto runtime = std::unique_ptr<Catalyst::Runtime::QuantumDevice>(
      createDevice(kwargs.c_str()));
  context.reset();
  auto second = std::unique_ptr<Catalyst::Runtime::QuantumDevice>(
      createDevice(kwargs.c_str()));
  const auto wires = runtime->AllocateQubits(1);
  EXPECT_THROW(runtime->NamedOperation(
                   "RY", {std::numeric_limits<double>::quiet_NaN()}, wires),
               std::runtime_error);
  runtime->ReleaseQubits(wires);
  const auto replacement = runtime->AllocateQubits(1);
  EXPECT_THROW(runtime->NamedOperation("PauliX", {}, wires),
               std::runtime_error);
  runtime->ReleaseQubits(replacement);
  runtime.reset();
  second.reset();
  EXPECT_THROW(createDevice(kwargs.c_str()), std::runtime_error);
}
