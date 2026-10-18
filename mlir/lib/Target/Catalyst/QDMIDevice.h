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

#include "qdmi/QDMI.hpp"

#include "QuantumDevice.hpp"
#include "mqt-catalyst-export.h"

#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <variant>
#include <vector>

namespace mqt::catalyst {

using JobParameter =
    std::variant<std::string, bool, int, double, std::vector<std::byte>>;
using JobParameters = std::array<std::optional<JobParameter>, 5>;

/// Locate this adapter's installed shared library.
[[nodiscard]] MQT_CATALYST_EXPORT std::string libraryPath();

class PreparedProgram;

/// Own one provider session and its retained submission configuration.
class MQT_CATALYST_EXPORT Context {
public:
  static std::shared_ptr<Context> create(qdmi::Device device,
                                         JobParameters parameters = {});
  ~Context();
  [[nodiscard]] uint64_t id() const noexcept;
  [[nodiscard]] size_t qubits() const noexcept;
  [[nodiscard]] std::optional<qdmi::Job>
  trySubmitBatch(const std::vector<std::shared_ptr<PreparedProgram>>& programs,
                 size_t shots) const;

private:
  struct Impl;
  explicit Context(qdmi::Device device, JobParameters parameters);
  std::unique_ptr<Impl> impl_;
  friend class PreparedProgram;
  friend class QDMIDevice;
};

/// An immutable compiled circuit retaining its provider and job parameters.
class MQT_CATALYST_EXPORT PreparedProgram {
public:
  ~PreparedProgram();
  [[nodiscard]] size_t qubits() const noexcept;
  [[nodiscard]] QDMI_Program_Format programFormat() const noexcept;
  [[nodiscard]] const std::string& payload() const noexcept;
  [[nodiscard]] qdmi::Job submit(size_t shots) const;
  /// Decode all logical wires into a row-major shot-by-wire array.
  [[nodiscard]] std::vector<int8_t> samples(const qdmi::Job& job, size_t shots,
                                            size_t programIndex = 0) const;

private:
  struct Impl;
  explicit PreparedProgram(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
  friend class Context;
  friend class QDMIDevice;
};

/// Record one terminal-measurement circuit through Catalyst's device API.
class MQT_CATALYST_EXPORT QDMIDevice final
    : public Catalyst::Runtime::QuantumDevice {
public:
  explicit QDMIDevice(std::shared_ptr<Context> context);
  ~QDMIDevice() override;
  [[nodiscard]] std::shared_ptr<PreparedProgram> prepare();

  std::vector<QubitIdType> AllocateQubits(size_t count) override;
  void ReleaseQubits(const std::vector<QubitIdType>& wires) override;
  [[nodiscard]] size_t GetNumQubits() const override;
  void SetDeviceShots(size_t shots) override;
  [[nodiscard]] size_t GetDeviceShots() const override;
  void SetDevicePRNG(std::mt19937* generator) override;
  void NamedOperation(
      const std::string& name, const std::vector<double>& parameters,
      const std::vector<QubitIdType>& wires, bool inverse = false,
      const std::vector<QubitIdType>& controls = {},
      const std::vector<bool>& controlValues = {},
      const std::vector<std::string>& optionalParameters = {}) override;
  Result Measure(QubitIdType wire, std::optional<int32_t> postselect) override;
  ObsIdType Observable(ObsId id,
                       const std::vector<std::complex<double>>& matrix,
                       const std::vector<QubitIdType>& wires) override;
  ObsIdType
  TensorObservable(const std::vector<ObsIdType>& observables) override;
  void Sample(DataView<double, 2>& samples) override;
  void PartialSample(DataView<double, 2>& samples,
                     const std::vector<QubitIdType>& wires) override;
  void Counts(DataView<double, 1>& eigenvalues,
              DataView<int64_t, 1>& counts) override;
  void PartialCounts(DataView<double, 1>& eigenvalues,
                     DataView<int64_t, 1>& counts,
                     const std::vector<QubitIdType>& wires) override;
  void Probs(DataView<double, 1>& probabilities) override;
  void PartialProbs(DataView<double, 1>& probabilities,
                    const std::vector<QubitIdType>& wires) override;
  double Expval(ObsIdType observable) override;
  double Var(ObsIdType observable) override;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace mqt::catalyst

extern "C" MQT_CATALYST_EXPORT Catalyst::Runtime::QuantumDevice*
// Catalyst resolves the factory by its required ABI name.
// NOLINTNEXTLINE(readability-identifier-naming)
MQTQDMIFactory(const char* kwargs);
