/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "pennylane.h"

#include "qdmi/QDMI.hpp"

#include "QDMIDevice.h"

#include "nanobind/ndarray.h"
#include "nanobind/stl/optional.h"
#include "nanobind/stl/pair.h"
#include "nanobind/stl/shared_ptr.h"
#include "nanobind/stl/string.h"
#include "nanobind/stl/unique_ptr.h"
#include "nanobind/stl/variant.h"
#include "nanobind/stl/vector.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace mqt {
namespace nb = nanobind;
using namespace nb::literals;

using PythonJobParameter =
    std::variant<std::string, bool, int, double, nb::bytes>;
using Samples =
    nb::ndarray<nb::numpy, int8_t, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

static std::optional<catalyst::JobParameter>
ownJobParameter(const std::optional<PythonJobParameter>& parameter) {
  if (!parameter) {
    return std::nullopt;
  }
  return std::visit(
      [](const auto& value) -> catalyst::JobParameter {
        if constexpr (std::is_same_v<std::decay_t<decltype(value)>,
                                     nb::bytes>) {
          const auto bytes =
              std::as_bytes(std::span(value.c_str(), value.size()));
          return std::vector<std::byte>(bytes.begin(), bytes.end());
        } else {
          return value;
        }
      },
      *parameter);
}

void registerPennyLane(nb::module_& m) {
  nb::class_<catalyst::Context>(m, "_PennyLaneQDMIContext")
      .def(nb::new_([](qdmi::Device device,
                       const std::optional<PythonJobParameter>& custom1,
                       const std::optional<PythonJobParameter>& custom2,
                       const std::optional<PythonJobParameter>& custom3,
                       const std::optional<PythonJobParameter>& custom4,
                       const std::optional<PythonJobParameter>& custom5) {
             catalyst::JobParameters parameters{
                 ownJobParameter(custom1), ownJobParameter(custom2),
                 ownJobParameter(custom3), ownJobParameter(custom4),
                 ownJobParameter(custom5),
             };
             const nb::gil_scoped_release release;
             return catalyst::Context::create(std::move(device),
                                              std::move(parameters));
           }),
           "device"_a, nb::kw_only(), "custom1"_a = nb::none(),
           "custom2"_a = nb::none(), "custom3"_a = nb::none(),
           "custom4"_a = nb::none(), "custom5"_a = nb::none())
      .def_prop_ro("id", &catalyst::Context::id)
      .def(
          "recorder",
          [](std::shared_ptr<catalyst::Context> context, size_t wires) {
            auto recorder =
                std::make_unique<catalyst::QDMIDevice>(std::move(context));
            recorder->AllocateQubits(wires);
            return recorder;
          },
          "num_wires"_a, nb::call_guard<nb::gil_scoped_release>())
      .def("try_submit_programs", &catalyst::Context::trySubmitBatch,
           "programs"_a, "shots"_a, nb::call_guard<nb::gil_scoped_release>());

  nb::class_<catalyst::QDMIDevice>(m, "_PennyLaneRecorder")
      .def(
          "named_operation",
          [](catalyst::QDMIDevice& device, const std::string& name,
             const std::vector<double>& parameters,
             const std::vector<QubitIdType>& wires, bool inverse,
             const std::vector<QubitIdType>& controls,
             const std::vector<bool>& controlValues) {
            device.NamedOperation(name, parameters, wires, inverse, controls,
                                  controlValues);
          },
          "name"_a, "parameters"_a, "wires"_a, "inverse"_a = false,
          "controls"_a = std::vector<QubitIdType>{},
          "control_values"_a = std::vector<bool>{},
          nb::call_guard<nb::gil_scoped_release>())
      .def("prepare", &catalyst::QDMIDevice::prepare,
           nb::call_guard<nb::gil_scoped_release>());

  nb::class_<catalyst::PreparedProgram>(m, "_PennyLaneProgram")
      .def_prop_ro("program_format", &catalyst::PreparedProgram::programFormat)
      .def("submit", &catalyst::PreparedProgram::submit, "shots"_a,
           nb::call_guard<nb::gil_scoped_release>())
      .def(
          "decode_samples",
          [](const catalyst::PreparedProgram& program, const qdmi::Job& job,
             size_t shots, size_t programIndex) {
            std::unique_ptr<std::vector<int8_t>> data;
            {
              const nb::gil_scoped_release release;
              data = std::make_unique<std::vector<int8_t>>(
                  program.samples(job, shots, programIndex));
            }
            const nb::capsule owner(data.get(), [](void* pointer) noexcept {
              delete static_cast<std::vector<int8_t>*>(pointer);
            });
            auto* const samples = data.release();
            return Samples(samples->data(), {shots, program.qubits()}, owner);
          },
          "job"_a, "shots"_a, "program_index"_a = 0);

  m.def("_pennylane_c_interface", [] {
    return std::pair{std::string("MQTQDMI"), catalyst::libraryPath()};
  });
}
} // namespace mqt
