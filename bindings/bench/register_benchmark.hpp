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

#include "bench/JSON.hpp"

#include "nanobind/nanobind.h"
#include "nanobind/stl/string.h"      // NOLINT(misc-include-cleaner)
#include "nanobind/stl/string_view.h" // NOLINT(misc-include-cleaner)

#include <string_view>

namespace mqt {

template <class Benchmark>
void registerBenchmarkIO(
    nanobind::class_<Benchmark>& benchmark,
    Benchmark (*fromInstanceSpecification)(std::string_view, std::string_view),
    Benchmark (*fromManifest)(std::string_view, std::string_view)) {
  namespace nb = nanobind;
  using namespace nb::literals;

  benchmark
      .def(
          "generate",
          [](const Benchmark& value) {
            return nb::module_::import_("mqt.core.mlir")
                .attr("_generate_benchmark")(
                    bench::toInstanceSpecificationJSON(value));
          },
          nb::sig("def generate(self) -> mqt.core.mlir.QCProgram"),
          "Generate the benchmark as a QC program.")
      .def_prop_ro(
          "instance_specification_json",
          [](const Benchmark& value) {
            return bench::toInstanceSpecificationJSON(value);
          },
          "The canonical instance specification JSON.")
      .def_prop_ro(
          "manifest_json",
          [](const Benchmark& value) { return bench::toManifestJSON(value); },
          "The canonical manifest JSON.")
      .def_prop_ro(
          "case_id",
          [](const Benchmark& value) { return bench::caseId(value); },
          "The stable semantic case ID.")
      .def_static("from_instance_specification_json", fromInstanceSpecification,
                  "json"_a, nb::kw_only(),
                  "source"_a = "<instance-specification>",
                  "Parse a strict benchmark instance specification.")
      .def_static("from_manifest_json", fromManifest, "json"_a, nb::kw_only(),
                  "source"_a = "<manifest>",
                  "Parse a strict benchmark manifest.");
}

} // namespace mqt
