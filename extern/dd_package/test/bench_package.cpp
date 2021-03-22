/*
 * This file is part of the JKQ DD Package which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum_dd/ for more information.
 */

#include "DDpackage.h"

#include <benchmark/benchmark.h>
#include <memory>

///
/// Test DD Package creation
///
/// At the moment packages are allocated with a fixed maximum number of qubits (=128)
/// In the future, the maximum number of qubits can be set at construction time
/// until then, each creation should take equally long
///

static void BM_PackageCreation(benchmark::State& state) {
    for (auto _: state) {
        [[maybe_unused]] unsigned short nqubits = state.range(0);
        auto                            dd      = std::make_unique<dd::Package>();
    }
}
BENCHMARK(BM_PackageCreation)->Unit(benchmark::kMillisecond)->RangeMultiplier(2)->Range(16, 128);

///
/// Test creation of identity matrix
///

static void BM_MakeIdent(benchmark::State& state) {
    [[maybe_unused]] unsigned short nqubits = state.range(0);
    auto                            dd      = std::make_unique<dd::Package>();
    for (auto _: state) {
        dd->reset();
        benchmark::DoNotOptimize(dd->makeIdent(nqubits));
    }
}
BENCHMARK(BM_MakeIdent)->Unit(benchmark::kMillisecond)->RangeMultiplier(2)->Range(16, 128);
