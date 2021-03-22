/*
 * This file is part of the JKQ DD Package which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum_dd/ for more information.
 */

#include "DDpackage.h"

#include <benchmark/benchmark.h>
#include <memory>

static void BM_PackageCreation(benchmark::State& state) {
    for (auto _: state) {
        // at the moment packages are allocated with a fixed maximum number of qubits (=128)
        // in the future, the maximum number of qubits can be set at construction time
        // until then, each creation takes equally long
        [[maybe_unused]] unsigned short nqubits = state.range(0);
        auto dd = std::make_unique<dd::Package>();
    }
}
BENCHMARK(BM_PackageCreation)->Unit(benchmark::kMillisecond)->RangeMultiplier(2)->Range(1, 128);
