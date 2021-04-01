/*
 * This file is part of the JKQ DD Package which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum_dd/ for more information.
 */

#include "DDpackage.h"
#include "GateMatrixDefinitions.h"

#include <benchmark/benchmark.h>
#include <memory>

static void QubitRange(benchmark::internal::Benchmark* b) {
    b->Unit(benchmark::kMicrosecond)->RangeMultiplier(2)->Range(2, 128);
}

///
/// Test class creation
///
/// At the moment packages are allocated with a fixed maximum number of qubits (=128)
/// In the future, the maximum number of qubits can be set at construction time
/// until then, each creation should take equally long
///

static void BM_DDVectorNodeCreation(benchmark::State& state) {
    for (auto _: state) {
        auto node = dd::Node{};
        benchmark::DoNotOptimize(node);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_DDVectorNodeCreation)->Unit(benchmark::kNanosecond);

static void BM_DDMatrixNodeCreation(benchmark::State& state) {
    for (auto _: state) {
        auto node = dd::Node{};
        benchmark::DoNotOptimize(node);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_DDMatrixNodeCreation)->Unit(benchmark::kNanosecond);

static void BM_ComplexNumbersCreation(benchmark::State& state) {
    for (auto _: state) {
        auto cn = dd::ComplexNumbers();
        benchmark::DoNotOptimize(cn);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_ComplexNumbersCreation)->Unit(benchmark::kMicrosecond);

static void BM_PackageCreation(benchmark::State& state) {
    [[maybe_unused]] unsigned short nqubits = state.range(0);
    for (auto _: state) {
        auto dd = std::make_unique<dd::Package>();
        benchmark::DoNotOptimize(dd);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_PackageCreation)->Unit(benchmark::kMillisecond)->RangeMultiplier(2)->Range(2, 128);

///
/// Test creation of identity matrix
///

static void BM_MakeIdentCached(benchmark::State& state) {
    unsigned short nqubits = state.range(0);
    auto           dd      = std::make_unique<dd::Package>();
    for (auto _: state) {
        benchmark::DoNotOptimize(dd->makeIdent(nqubits));
    }
}
BENCHMARK(BM_MakeIdentCached)->Unit(benchmark::kNanosecond)->RangeMultiplier(2)->Range(2, 128);

static void BM_MakeIdent(benchmark::State& state) {
    unsigned short nqubits = state.range(0);
    auto           dd      = std::make_unique<dd::Package>();
    auto           line    = std::array<short, dd::MAXN>{};
    for (auto _: state) {
        line.fill(-1);
        line[0] = 2;
        benchmark::DoNotOptimize(dd->makeGateDD(dd::Imat, nqubits, line));
    }
}
BENCHMARK(BM_MakeIdent)->Apply(QubitRange);

///
/// Test makeGateDD
///

static void BM_MakeSingleQubitGateDD_TargetTop(benchmark::State& state) {
    unsigned short nqubits = state.range(0);
    auto           dd      = std::make_unique<dd::Package>();
    auto           line    = std::array<short, dd::MAXN>{};
    for (auto _: state) {
        line.fill(-1);
        line[nqubits - 1] = 2;
        benchmark::DoNotOptimize(dd->makeGateDD(dd::Xmat, nqubits, line));
    }
}
BENCHMARK(BM_MakeSingleQubitGateDD_TargetTop)->Apply(QubitRange);

static void BM_MakeSingleQubitGateDD_TargetMiddle(benchmark::State& state) {
    unsigned short nqubits = state.range(0);
    auto           dd      = std::make_unique<dd::Package>();
    auto           line    = std::array<short, dd::MAXN>{};
    for (auto _: state) {
        line.fill(-1);
        line[nqubits / 2] = 2;
        benchmark::DoNotOptimize(dd->makeGateDD(dd::Xmat, nqubits, line));
    }
}
BENCHMARK(BM_MakeSingleQubitGateDD_TargetMiddle)->Apply(QubitRange);

static void BM_MakeSingleQubitGateDD_TargetBottom(benchmark::State& state) {
    unsigned short nqubits = state.range(0);
    auto           dd      = std::make_unique<dd::Package>();
    auto           line    = std::array<short, dd::MAXN>{};
    for (auto _: state) {
        line.fill(-1);
        line[0] = 2;
        benchmark::DoNotOptimize(dd->makeGateDD(dd::Xmat, nqubits, line));
    }
}
BENCHMARK(BM_MakeSingleQubitGateDD_TargetBottom)->Apply(QubitRange);

static void BM_MakeControlledQubitGateDD_ControlBottom_TargetTop(benchmark::State& state) {
    unsigned short nqubits = state.range(0);
    auto           dd      = std::make_unique<dd::Package>();
    auto           line    = std::array<short, dd::MAXN>{};
    for (auto _: state) {
        line.fill(-1);
        line[0]           = 1;
        line[nqubits - 1] = 2;
        benchmark::DoNotOptimize(dd->makeGateDD(dd::Xmat, nqubits, line));
    }
}
BENCHMARK(BM_MakeControlledQubitGateDD_ControlBottom_TargetTop)->Apply(QubitRange);

static void BM_MakeControlledQubitGateDD_ControlBottom_TargetMiddle(benchmark::State& state) {
    unsigned short nqubits = state.range(0);
    auto           dd      = std::make_unique<dd::Package>();
    auto           line    = std::array<short, dd::MAXN>{};
    for (auto _: state) {
        line.fill(-1);
        line[0]           = 1;
        line[nqubits / 2] = 2;
        benchmark::DoNotOptimize(dd->makeGateDD(dd::Xmat, nqubits, line));
    }
}
BENCHMARK(BM_MakeControlledQubitGateDD_ControlBottom_TargetMiddle)->Apply(QubitRange);

static void BM_MakeControlledQubitGateDD_ControlTop_TargetMiddle(benchmark::State& state) {
    unsigned short nqubits = state.range(0);
    auto           dd      = std::make_unique<dd::Package>();
    auto           line    = std::array<short, dd::MAXN>{};
    for (auto _: state) {
        line.fill(-1);
        line[nqubits - 1] = 1;
        line[nqubits / 2] = 2;
        benchmark::DoNotOptimize(dd->makeGateDD(dd::Xmat, nqubits, line));
    }
}
BENCHMARK(BM_MakeControlledQubitGateDD_ControlTop_TargetMiddle)->Apply(QubitRange);

static void BM_MakeControlledQubitGateDD_ControlTop_TargetBottom(benchmark::State& state) {
    unsigned short nqubits = state.range(0);
    auto           dd      = std::make_unique<dd::Package>();
    auto           line    = std::array<short, dd::MAXN>{};
    for (auto _: state) {
        line.fill(-1);
        line[nqubits - 1] = 1;
        line[0]           = 2;
        benchmark::DoNotOptimize(dd->makeGateDD(dd::Xmat, nqubits, line));
    }
}
BENCHMARK(BM_MakeControlledQubitGateDD_ControlTop_TargetBottom)->Apply(QubitRange);

static void BM_MakeFullControlledToffoliDD_TargetTop(benchmark::State& state) {
    unsigned short nqubits = state.range(0);
    auto           dd      = std::make_unique<dd::Package>();
    auto           line    = std::array<short, dd::MAXN>{};
    for (auto _: state) {
        line.fill(1);
        line[nqubits - 1] = 2;
        benchmark::DoNotOptimize(dd->makeGateDD(dd::Xmat, nqubits, line));
    }
}
BENCHMARK(BM_MakeFullControlledToffoliDD_TargetTop)->Apply(QubitRange);

static void BM_MakeFullControlledToffoliDD_TargetMiddle(benchmark::State& state) {
    unsigned short nqubits = state.range(0);
    auto           dd      = std::make_unique<dd::Package>();
    auto           line    = std::array<short, dd::MAXN>{};
    for (auto _: state) {
        line.fill(1);
        line[nqubits / 2] = 2;
        benchmark::DoNotOptimize(dd->makeGateDD(dd::Xmat, nqubits, line));
    }
}
BENCHMARK(BM_MakeFullControlledToffoliDD_TargetMiddle)->Apply(QubitRange);

static void BM_MakeFullControlledToffoliDD_TargetBottom(benchmark::State& state) {
    unsigned short nqubits = state.range(0);
    auto           dd      = std::make_unique<dd::Package>();
    auto           line    = std::array<short, dd::MAXN>{};
    for (auto _: state) {
        line.fill(1);
        line[0] = 2;
        benchmark::DoNotOptimize(dd->makeGateDD(dd::Xmat, nqubits, line));
    }
}
BENCHMARK(BM_MakeFullControlledToffoliDD_TargetBottom)->Apply(QubitRange);

static void BM_MakeSWAPDD(benchmark::State& state) {
    unsigned short nqubits = state.range(0);
    auto           dd      = std::make_unique<dd::Package>();
    auto           line    = std::array<short, dd::MAXN>{};
    line.fill(-1);

    for (auto _: state) {
        line[nqubits - 1] = 1;
        line[0]           = 2;
        auto sv           = dd->makeGateDD(dd::Xmat, nqubits, line);
        line[nqubits - 1] = 2;
        line[0]           = 1;
        sv                = dd->multiply(sv, dd->multiply(dd->makeGateDD(dd::Xmat, nqubits, line), sv));
        benchmark::DoNotOptimize(sv);
        dd->clearComputeTables();
    }
}
BENCHMARK(BM_MakeSWAPDD)->Apply(QubitRange);

///
/// Test multiplication
///

static void BM_MxV_X(benchmark::State& state) {
    unsigned short nqubits = state.range(0);
    auto           dd      = std::make_unique<dd::Package>();
    auto           line    = std::array<short, dd::MAXN>{};

    for (auto _: state) {
        auto zero = dd->makeZeroState(nqubits);
        line.fill(-1);
        line[0]  = 2;
        auto x   = dd->makeGateDD(dd::Xmat, nqubits, line);
        auto sim = dd->multiply(x, zero);
        benchmark::DoNotOptimize(sim);
        // clear compute table so the next iteration does not find the result cached
        dd->clearComputeTables();
    }
}
BENCHMARK(BM_MxV_X)->Apply(QubitRange);

static void BM_MxV_H(benchmark::State& state) {
    unsigned short nqubits = state.range(0);
    auto           dd      = std::make_unique<dd::Package>();
    auto           line    = std::array<short, dd::MAXN>{};

    for (auto _: state) {
        auto zero = dd->makeZeroState(nqubits);
        line.fill(-1);
        line[0]  = 2;
        auto h   = dd->makeGateDD(dd::Hmat, nqubits, line);
        auto sim = dd->multiply(h, zero);
        benchmark::DoNotOptimize(sim);
        // clear compute table so the next iteration does not find the result cached
        dd->clearComputeTables();
    }
}
BENCHMARK(BM_MxV_H)->Apply(QubitRange);

static void BM_MxV_T(benchmark::State& state) {
    unsigned short nqubits = state.range(0);
    auto           dd      = std::make_unique<dd::Package>();
    auto           line    = std::array<short, dd::MAXN>{};

    for (auto _: state) {
        auto zero = dd->makeZeroState(nqubits);
        line.fill(-1);
        line[0]  = 2;
        auto t   = dd->makeGateDD(dd::Tmat, nqubits, line);
        auto sim = dd->multiply(t, zero);
        benchmark::DoNotOptimize(sim);
        // clear compute table so the next iteration does not find the result cached
        dd->clearComputeTables();
    }
}
BENCHMARK(BM_MxV_T)->Apply(QubitRange);

static void BM_MxV_CX_ControlTop_TargetBottom(benchmark::State& state) {
    unsigned short nqubits     = state.range(0);
    auto           dd          = std::make_unique<dd::Package>();
    auto           line        = std::array<short, dd::MAXN>{};
    auto           basisStates = std::vector<dd::BasisStates>{nqubits, dd::BasisStates::zero};
    basisStates[nqubits - 1]   = dd::BasisStates::plus;

    for (auto _: state) {
        auto plus = dd->makeBasisState(nqubits, basisStates);
        line.fill(-1);
        line[0]           = 2;
        line[nqubits - 1] = 1;
        auto cx           = dd->makeGateDD(dd::Xmat, nqubits, line);
        auto sim          = dd->multiply(cx, plus);
        benchmark::DoNotOptimize(sim);
        // clear compute table so the next iteration does not find the result cached
        dd->clearComputeTables();
    }
}
BENCHMARK(BM_MxV_CX_ControlTop_TargetBottom)->Apply(QubitRange);

static void BM_MxV_CX_ControlBottom_TargetTop(benchmark::State& state) {
    unsigned short nqubits     = state.range(0);
    auto           dd          = std::make_unique<dd::Package>();
    auto           line        = std::array<short, dd::MAXN>{};
    auto           basisStates = std::vector<dd::BasisStates>{nqubits, dd::BasisStates::zero};
    basisStates[0]             = dd::BasisStates::plus;

    for (auto _: state) {
        auto plus = dd->makeBasisState(nqubits, basisStates);
        line.fill(-1);
        line[nqubits - 1] = 2;
        line[0]           = 1;
        auto cx           = dd->makeGateDD(dd::Xmat, nqubits, line);
        auto sim          = dd->multiply(cx, plus);
        benchmark::DoNotOptimize(sim);
        // clear compute table so the next iteration does not find the result cached
        dd->clearComputeTables();
    }
}
BENCHMARK(BM_MxV_CX_ControlBottom_TargetTop)->Apply(QubitRange);

static void BM_MxV_HadamardLayer(benchmark::State& state) {
    unsigned short nqubits = state.range(0);
    auto           dd      = std::make_unique<dd::Package>();
    auto           line    = std::array<short, dd::MAXN>{};

    for (auto _: state) {
        auto sv = dd->makeZeroState(nqubits);
        line.fill(-1);
        for (int i = 0; i < nqubits; ++i) {
            line[std::max(0, i - 1)] = -1;
            line[i]                  = 2;
            auto h                   = dd->makeGateDD(dd::Hmat, nqubits, line);
            sv                       = dd->multiply(h, sv);
        }
        // clear compute table so the next iteration does not find the result cached
        dd->clearComputeTables();
    }
}
BENCHMARK(BM_MxV_HadamardLayer)->Apply(QubitRange);

static void BM_MxV_GHZ(benchmark::State& state) {
    unsigned short nqubits = state.range(0);
    auto           dd      = std::make_unique<dd::Package>();
    auto           line    = std::array<short, dd::MAXN>{};

    for (auto _: state) {
        auto sv = dd->makeZeroState(nqubits);
        line.fill(-1);
        line[nqubits - 1] = 2;
        auto h            = dd->makeGateDD(dd::Hmat, nqubits, line);
        sv                = dd->multiply(h, sv);
        line[nqubits - 1] = 1;
        for (int i = nqubits - 2; i >= 0; --i) {
            line[std::min(nqubits-2, i+1)] = -1;
            line[i] = 2;
            auto cx = dd->makeGateDD(dd::Xmat, nqubits, line);
            sv      = dd->multiply(cx, sv);
        }
        // clear compute table so the next iteration does not find the result cached
        dd->clearComputeTables();
    }
}
BENCHMARK(BM_MxV_GHZ)->Apply(QubitRange);

static void BM_MxM_Bell(benchmark::State& state) {
    unsigned short nqubits = state.range(0);
    auto           dd      = std::make_unique<dd::Package>();
    auto           line    = std::array<short, dd::MAXN>{};

    for (auto _: state) {
        line.fill(-1);
        line[nqubits - 1] = 2;
        auto h            = dd->makeGateDD(dd::Hmat, nqubits, line);
        line[nqubits - 1] = 1;
        line[0]           = 2;
        auto cx           = dd->makeGateDD(dd::Xmat, nqubits, line);
        auto bell         = dd->multiply(cx, h);
        benchmark::DoNotOptimize(bell);
        // clear compute table so the next iteration does not find the result cached
        dd->clearComputeTables();
    }
}
BENCHMARK(BM_MxM_Bell)->Apply(QubitRange);

static void BM_MxM_GHZ(benchmark::State& state) {
    unsigned short nqubits = state.range(0);
    auto           dd      = std::make_unique<dd::Package>();
    auto           line    = std::array<short, dd::MAXN>{};

    for (auto _: state) {
        line.fill(-1);
        line[nqubits - 1] = 2;
        auto func         = dd->makeGateDD(dd::Hmat, nqubits, line);
        line[nqubits - 1] = 1;
        for (int i = nqubits - 2; i >= 0; --i) {
            line[std::min(nqubits-2, i+1)] = -1;
            line[i] = 2;
            auto cx = dd->makeGateDD(dd::Xmat, nqubits, line);
            func    = dd->multiply(cx, func);
        }
        // clear compute table so the next iteration does not find the result cached
        dd->clearComputeTables();
    }
}
BENCHMARK(BM_MxM_GHZ)->Apply(QubitRange);
