/*
 * This file is part of the MQT DD Package which is released under the MIT license.
 * See file README.md or go to https://www.cda.cit.tum.de/research/quantum_dd/ for more information.
 */

#include "dd/GateMatrixDefinitions.hpp"
#include "dd/Package.hpp"

#include <benchmark/benchmark.h>
#include <memory>

using namespace dd::literals;

static void qubitRange(benchmark::internal::Benchmark* b) {
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
    for ([[maybe_unused]] auto _: state) {
        auto node = dd::vNode{};
        benchmark::DoNotOptimize(node);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_DDVectorNodeCreation)->Unit(benchmark::kNanosecond); // NOLINT

static void BM_DDMatrixNodeCreation(benchmark::State& state) {
    for ([[maybe_unused]] auto _: state) {
        auto node = dd::mNode{};
        benchmark::DoNotOptimize(node);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_DDMatrixNodeCreation)->Unit(benchmark::kNanosecond); // NOLINT

static void bmComplexNumbersCreation(benchmark::State& state) {
    for ([[maybe_unused]] auto _: state) {
        auto cn = dd::ComplexNumbers();
        benchmark::DoNotOptimize(cn);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(bmComplexNumbersCreation)->Unit(benchmark::kMicrosecond); // NOLINT

static void bmPackageCreation(benchmark::State& state) {
    const auto nqubits = static_cast<dd::QubitCount>(state.range(0));
    for ([[maybe_unused]] auto _: state) {
        auto dd = std::make_unique<dd::Package<>>(nqubits);
        benchmark::DoNotOptimize(dd);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(bmPackageCreation)->Unit(benchmark::kMillisecond)->RangeMultiplier(2)->Range(2, 128); // NOLINT

///
/// Test creation of identity matrix
///

static void bmMakeIdentCached(benchmark::State& state) {
    const auto nqubits = static_cast<dd::QubitCount>(state.range(0));
    auto       dd      = std::make_unique<dd::Package<>>(nqubits);
    for ([[maybe_unused]] auto _: state) {
        benchmark::DoNotOptimize(dd->makeIdent(nqubits));
    }
}
BENCHMARK(bmMakeIdentCached)->Unit(benchmark::kNanosecond)->RangeMultiplier(2)->Range(2, 128); // NOLINT

static void bmMakeIdent(benchmark::State& state) {
    const auto nqubits = static_cast<dd::QubitCount>(state.range(0));
    auto       dd      = std::make_unique<dd::Package<>>(nqubits);
    for ([[maybe_unused]] auto _: state) {
        benchmark::DoNotOptimize(dd->makeGateDD(dd::Imat, nqubits, 0));
    }
}
BENCHMARK(bmMakeIdent)->Apply(qubitRange); // NOLINT

///
/// Test makeGateDD
///

static void BM_MakeSingleQubitGateDD_TargetTop(benchmark::State& state) {
    const auto nqubits = static_cast<dd::QubitCount>(state.range(0));
    auto       dd      = std::make_unique<dd::Package<>>(nqubits);
    for ([[maybe_unused]] auto _: state) {
        benchmark::DoNotOptimize(dd->makeGateDD(dd::Xmat, nqubits, static_cast<dd::Qubit>(nqubits - 1)));
    }
}
BENCHMARK(BM_MakeSingleQubitGateDD_TargetTop)->Apply(qubitRange); // NOLINT

static void BM_MakeSingleQubitGateDD_TargetMiddle(benchmark::State& state) {
    const auto nqubits = static_cast<dd::QubitCount>(state.range(0));
    auto       dd      = std::make_unique<dd::Package<>>(nqubits);
    for ([[maybe_unused]] auto _: state) {
        benchmark::DoNotOptimize(dd->makeGateDD(dd::Xmat, nqubits, static_cast<dd::Qubit>(nqubits / 2)));
    }
}
BENCHMARK(BM_MakeSingleQubitGateDD_TargetMiddle)->Apply(qubitRange); // NOLINT

static void BM_MakeSingleQubitGateDD_TargetBottom(benchmark::State& state) {
    const auto nqubits = static_cast<dd::QubitCount>(state.range(0));
    auto       dd      = std::make_unique<dd::Package<>>(nqubits);
    for ([[maybe_unused]] auto _: state) {
        benchmark::DoNotOptimize(dd->makeGateDD(dd::Xmat, nqubits, 0));
    }
}
BENCHMARK(BM_MakeSingleQubitGateDD_TargetBottom)->Apply(qubitRange); // NOLINT

static void BM_MakeControlledQubitGateDD_ControlBottom_TargetTop(benchmark::State& state) {
    const auto nqubits = static_cast<dd::QubitCount>(state.range(0));
    auto       dd      = std::make_unique<dd::Package<>>(nqubits);
    for ([[maybe_unused]] auto _: state) {
        benchmark::DoNotOptimize(dd->makeGateDD(dd::Xmat, nqubits, 0_pc, static_cast<dd::Qubit>(nqubits - 1)));
    }
}
BENCHMARK(BM_MakeControlledQubitGateDD_ControlBottom_TargetTop)->Apply(qubitRange); // NOLINT

static void BM_MakeControlledQubitGateDD_ControlBottom_TargetMiddle(benchmark::State& state) {
    const auto nqubits = static_cast<dd::QubitCount>(state.range(0));
    auto       dd      = std::make_unique<dd::Package<>>(nqubits);
    for ([[maybe_unused]] auto _: state) {
        benchmark::DoNotOptimize(dd->makeGateDD(dd::Xmat, nqubits, 0_pc, static_cast<dd::Qubit>(nqubits / 2)));
    }
}
BENCHMARK(BM_MakeControlledQubitGateDD_ControlBottom_TargetMiddle)->Apply(qubitRange); // NOLINT

static void BM_MakeControlledQubitGateDD_ControlTop_TargetMiddle(benchmark::State& state) {
    const auto nqubits = static_cast<dd::QubitCount>(state.range(0));
    auto       dd      = std::make_unique<dd::Package<>>(nqubits);
    for ([[maybe_unused]] auto _: state) {
        benchmark::DoNotOptimize(dd->makeGateDD(dd::Xmat, nqubits, dd::Control{static_cast<dd::Qubit>(nqubits - 1)}, static_cast<dd::Qubit>(nqubits / 2)));
    }
}
BENCHMARK(BM_MakeControlledQubitGateDD_ControlTop_TargetMiddle)->Apply(qubitRange); // NOLINT

static void BM_MakeControlledQubitGateDD_ControlTop_TargetBottom(benchmark::State& state) {
    const auto nqubits = static_cast<dd::QubitCount>(state.range(0));
    auto       dd      = std::make_unique<dd::Package<>>(nqubits);
    for ([[maybe_unused]] auto _: state) {
        benchmark::DoNotOptimize(dd->makeGateDD(dd::Xmat, nqubits, dd::Control{static_cast<dd::Qubit>(nqubits - 1)}, 0));
    }
}
BENCHMARK(BM_MakeControlledQubitGateDD_ControlTop_TargetBottom)->Apply(qubitRange); // NOLINT

static void BM_MakeFullControlledToffoliDD_TargetTop(benchmark::State& state) {
    const auto   nqubits = static_cast<dd::QubitCount>(state.range(0));
    auto         dd      = std::make_unique<dd::Package<>>(nqubits);
    dd::Controls controls;
    for (std::size_t i = 0; i < static_cast<std::size_t>(nqubits - 1); i++) {
        controls.insert({static_cast<dd::Qubit>(i)});
    }
    for ([[maybe_unused]] auto _: state) {
        benchmark::DoNotOptimize(dd->makeGateDD(dd::Xmat, nqubits, controls, static_cast<dd::Qubit>(nqubits - 1)));
    }
}
BENCHMARK(BM_MakeFullControlledToffoliDD_TargetTop)->Apply(qubitRange); // NOLINT

static void BM_MakeFullControlledToffoliDD_TargetMiddle(benchmark::State& state) {
    const auto   nqubits = static_cast<dd::QubitCount>(state.range(0));
    auto         dd      = std::make_unique<dd::Package<>>(nqubits);
    dd::Controls controls;
    for (std::size_t i = 0; i < nqubits; i++) {
        if (i != nqubits / 2) {
            controls.insert({static_cast<dd::Qubit>(i)});
        }
    }
    for ([[maybe_unused]] auto _: state) {
        benchmark::DoNotOptimize(dd->makeGateDD(dd::Xmat, nqubits, controls, static_cast<dd::Qubit>(nqubits / 2)));
    }
}
BENCHMARK(BM_MakeFullControlledToffoliDD_TargetMiddle)->Apply(qubitRange); // NOLINT

static void BM_MakeFullControlledToffoliDD_TargetBottom(benchmark::State& state) {
    const auto   nqubits = static_cast<dd::QubitCount>(state.range(0));
    auto         dd      = std::make_unique<dd::Package<>>(nqubits);
    dd::Controls controls;
    for (std::size_t i = 1; i < nqubits; i++) {
        controls.insert({static_cast<dd::Qubit>(i)});
    }
    for ([[maybe_unused]] auto _: state) {
        benchmark::DoNotOptimize(dd->makeGateDD(dd::Xmat, nqubits, controls, 0));
    }
}
BENCHMARK(BM_MakeFullControlledToffoliDD_TargetBottom)->Apply(qubitRange); // NOLINT

static void BM_MakeSWAPDD(benchmark::State& state) {
    const auto nqubits = static_cast<dd::QubitCount>(state.range(0));
    auto       dd      = std::make_unique<dd::Package<>>(nqubits);

    for ([[maybe_unused]] auto _: state) {
        auto sv = dd->makeGateDD(dd::Xmat, nqubits, dd::Control{static_cast<dd::Qubit>(nqubits - 1)}, 0);
        sv      = dd->multiply(sv, dd->multiply(dd->makeGateDD(dd::Xmat, nqubits, 0_pc, static_cast<dd::Qubit>(nqubits - 1)), sv));

        benchmark::DoNotOptimize(sv);
        dd->clearComputeTables();
    }
}
BENCHMARK(BM_MakeSWAPDD)->Apply(qubitRange); // NOLINT

///
/// Test multiplication
///

static void bmMxVX(benchmark::State& state) {
    const auto nqubits = static_cast<dd::QubitCount>(state.range(0));
    auto       dd      = std::make_unique<dd::Package<>>(nqubits);
    auto       zero    = dd->makeZeroState(nqubits);
    auto       x       = dd->makeGateDD(dd::Xmat, nqubits, 0);

    for ([[maybe_unused]] auto _: state) {
        auto sim = dd->multiply(x, zero);
        benchmark::DoNotOptimize(sim);
        // clear compute table so the next iteration does not find the result cached
        dd->clearComputeTables();
    }
}
BENCHMARK(bmMxVX)->Apply(qubitRange); // NOLINT

static void bmMxVH(benchmark::State& state) {
    const auto nqubits = static_cast<dd::QubitCount>(state.range(0));
    auto       dd      = std::make_unique<dd::Package<>>(nqubits);
    auto       zero    = dd->makeZeroState(nqubits);
    auto       h       = dd->makeGateDD(dd::Hmat, nqubits, 0);

    for ([[maybe_unused]] auto _: state) {
        auto sim = dd->multiply(h, zero);
        benchmark::DoNotOptimize(sim);
        // clear compute table so the next iteration does not find the result cached
        dd->clearComputeTables();
    }
}
BENCHMARK(bmMxVH)->Apply(qubitRange); // NOLINT

static void bmMxVT(benchmark::State& state) {
    const auto nqubits = static_cast<dd::QubitCount>(state.range(0));
    auto       dd      = std::make_unique<dd::Package<>>(nqubits);
    auto       zero    = dd->makeZeroState(nqubits);
    auto       t       = dd->makeGateDD(dd::Tmat, nqubits, 0);

    for ([[maybe_unused]] auto _: state) {
        auto sim = dd->multiply(t, zero);
        benchmark::DoNotOptimize(sim);
        // clear compute table so the next iteration does not find the result cached
        dd->clearComputeTables();
    }
}
BENCHMARK(bmMxVT)->Apply(qubitRange); // NOLINT

static void bmMxVCxControlTopTargetBottom(benchmark::State& state) {
    const auto nqubits       = static_cast<dd::QubitCount>(state.range(0));
    auto       dd            = std::make_unique<dd::Package<>>(nqubits);
    auto       basisStates   = std::vector<dd::BasisStates>{nqubits, dd::BasisStates::zero};
    basisStates[nqubits - 1] = dd::BasisStates::plus;
    auto plus                = dd->makeBasisState(nqubits, basisStates);
    auto cx                  = dd->makeGateDD(dd::Xmat, nqubits, dd::Control{static_cast<dd::Qubit>(nqubits - 1)}, 0);

    for ([[maybe_unused]] auto _: state) {
        auto sim = dd->multiply(cx, plus);
        benchmark::DoNotOptimize(sim);
        // clear compute table so the next iteration does not find the result cached
        dd->clearComputeTables();
    }
}
BENCHMARK(bmMxVCxControlTopTargetBottom)->Apply(qubitRange); // NOLINT

static void bmMxVCxControlBottomTargetTop(benchmark::State& state) {
    const auto nqubits     = static_cast<dd::QubitCount>(state.range(0));
    auto       dd          = std::make_unique<dd::Package<>>(nqubits);
    auto       basisStates = std::vector<dd::BasisStates>{nqubits, dd::BasisStates::zero};
    basisStates[0]         = dd::BasisStates::plus;
    auto plus              = dd->makeBasisState(nqubits, basisStates);
    auto cx                = dd->makeGateDD(dd::Xmat, nqubits, 0_pc, static_cast<dd::Qubit>(nqubits - 1));

    for ([[maybe_unused]] auto _: state) {
        auto sim = dd->multiply(cx, plus);
        benchmark::DoNotOptimize(sim);
        // clear compute table so the next iteration does not find the result cached
        dd->clearComputeTables();
    }
}
BENCHMARK(bmMxVCxControlBottomTargetTop)->Apply(qubitRange); // NOLINT

static void bmMxVHadamardLayer(benchmark::State& state) {
    const auto nqubits = static_cast<dd::QubitCount>(state.range(0));
    auto       dd      = std::make_unique<dd::Package<>>(nqubits);
    auto       zero    = dd->makeZeroState(nqubits);

    for ([[maybe_unused]] auto _: state) {
        auto sv = zero;
        for (int i = 0; i < nqubits; ++i) {
            auto h = dd->makeGateDD(dd::Hmat, nqubits, static_cast<dd::Qubit>(i));
            sv     = dd->multiply(h, sv);
        }
        // clear compute table so the next iteration does not find the result cached
        dd->clearComputeTables();
    }
}
BENCHMARK(bmMxVHadamardLayer)->Apply(qubitRange); // NOLINT

static void bmMxVGhz(benchmark::State& state) {
    const auto nqubits = static_cast<dd::QubitCount>(state.range(0));
    auto       dd      = std::make_unique<dd::Package<>>(nqubits);
    auto       zero    = dd->makeZeroState(nqubits);
    auto       h       = dd->makeGateDD(dd::Hmat, nqubits, static_cast<dd::Qubit>(nqubits - 1));

    for ([[maybe_unused]] auto _: state) {
        auto sv = zero;
        sv      = dd->multiply(h, sv);
        for (auto i = static_cast<std::int64_t>(nqubits - 2); i >= 0; --i) {
            auto cx = dd->makeGateDD(dd::Xmat, nqubits, dd::Control{static_cast<dd::Qubit>(nqubits - 1)}, static_cast<dd::Qubit>(i));
            sv      = dd->multiply(cx, sv);
        }
        // clear compute table so the next iteration does not find the result cached
        dd->clearComputeTables();
    }
}
BENCHMARK(bmMxVGhz)->Apply(qubitRange); // NOLINT

static void bmMxMBell(benchmark::State& state) {
    const auto nqubits = static_cast<dd::QubitCount>(state.range(0));
    auto       dd      = std::make_unique<dd::Package<>>(nqubits);
    auto       h       = dd->makeGateDD(dd::Hmat, nqubits, static_cast<dd::Qubit>(nqubits - 1));
    auto       cx      = dd->makeGateDD(dd::Xmat, nqubits, dd::Control{static_cast<dd::Qubit>(nqubits - 1)}, 0);

    for ([[maybe_unused]] auto _: state) {
        auto bell = dd->multiply(cx, h);
        benchmark::DoNotOptimize(bell);
        // clear compute table so the next iteration does not find the result cached
        dd->clearComputeTables();
    }
}
BENCHMARK(bmMxMBell)->Apply(qubitRange); // NOLINT

static void bmMxMGhz(benchmark::State& state) {
    const auto nqubits = static_cast<dd::QubitCount>(state.range(0));
    auto       dd      = std::make_unique<dd::Package<>>(nqubits);

    for ([[maybe_unused]] auto _: state) {
        auto func = dd->makeGateDD(dd::Hmat, nqubits, static_cast<dd::Qubit>(nqubits - 1));
        for (auto i = static_cast<std::int64_t>(nqubits - 2); i >= 0; --i) {
            auto cx = dd->makeGateDD(dd::Xmat, nqubits, dd::Control{static_cast<dd::Qubit>(nqubits - 1)}, static_cast<dd::Qubit>(i));
            func    = dd->multiply(cx, func);
        }
        // clear compute table so the next iteration does not find the result cached
        dd->clearComputeTables();
    }
}
BENCHMARK(bmMxMGhz)->Apply(qubitRange); // NOLINT

static void bmVUniqueTableGet(benchmark::State& state) {
    auto allocs = state.range(0);
    for ([[maybe_unused]] auto _: state) {
        auto dd = std::make_unique<dd::Package<>>(1);
        for (int i = 0; i < allocs; ++i) {
            auto* p = dd->vUniqueTable.getNode();
            benchmark::DoNotOptimize(p);
        }
    }
}
BENCHMARK(bmVUniqueTableGet)->Unit(benchmark::kMillisecond)->RangeMultiplier(10)->Range(10, 10000000); // NOLINT

static void bmVUniqueTableGetAndReturn(benchmark::State& state) {
    auto allocs = state.range(0);
    for ([[maybe_unused]] auto _: state) {
        auto dd = std::make_unique<dd::Package<>>(1);
        for (int i = 0; i < allocs; ++i) {
            auto* p = dd->vUniqueTable.getNode();
            benchmark::DoNotOptimize(p);
            if (i % 5 == 0) {
                dd->vUniqueTable.returnNode(p);
            }
        }
    }
}
BENCHMARK(bmVUniqueTableGetAndReturn)->Unit(benchmark::kMillisecond)->RangeMultiplier(10)->Range(10, 10000000); // NOLINT

static void bmMUniqueTableGet(benchmark::State& state) {
    auto allocs = state.range(0);
    for ([[maybe_unused]] auto _: state) {
        auto dd = std::make_unique<dd::Package<>>(1);
        for (int i = 0; i < allocs; ++i) {
            auto* p = dd->mUniqueTable.getNode();
            benchmark::DoNotOptimize(p);
            if (i % 5 == 0) {
                dd->mUniqueTable.returnNode(p);
            }
        }
    }
}
BENCHMARK(bmMUniqueTableGet)->Unit(benchmark::kMillisecond)->RangeMultiplier(10)->Range(10, 10000000); // NOLINT

static void bmMUniqueTableGetAndReturn(benchmark::State& state) {
    auto allocs = state.range(0);
    for ([[maybe_unused]] auto _: state) {
        auto dd = std::make_unique<dd::Package<>>(1);
        for (int i = 0; i < allocs; ++i) {
            auto* p = dd->mUniqueTable.getNode();
            benchmark::DoNotOptimize(p);
            if (i % 5 == 0) {
                dd->mUniqueTable.returnNode(p);
            }
        }
    }
}
BENCHMARK(bmMUniqueTableGetAndReturn)->Unit(benchmark::kMillisecond)->RangeMultiplier(10)->Range(10, 10000000); // NOLINT
