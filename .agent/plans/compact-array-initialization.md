# Compact array initialization

Status: complete.

## Scope and decisions

OpenQASM and jeff numeric literals above 16 elements use an `arith.constant`
dense tensor and `bufferization.materialize_in_destination` into existing
mutable storage. DD, QIR, jeff, and OpenQASM consumers preserve this compact
representation. Small arrays retain scalar initialization for LLVM/MLIR 23.1
memory promotion. Runtime expressions and unsupported attribute conversions
retain scalar emission and evaluation order.

Initialization writes whole local storage. DD reuses uniquely owned buffers and
replaces shared branch buffers without copying overwritten values; aliases
retain storage identity. OpenQASM export requires initialization to precede
every use of an entry-block allocation and emits a declaration initializer,
which preserves definite initialization on reimport.

Equal literals share immutable data. The jeff importer rematerializes numeric
literal operations when mutating control flow needs independent storage; live
computed arrays retain their existing restriction. Export removes unused zero
placeholders only when static shape and constant length match. The pinned jeff
dependency lacks the effects interfaces needed for generic dead-code removal.

Existing MLIR facilities suffice; no custom initialization operation, pooling
pass, general constant evaluator, or broader ownership analysis is needed.

## Validation

`CompilerPipelineTest.ConstantArrayInitializationPreservesValues` checks narrow
integers, row-major ordering, signed zero, and independent mutable arrays across
branches through DD, jeff serialization, OpenQASM reimport, and native QIR. The
regression detected shared-literal storage defects before their fix; its
weighted checksum detects reordered elements that an unweighted sum misses. The
existing branch/alias test also covers initialization after measurement;
removing the buffer-sharing guard makes its outcome check fail.

Historical measurements of `1c214f629d` against `04de176835`, using the same
Release toolchain and seven alternating process pairs, reduced the imported IR
for 16,384 varying literals from 65,568 operations to 34. Median QIR-path CPU
time fell from 543 to 16 ms and allocation volume from 975 to 29 MB. These are
workload-specific results on a shared heterogeneous host. Literal payload and
mutable storage still scale with element count. Raw evidence stays outside the
repository.

Follow-up measurements against `d88f5b5820` confirm one fewer 128 KiB buffer
allocation in the 16,384-element DD probe. Peak live heap remains 16.53 MB.
Fifteen alternating pairs on one core across nine public routes give median
CPU-time ratios of 0.994–1.011; no runtime speedup is established. The existing
jeff normalization and placeholder cleanup remain necessary for the pinned
dependency's rank, integer-width, and memory-effect contracts.

Local validation on main `0e0826b1bf` passes: 4,081 tests with
`ctest --preset release`, 2,038 Python tests with `-m 'not slurm'`, whole-file
C++ lint, and repository lint. Existing user documentation describes the
unchanged behavior and limits. Hosted CI and cross-platform validation remain
separate from these local checks.
