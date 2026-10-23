# Compact array initialization

Status: complete.

## Scope and decisions

OpenQASM and jeff numeric literals above 16 elements use an `arith.constant`
dense tensor and `bufferization.materialize_in_destination` into existing
mutable storage. DD, QIR, jeff, and OpenQASM consumers preserve this compact
representation. Small arrays retain scalar initialization for LLVM/MLIR 23.1
memory promotion. Runtime expressions and unsupported attribute conversions
retain scalar emission and evaluation order.

Initialization writes whole local storage. OpenQASM export requires it to
precede every use of an entry-block allocation and emits a declaration
initializer, which preserves definite initialization on reimport.

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
weighted checksum detects reordered elements that an unweighted sum misses.

Historical measurements of `1c214f629d` against `04de176835`, using the same
Release toolchain and seven alternating process pairs, reduced the imported IR
for 16,384 varying literals from 65,568 operations to 34. Median QIR-path CPU
time fell from 543 to 16 ms and allocation volume from 975 to 29 MB. These are
workload-specific results on a shared heterogeneous host. Literal payload and
mutable storage still scale with element count. Raw evidence stays outside the
repository.

Local validation on main `0e0826b1bf` passes: 4,081 tests with
`ctest --preset release`, 2,021 configured Python tests, whole-file C++ lint,
repository lint, and executable documentation with internal link checks.
Rendered OpenQASM documentation was inspected. Hosted CI and cross-platform
validation remain separate from these local checks.
