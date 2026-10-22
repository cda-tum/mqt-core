# Forward QCO cleanup

Status: complete. QCO cleanup uses explicit QTensor normalization and forward
reset analysis, with alias-safe classical forwarding across effect-free regions.

## Outcome and scope

`qtensor-normalize` folds extract/reinsert identities and batches constant-index
accesses within SSA blocks. It forwards pending scalar writes, preserves
repeated writes and retained metadata, and stops at dynamic indices, unknown
users, and block boundaries. `remove-redundant-resets` tracks zero scalar values
and fresh untouched tensor slots. Both passes skip graph regions and preserve
analyses when unchanged.

QCO cleanup schedules both passes before its existing canonicalizer. Tensor
canonicalizers remain necessary for accesses exposed later by control-flow
simplification. Replacing those fallbacks or adding a preparatory canonicalizer
would add work without a demonstrated benefit.

The existing `SimplifyClassicalControl` retains facts across effect-free
regions. Stores invalidate facts for other potentially aliased register values;
unknown memory effects invalidate all facts. Facts do not propagate into nested
blocks. Dynamic rotation merging remains restricted to bounded constant angles;
the historical dominance fix is superseded by that existing restriction.

The implementations live under `mlir/lib/Dialect/{QTensor,QCO,MQT}/Transforms`,
with registration and scheduling in `mlir/lib/Support/Passes.cpp`. Their
TableGen pass descriptions own the supported scope and limitations. Tests in the
owning subsystems cover wire forwarding, repeated writes, metadata, idempotence,
computed and dynamic indices, block and graph boundaries, aliases, unknown
memory effects, and cleanup after loop inlining.

## Validation and performance evidence

The release build and MLIR reference generation pass with LLVM/MLIR 23.1.0. The
complete MLIR suite registers 2963 tests: 2961 pass and two device-failure
checks skip on macOS. Final focused QTensor and reset tests also pass.
Whole-file C++ lint uses clang-tidy 23.1.3 on all seven changed C++ files;
repository lint passes.

The baseline is main at `422a69c1a`. Measurements use Apple M1 Pro, macOS
26.6.2, Apple clang 21, Release without LTO, and the same LLVM/MLIR SDK. They
time `runQCOCleanupPipeline`, including pass-manager setup and its final
verifier. Parsing, module cloning, and additional output verification/linearity
checks are outside the timer. Release disables intermediate verification in both
binaries. Seven independent processes per binary each warm up once and measure
seven fresh modules, alternating baseline/candidate order. Reported times are
the median of the seven process medians.

| Imported circuit             | Baseline (ms) | Forward cleanup (ms) | Speedup |
| ---------------------------- | ------------: | -------------------: | ------: |
| QFT, 32 qubits               |        11.497 |               11.125 |   1.03x |
| Eight gate layers, 64 qubits |        15.352 |               14.307 |   1.07x |
| Fresh resets, 128 qubits     |         9.185 |                3.837 |   2.39x |
| Feedback, two rounds         |         3.684 |                3.162 |   1.16x |
| Feedback, eight rounds       |        15.480 |               11.718 |   1.32x |
| Feedback, 32 rounds          |       106.430 |               49.259 |   2.16x |

All six outputs preserve quantum-operation, reset, and tensor-access counts.
Feedback has one fewer classical load. Counts are a quality check, not an
independent equivalence proof; wire and memory semantics have direct regression
tests. These synthetic imported circuits measure cleanup, not total compilation
or execution. Small gains are subject to host noise. Ad hoc harnesses and raw
samples stay outside the repository; the PR records the reproduction procedure.
