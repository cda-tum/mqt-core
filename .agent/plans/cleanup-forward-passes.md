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

The retained accesses now move in place. Chain indices are evaluated once, and
dead tensor results are erased without unnecessary rewiring. Classical memory
tracking runs only in blocks with scalar loads; branch facts remain scoped and
nested regions keep their own memory maps.

## Validation and performance evidence

The optimized code is `563720aed`, compared with main `422a69c1a` and the
previous PR `764fc2383`. The complete MLIR suite registers 2966 tests: 2964 pass
and two `/dev/full` checks skip on macOS. Whole-file C++ lint and repository
lint pass. New tests cover independent interleaved tensor chains, repeated
stores, and condition propagation through nested regions.

The matched Benchpress run covers 258 cases in six fresh processes per build,
seven measured compilations and one warmup per case. All 4644 test executions
pass. The optimized outputs exactly match the previous PR's instruction
snapshots in all cases. Main differs only by independent instruction ordering in
18 cases; all per-wire traces and quality metrics match.

The geometric-mean speedups are 1.007x versus main and 1.001x versus the
previous PR. Whole-process resampling intervals include no gain. Total
compilation is effectively flat. Native profiles show about 7.5% less time in
classical simplification, about 1% less in normalization, and about 2% less in
the initial cleanup block. The latter saves only about 20 ms across all 258
cases.

Independent checks on both revisions cover 128 seeded random full-unitary
matrices and 64 deterministic feedback/reset circuits with writes inside
branches. All pass; the largest matrix error is 8.01e-16 at tolerance 2e-12.

Fresh standalone cleanup measurements use nine processes per build, each with
one warmup and seven measured modules. All three builds use Release, unity,
`ENABLE_IPO=OFF`, Apple clang 21, and LLVM/MLIR 23.1.0 with assertions enabled.
The SDK adds `-UNDEBUG`, so intermediate pass verification remains enabled.
Earlier metadata incorrectly stated both that LTO and intermediate verification
were disabled. Setting only `CMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF` did not
override `ENABLE_IPO`; the previous standalone reference has been rebuilt. The
Python wheels already used matching LTO-off settings.

| Imported circuit         | Main (ms) | Previous PR (ms) | Optimized (ms) |
| ------------------------ | --------: | ---------------: | -------------: |
| QFT, 32 qubits           |    11.543 |           11.093 |         10.857 |
| Eight layers, 64 qubits  |    15.159 |           14.156 |         13.912 |
| Fresh resets, 128 qubits |     9.142 |            3.899 |          3.842 |
| Feedback, 2 rounds       |     3.639 |            3.341 |          3.212 |
| Feedback, 8 rounds       |    16.332 |           12.097 |         11.601 |
| Feedback, 32 rounds      |   113.300 |           50.292 |         50.170 |

Operation counts and linearity checks pass for every sample. The larger reset
and feedback gains remain. An external diagnostic with intermediate ordinary
verification outside the timer shows about 17% faster normalization for QFT and
layered circuits and 72% for resets. Ordinary verification contributes roughly
54–89% of normalization-pass elapsed time on these inputs. Production
verification policy is unchanged. Ad hoc scripts and raw data remain outside the
repository in the task's local result bundle.

An independent selected follow-up passes all 66 executions on 11 small cases,
with matching quality and wire traces. Several cases retain about 3–5% slowdowns
versus the previous PR, including `ghz_state_n23-all-to-all`,
`bell_n4-all-to-all`, and `qec_sm_n5-linear`. The latter partly loses its
previously reproducible gain. These changes are not an improvement on every
input; the overall Benchpress result remains flat.
