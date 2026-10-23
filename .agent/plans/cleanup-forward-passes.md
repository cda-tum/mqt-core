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

`SimplifyClassicalControl` retains facts across effect-free regions. Stores
invalidate facts for other potentially aliased registers; unknown effects
invalidate all facts. Memory facts remain block-local, and branch facts stay
scoped. Blocks without scalar loads skip memory tracking. Dynamic rotation
merging remains restricted to bounded constant angles; the historical RX
dominance fix is superseded by that existing restriction.

The implementations live under `mlir/lib/Dialect/QTensor/Transforms`,
`mlir/lib/Dialect/QCO/Transforms/Optimizations`, and
`mlir/lib/Dialect/MQT/Transforms`. `mlir/lib/Support/Passes.cpp` owns
scheduling; the TableGen pass descriptions own supported scope. Tests in the
corresponding `mlir/unittests/Dialect` trees cover forwarding, repeated writes,
metadata, computed/dynamic indices, block/graph boundaries, aliases, unknown
effects, nested writes, and cleanup after loop inlining.

Retained tensor accesses move in place. Each chain index is evaluated once, and
dead tensor results are erased without unnecessary rewiring. The final
local-fold scan runs only when preceding rewrites changed the block. An
unchanged block cannot expose another local round trip. The private chain record
has internal linkage and uses designated initialization, addressing the two
clang-tidy diagnostics reported on the PR.

## Validation and performance evidence

Final code is `9f63c53b2`, based on main `422a69c1a`. References are the
original PR `764fc2383` and the earlier overhead optimization `563720aed`. The
complete MLIR suite registers 2966 tests: 2964 pass and two `/dev/full` checks
skip on macOS. Run the native suite with `ctest --preset release` after a
Release build. Whole-file clang-tidy 23.1.3 checks all seven changed C++ files
without findings; repository lint passes. These are local checks; hosted CI was
not monitored.

The original broader Benchpress comparison used six fresh processes per build,
258 cases, seven measured calls and one warmup per case. All 4644 executions
passed. Geometric-mean ratios were 1.007x versus main and 1.001x versus the
original PR, with whole-process resampling intervals including no gain. Outputs
matched the original PR exactly. Main differed only by independent instruction
ordering in 18 cases; per-wire traces, phase, and quality matched.

The exact final code has a fresh four-pair comparison against `563720aed` on the
same 258 cases. All 2064 executions pass with exact instruction snapshots and
matching quality. The ratio is 1.005x, with a diagnostic whole-process
resampling interval of 1.003–1.016x. Summed per-case medians are 8.696 s before
and 8.658 s after. All four paired-process aggregate ratios favor the final
code. This is a small improvement over `563720aed` on this host, and does not
establish a broad speedup over main. The interval is diagnostic, with only four
process pairs.

Final timings require three quiet process samples ten seconds apart and monitor
every three seconds. Competing compiler/linter CPU above 5%, or
Python/pytest/MQT CPU above 80% in consecutive samples, rejects the attempt.
Interrupted data stay in the evidence bundle; retries do not depend on timing
values. OS/desktop CPU activity is recorded. Timing does not overlap this task's
builds, tests, or lint.

An initial selected follow-up suggested 3–5% small-case regressions. A longer
recheck of five selected cases uses six counterbalanced process triplets, 101
measured calls, and an identical-wheel control. All 90 executions pass with
exact outputs. Ratios on the three suspected regressions are within 1% of
parity; all diagnostic intervals include no change, and the identical-wheel
control varies by a similar amount. The earlier losses are not reproduced.

The final wheel also passes 128 seeded random full-unitary checks and 64
deterministic feedback/reset checks with nested writes. Maximum matrix error is
8.01e-16 at tolerance 2e-12; dynamic cases use 32 shots. All 192 cleanup
operation counts match `563720aed`.

## Cleanup and workload sensitivity

Fresh native cleanup measurements use nine processes per build, with seven
measured modules and one warmup per process. All builds use Release, unity,
`ENABLE_IPO=OFF`, Apple clang 21, and LLVM/MLIR 23.1.0 with assertions enabled.
The SDK appends `-UNDEBUG`, so intermediate verification remains enabled. All
three references use matching settings, including LTO disabled through
`ENABLE_IPO=OFF`.

| Imported circuit         | Main (ms) | Original PR (ms) | Final (ms) |
| ------------------------ | --------: | ---------------: | ---------: |
| QFT, 32 qubits           |    11.556 |           10.784 |     10.809 |
| Eight layers, 64 qubits  |    15.396 |           13.916 |     13.451 |
| Fresh resets, 128 qubits |     8.861 |            3.817 |      3.785 |
| Feedback, 2 rounds       |     3.496 |            3.230 |      3.193 |
| Feedback, 8 rounds       |    16.277 |           11.757 |     11.512 |
| Feedback, 32 rounds      |   111.894 |           49.620 |     50.128 |

Every native output passes ordinary verification and QCO linearity checks.
Quantum, reset, and tensor-access counts match main; all operation counts match
the original PR. Feedback removes one redundant scalar classical load.

An external seven-process diagnostic isolates normalization without intermediate
verification in the timer, then independently verifies outputs. Speedups over
`563720aed` on two-, eight-, and 32-round feedback are 1.086x, 1.076x, and
1.097x. Production verification policy is unchanged. Prior pass profiles show
about 7.5% less classical simplification time, saving only about 20 ms in
initial cleanup across a roughly nine-second QASMBench compilation set. Ordinary
verification and later compiler stages limit the overall gain. The 69 QASMBench
sources have no scalar classical loads and only five redundant resets among 70
resets.

The six imported cleanup inputs also run through the existing Benchpress gym
compile helpers in a separate three-process-per-build subset. All 54 executions
pass native-basis/topology validation. Quantum gate counts and applicable depths
match. Original-PR and final snapshots match exactly. Main has two feedback
branches where the PR has one; a strict single-measurement checker exhausts both
outcomes and compares exact remapped traces and phase. Static cases use the
original per-wire check. This small timing subset varies across processes,
especially for QFT, and does not establish a gain on every input.

Three alternatives were tested and rejected: root discovery without a visited
set slowed feedback, an early region guard gave no consistent gain, and direct
constant matching slowed QFT, layered, and reset normalization. Ad hoc sources,
inputs, manifests, raw timings, snapshots, and comparison scripts stay in the
task-local benchmark bundle outside the repository. The broader Large suite,
other gyms, and assertions-disabled SDK builds remain outside the measurements.
