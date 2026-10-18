# Contract audit: finite parameters and global phases

Status: all confirmed findings applied and locally validated. Original baseline:
`9ca32dd3e4df86fc313eca1c326adbf6ec27b7f9`. Implementation base:
`056bef24722dd0a1f0813f3f7b4197c10f8eb950`.

## Scope and contract

The audit follows QC/QCO parameter producers through `Angles.cpp`,
`ConstantFolding.cpp`, `GatePowering.cpp`, modifier canonicalization,
`NormalizeGlobalPhases.cpp`, and OpenQASM, Qiskit, and DD consumers. Non-finite
quantum parameters are invalid. Direct-constant sanity checks and point-of-use
numerical guards remain. Classical floating-point and external-format contracts
are outside this simplification.

## Findings and disposition

### P1: phase extraction through powers changed valid unitaries

Reducing a body phase before scaling can amplify rounding error. The original
`gphase(3)` with exponent `1e16` changed the scalar unitary by 1.10394; exponent
`1e308` produced NaN.

Applied: leave all power subtrees unchanged, including under inverse/control
modifiers. Delete phase scaling and `factorPower`. Modifier canonicalization
retains safe power folding. The pass description states the boundary. Existing
QC/QCO power tests now check exact preservation for dynamic, fractional,
ordinary integer, and huge exponents; full-unitary inverse/control tests remain.

### P2: sanity checks repeatedly traversed shared expressions

`verifyGlobalPhaseAngle` performed fresh recursive folding for every phase.
Shared prefixes caused quadratic work; a valid 64000-operation chain exhausted
the stack during verification and normalization.

Applied: inspect direct constants in the verifier. Use a cached worklist for
actual constant evaluation, memoizing both successes and failures. Stop at the
first unknown operand and avoid cache allocation for a bare block argument.
Existing shared-DAG tests cover deep finite/dynamic roots and both operand
orders. Complete evaluation remains available to composite-gate exporters.

### P2: merged symbolic phases exceeded the supported range

Two valid phases bound to 6000 became 12000, exceeding the QC/QCO phase limit of
10000. Applied: balance sums and reduce newly merged symbolic phases with
`2 * atan(tan(angle / 2))`. Existing exporters support these operations. Known
constants fold directly; lone phases and negations incur no wrapping. Extended
existing tests check large bindings, the pi boundary, bounded output, complex
phase equality, idempotence, and full-unitary consumer round trips.

### P2: normalization repeated folding, moves, and allocations

Independent block caches discarded shared ancestor results. Scalar producers
moved once per enclosing modifier. Derived constants created temporary addition
trees that were folded again, and conditional returns copied phase buffers.

Applied: each isolated scope owns a cache; nested classical regions share it.
Move scalar producers directly before the outermost eager modifier in source
order. Modifier verifiers exclude classical regions, memory effects, and qubit
dependencies, so no recursive dependency search is needed. Materialize constants
once and move released phase buffers. Delete leaf bookkeeping, redundant
folding, and dead-body scans. Existing tests verify one move per original
producer, classical-scope preservation, dependencies, and no dead accumulation
additions.

### P3: redundant validation and cleanup work

Applied: remove repeated input-finiteness predicates from angle arithmetic and
U-gate powering. Keep sum/product overflow, TwoSum recovery, FMA rounding,
exponent eligibility, and matrix reconstruction guards. Even-exponent detection
uses a zero remainder. Invalid-input recovery tests are removed; finite
overflow, precision, reconstruction, and signed/unsigned wide-integer conversion
remain.

Skip modifier dispatch for regionless operations. Reuse modifier inlining to
remove inverses emptied by phase extraction, allowing enclosing empty controls
to disappear too. Existing QC/QCO control-removal tests cover this cleanup and
QCO result-wire identity. Modifier unrolling excludes terminators by iteration
instead of checking their trait on every operation.

### P3: a synthesis test constrained phase-preparation layout

`TargetNativeSynthesis` permits input changes on failure, and its phase
preparation can move eager scalar producers. The unsupported runtime-controlled
R regression compared against the original layout, conflating preparation with
failed lowering. Applied: normalize a clone for the expected result while
running synthesis on the original input. Retain exact IR comparison, the failure
diagnostic, verification, and qubit linearity. No rollback or
conditional-hoisting machinery is needed.

The connected synthesis audit also replaces three modifier-ancestor walks with
MLIR's variadic query. Final phase preparation runs only when a synthesis basis
exists; successful native-only lowering preserves or reorders gates and creates
no phases. Existing wire-reordering, modifier-shell, and retained/dropped-phase
tests cover this path. Keep all preparation stages when synthesis can emit
phases between them.

## Rejected candidates

A folding-depth cap would reject valid composite-gate exports. Changing or
removing the public cache-taking overload would break a header API shipped in
Core 4.0 and 4.1; an adapter solely to reduce cache storage adds needless code.
Remainder-based phase wrapping needs new Qiskit expression support. Keep the
existing exporter-supported expression instead.

## Validation and limits

The [decision record](../plans/finite-parameters.md) contains final local
checks. The surviving test oracles cover numerical limits, complete unitaries,
exporter compatibility, scope, exact power preservation, deep expressions, and
allocation sources. Test consolidation retains these contracts without promising
recovery from unsupported non-finite parameters.

Matched synthetic experiments and original reproducers are kept locally under
`.agent/benchmarks/finite-parameters/`, with raw measurements and exact source
patches. Timings use unreserved CPUs and do not establish application speedups;
allocation requests do not measure peak memory. No actionable finding remains
unimplemented. Hosted CI and other platforms require separate validation.
