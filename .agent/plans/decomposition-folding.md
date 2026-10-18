# Decomposition and constant folding

Status: complete.

## Scope and decisions

Fix deep-expression stack exhaustion, custom-basis gate-count selection, and
repeated work in constant folding and gate synthesis. Preserve numerical,
global-phase, wire, and failure contracts.

- `mlir::mqt::valueToConstantAttr` evaluates a pure SSA expression DAG with an
  explicit stack. Complete frames immediately; a failed operand fails every
  pending ancestor. Retain successful siblings and identity folds. Bare block
  arguments and literals bypass cache allocation. The explicit cache is now
  `DenseMap<Value, Attribute>`: absent entries are unevaluated, null attributes
  record failure, and nonnull attributes record success. Discard it when the IR
  changes. One-argument overloads and optional return types are unchanged.
- Custom two-qubit bases support zero or one use unless super-controlled. Rank
  only supported counts and use all three basis coordinates for one-use
  fidelity. Skip super-controlled templates when they cannot be used.
- Count-only native queries share Weyl validation and count selection with full
  synthesis. Fixed and parameterized Pauli bases avoid local-factor
  construction. Square-root iSWAP retains full synthesis because its residual
  and sandwich decompositions can fail after target validation.
- Reserve the exact Pauli factor count before inserting the first pair. Reuse
  four fixed square-root iSWAP prefix decompositions; their first
  diagonalization attempt does not use the caller's seed. Target, residual, and
  sandwich decompositions still receive that seed.
- Symbolic fusion reuses normalized parameters within each emitted chain.
  `createOrFold` does not eliminate all common expressions. The cache stays
  local to the chain so cached values dominate their uses. Normalize accumulated
  phase after every addition without rewalking its expression graph; delaying
  normalization would change large-angle numerical behavior.
- Consolidate test setup and matrix oracles while retaining numerical
  boundaries, failure cases, full global phase, controlled inputs, borrowed-wire
  states, and resource limits. Remove frozen Euler tuples only where full
  matrices and required output forms already establish the result.

## Validation

All 987 tests passed across the MQT utility and transform, decomposition,
optimization, target-synthesis, and compiler binaries. Build these targets with
the release preset and run their binaries under `build/release/mlir/unittests/`:
`mqt-core-mlir-unittests-mqt-utils`, `mqt-core-mlir-unittests-mqt-transforms`,
`mqt-core-mlir-unittest-decomposition`, `mqt-core-mlir-unittest-optimizations`,
`mqt-core-mlir-unittest-target-synthesis`, and
`mqt-core-mlir-unittests-compiler`.

The 100,000-operation CLI expression succeeds with an 8 MiB stack. All 60,000
numerical decompositions passed phase-aware reconstruction; worst infidelity was
8.57e-13. Count-only results match full synthesis on 2,003 inputs across three
entanglers, including numerical failure. Eight fresh CLI processes produced
identical output. Independent review found no actionable defects.
`uvx nox -s lint` and whole-file `uvx nox -s cpp-lint` passed.

On ARM64 with Clang 23.1.4, LLVM/MLIR 23.1.0, and release ThinLTO:

- A bare runtime argument makes no folding-cache allocation. A 100,001-entry
  cache retains 4,227,072 bytes instead of 6,324,224 bytes. Short dynamic-query
  CPU time fell from 1.36 to 0.69 microseconds in the standalone probe.
- Generic Pauli factors use one 512-byte allocation instead of three requests
  totaling 1,600 bytes. CX and RZZ count-only queries allocate no factors.
- Five-round numerical probes measured about 21% less CPU for a three-gate
  square-root iSWAP decomposition and about 110 nanoseconds saved per custom RZZ
  basis creation.
- Three-round CLI probes for 1,024, 2,048, 4,096, and 8,192 runtime U gates took
  0.12, 0.28, 0.60, and 1.29 CPU seconds. Recorded baselines were 0.34, 1.13,
  4.11, and 16.42 seconds. The command runs
  `builtin.module(merge-single-qubit-rotation-gates)` and includes parsing,
  verification, and output.

These measurements use CPU affinity, not reserved cores. They establish costs
for the named routines and inputs, not whole target-pipeline speedups.
Allocation counts, retained map storage, and peak process memory are distinct
measures. Probes and raw measurements remain outside the repository. Sanitizers,
other architectures, and hosted CI were not run locally.
