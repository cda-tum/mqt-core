# Finite parameters and global phases

Status: complete. Base: upstream main
`3bf8849e5d5b671fe03b168aa38e3357a4bded0c`.

## Outcome and scope

Finite quantum parameters are a precondition. Direct constants receive cheap
sanity checks. Numerical guards remain where finite arithmetic can overflow,
lose phase information, or fail matrix reconstruction. General classical
floating-point behavior and external-format validation are unchanged.

The changes cover shared MQT angle and constant-folding utilities, modifier
canonicalization, and QC/QCO global-phase normalization. The
[contract audit](../audits/finite-parameters-20261017.md) records the findings
and retained consumer contracts. Existing tests are consolidated or extended;
the two MQT suites have 69 cases instead of 83.

## Decisions

- Leave power subtrees unchanged during phase normalization. Large exponents
  amplify phase rounding. Modifier canonicalization owns safe power folding.
- Keep balanced symbolic sums and bound newly merged phases with
  `2 * atan(tan(angle / 2))`. Existing Qiskit, OpenQASM, and DD consumers
  support these operations. Single phases and negations need no wrapping.
- Evaluate constant expressions with a cached worklist. Stop at the first
  unknown operand; retain complete evaluation for constant-dependent exporters.
- Cache scalar results within each isolated scope. Nested classical regions
  share the cache because normalization preserves scalar meanings.
- Hoist eager scalar producers once, in source order, before the outermost
  enclosing inverse or control. Verified modifier bodies exclude classical
  regions, memory effects, and scalar dependencies on qubits. Classical scope
  boundaries and power bodies remain intact.
- Merge known phases before emitting arithmetic, move phase-expression buffers,
  skip recursive dispatch for regionless operations, and remove inverse/control
  wrappers emptied by extraction. Reuse existing modifier inlining helpers.
- Retain the shipped cache-taking constant-folding API. A storage-only change
  does not justify breaking callers or adding an adapter.

## Validation

The final implementation passed 2914 C++ tests in 15 affected binaries using
`release-clang-ipo`, Clang 23.1.4, LLVM/MLIR 23.1.0, ThinLTO, and mold on Linux
aarch64. These cover MQT utilities/transforms, CBit/QC/QCO/QTensor IR, compiler
APIs and CLI, QC/QCO conversions, decomposition/optimization, and OpenQASM.

The Python command

```sh
uvx nox -s tests-3.14 -- test/python/test_mlir.py \
  test/python/test_mlir_parameter_binding.py \
  test/python/test_mlir_qiskit_translation.py \
  test/python/test_mlir_qiskit_target.py -n 4
```

passed all 840 cases with Python 3.14.7 and Qiskit 2.6.0. The 243
phase/symbolic/parameter cases also passed with Qiskit 2.5.2. Both versions
passed six QC/QCO full-unitary binding, Qiskit export, and OpenQASM round-trip
probes. All four original deep-expression and large-power reproducers passed.

`uvx nox -s lint`, `uvx nox -s cpp-lint`, and `git diff --check` passed. C++
lint checked every changed implementation/test file in full. No binding
declarations changed. Implementation and tests are 246 lines smaller: 79
production lines and 167 test lines. Existing numerical and consumer oracles
remain covered.

One-off correctness probes and matched timing experiments live in the ignored
`.agent/benchmarks/finite-parameters/` folder. Each comparison retains its exact
source patch, binary hashes, raw samples, reproduction commands, and plots. CPU
affinity does not reserve host resources; timings are diagnostic and do not
establish application-wide speedups. Against the preceding implementation, 8192
scalar producers at depth 64 fell from a median 14.82 ms to 2.17 ms; a shared
expression across 128 classical regions fell from 280.2 ms to 25.8 ms. The
phase-only control and observed allocation requests were unchanged within sample
spread. Allocation-request counts are not peak or resident memory measurements.

## Limits

Non-finite quantum parameters remain unsupported. Power normalization is owned
by modifier canonicalization. Hosted CI and other platforms are separate from
local validation.
