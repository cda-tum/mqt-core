# Benchmark audit and consolidation

Status: complete. Baseline: PR #2788 at
`c7a73a904827690969514bc39ac7fdbb8561bf60`.

## Scope

Audit the typed benchmark families, native evaluation, strict JSON registry,
Python bindings, structured MLIR generation, CLI, documentation, and tests.
Preserve public APIs, case identities, diagnostics, input validation, numerical
limits, and circuit semantics. Stack changes on PR #2788.

Assume local generation and evaluation of benchmark instances, including large
structured programs and sparse sampled counts. This is not an execution service.

## Outcome

The [audit](../audits/benchmarks.md) records two numerical fixes, four focused
consolidations, validation, and rejected candidates. Native, MLIR, Python, CLI,
stub, and lint checks passed. Public schemas and generated Python signatures are
unchanged. Production code is 489 lines smaller with no new dependency.

## Decisions

Family-specific APIs and analytic models remain explicit. Consolidation must
remove a repeated contract or maintenance burden without a new generic
framework. The QAOA template retains named angle arrays, structured loops, and
alternating cycle-edge groups from PR #2788.
