# QAOA benchmark

Status: complete. The benchmark generates a reusable symbolic circuit with array
binding, execution, and serialization coverage.

## Goal and scope

Add the fixed-repetition QAOA family from the jeff benchmark table. The instance
is unweighted MaxCut on an `n`-vertex cycle, with `n >= 3` and depth `d >= 1`.
The generated function takes two arrays of `d` cost and mixer angles. Each run
measures all `n` qubits. Angle optimization and arbitrary input graphs are
outside this addition.

## Definition and source

[Farhi, Goldstone, and Gutmann](https://arxiv.org/html/1411.4028), equations
(2)–(6), define a uniform initial state followed by cost and mixer layers:

```text
|ψ⟩ = U_B(β_d) U_C(γ_d) … U_B(β_1) U_C(γ_1) |+⟩^n
U_C(γ) = exp(-i γ C), U_B(β) = exp(-i β Σ_j X_j)
```

Their equation (12) gives each MaxCut term as `(1 - Z_j Z_k)/2`. Section IV
specializes this objective to a cycle; its maximum is `n` for even `n` and
`n - 1` for odd `n`. Prescribed angles define the quantum workload, not an
optimization loop or a promise of an optimal cut. Section II's local-light-cone
calculation is a way to obtain an expectation, not a full outcome distribution.

## Implementation decisions

- Use the existing `RZZ(-gamma)` operation for each cycle edge; it differs from
  `exp(-i gamma C_jk)` only by a global phase. Each mixer is `Rx(2 beta)`. Apply
  even-start edges, then odd-start edges, then the closing edge `(n - 1, 0)`
  once. This gives cost-layer dependency depth two for even cycles and three for
  odd cycles without expanding the structured loops.
- Configure `qubits` and `depth` (default one). The inputs are `gammas` and
  `betas`, each a static one-dimensional `f64` tensor of length `depth`. Reuse
  shared array parameter binding and its length and finite-value checks. Both
  widths and depths fit signed 64-bit dimensions.
- Index the angle arrays in the layer loop. The benchmark manifest identifies
  the circuit template, not an angle assignment. Record bound angles separately
  to reproduce a run. Bind a copy and run cleanup before sampling, OpenQASM
  export, or QIR export. Binding creates dense constants. Store reusable named
  templates as QC and bind angles before binary jeff serialization.
- Follow the Shor verification pattern: compute the mean sampled cut and its
  ratio to the exact cycle optimum. These scores measure objective quality; they
  do not verify the quantum distribution. Reuse shared count and outcome
  validation. Avoid integer bitstring conversion and integer multiplication of
  shot counts by cut sizes.
- Do not add a statevector simulator or an artificial small qubit/depth limit to
  the native reference. Structured register and layer loops keep the operation
  count and the two function arguments independent of the number of vertices and
  layers. Concrete angle data requires storage proportional to the depth. No
  constant hardware-depth claim is made for the emitted decomposition.
- Keep the dry class summary and detailed mathematical explanation in their
  existing documentation layers. Reuse registration and JSON machinery; keep
  family-specific checks in the family tests. Regenerate Python stubs.

The family-specific evaluation uses the JSON serializer pattern established by
`ShorEvaluation`. Layer-indexed tensor extraction follows Shor's table-lookup
pattern. Measure after the layer loop to avoid a conditional in every mixer
iteration. No new helper, released API change, or generated jeff file is needed.

## Validation

- Small generated coherent states match an independent mathematical oracle:
  apply `exp(-i gamma * cut(z))` directly to amplitudes, then the single-qubit
  mixer matrix. Cover triangles, odd/even cycles, and one or three unequal
  layers, including negative angles outside one period. This catches sign,
  factor-of-two, closing-edge, and layer-order mistakes. The one-layer cases
  also check cost-gate dependency depth after unrolling. Python sampling covers
  zero-angle layers and binding copies of a reusable template.
- As a separate derived check, for one layer and `n >= 4`, the expected cut per
  edge is `1/2 + sin(4 beta) sin(gamma) cos(gamma)/2`. At `gamma = pi/4` and
  `beta = pi/8` this is `3/4`. A direct small-state calculation checked
  `n = 4, 5, 6`; the triangle instead gives `5/8` at these angles and must not
  use the same formula.
- Native checks cover scoring, count validation, JSON round trips, schema, and
  diagnostic locations. A large-count regression covers rounding on platforms
  where `long double` has double precision. Mean scores are bounded by the exact
  cycle optimum before computing the approximation ratio.
- A 1,024-qubit instance with two or one billion layers passes the jeff
  round-trip. Its operation-count bound guards compact generation in both
  dimensions, without attempting a full statevector.
- The benchmark suites pass after the edge-ordering refactor: 160 native tests
  and 52 MLIR tests, plus the three QAOA Python tests on Python 3.14. The six
  coherent-state cases cover one and three layers on three, four, and five
  qubits. Whole-file clang-tidy passes for the three changed C++ files.
  Repository lint passes with Python 3.14 for hook environments.
- Binary jeff preserves argument positions and types, but not input names. Name
  preservation is outside this change; named binding works before export.
- Bound two-layer QAOA exports to OpenQASM 3 and Adaptive QIR. Direct Base QIR
  export rejects the remaining structured loops; array binding does not change
  that profile restriction.
- Hosted CI and execution of large instances were not run. Large generated
  programs remain subject to compiler and executor resource limits.
