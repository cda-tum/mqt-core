# QAOA benchmark

Status: complete. The cycle instance, native and Python interfaces, JSON
registry, documentation, and independent circuit tests are implemented.

## Goal and scope

Add the fixed-repetition QAOA family from the jeff benchmark table. The instance
is unweighted MaxCut on an `n`-vertex cycle, with `n >= 3` and caller-supplied
cost and mixer angles for each layer. Each run measures all `n` qubits. Angle
optimization and arbitrary input graphs are outside this addition.

## Definition and source

[Farhi, Goldstone, and Gutmann](https://arxiv.org/html/1411.4028), equations
(2)–(6), define a uniform initial state followed by cost and mixer layers:

```text
|ψ⟩ = U_B(β_p) U_C(γ_p) … U_B(β_1) U_C(γ_1) |+⟩^n
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
  the closing edge `(n - 1, 0)` once.
- Match `gammas` and `betas` vector lengths; their common nonzero length is the
  repetition count. Use the paper's finite angle domains `[0, 2 pi]` and
  `[0, pi]`. Defaults contain one layer with angles `pi/4` and `pi/8`. Widths
  start at three and fit signed 64-bit circuit dimensions, as for W state.
- Follow the Shor verification pattern: compute the mean sampled cut and its
  ratio to the exact cycle optimum. These scores measure objective quality; they
  do not verify the quantum distribution. Reuse shared count and outcome
  validation. Avoid integer bitstring conversion and integer multiplication of
  shot counts by cut sizes.
- Do not add a statevector simulator or an artificial small qubit/depth limit to
  the native reference. Structured register and layer loops keep the operation
  count independent of the number of vertices and layers. The two constant angle
  tensors still require data proportional to the number of layers. No constant
  hardware-depth claim is made for the emitted decomposition.
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
  mixer matrix. Cover triangles, odd/even cycles, and multiple unequal layers.
  This catches sign, factor-of-two, closing-edge, and layer-order mistakes
  without inspecting the emitted operation tree.
- As a separate derived check, for one layer and `n >= 4`, the expected cut per
  edge is `1/2 + sin(4 beta) sin(gamma) cos(gamma)/2`. At `gamma = pi/4` and
  `beta = pi/8` this is `3/4`. A direct small-state calculation checked
  `n = 4, 5, 6`; the triangle instead gives `5/8` at these angles and must not
  use the same formula.
- Native checks cover scoring, count validation, JSON round trips, schema, and
  diagnostic locations. A large-count regression covers rounding on platforms
  where `long double` has double precision. Mean scores are bounded by the exact
  cycle optimum before computing the approximation ratio.
- A 1,024-qubit instance with two or 128 layers passes the jeff round-trip. Its
  operation-count bound guards compact generation in both dimensions, without
  attempting a full statevector.
- All benchmark suites pass: 163 native tests, 51 MLIR tests, 82 Python tests,
  and the CLI CTest. Python stubs were regenerated. Repository lint and
  whole-file C++ lint pass. The executable documentation build and local-link
  checks pass; the new API formulas are parsed as math.
- Hosted CI and execution of large instances were not run. Large generated
  programs remain subject to compiler and executor resource limits.
