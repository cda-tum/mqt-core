# Native entangler selection and symbolic square-root iSWAP

Status: implemented, reviewed, and validated locally. Hosted CI and human review
remain pending in PR #2775. Base: `056bef24722dd0a1f0813f3f7b4197c10f8eb950`.

## Scope and decisions

Recognize a square-root iSWAP declared as `xx_plus_yy` with parameters
`(-pi/2, 0)`, fuse commuting runtime Pauli rotations with this gate, and choose
native entanglers by two-qubit gate count. Preserve numerical and phase checks,
ordered placements, deterministic ties, and existing native operations.

- `CompilerTarget` owns capability recognition and the ordered set of entanglers
  usable on every coupling. Keep its existing single-qubit basis and default
  entangler API. Use the widest supported angle domain for each gate.
- `NativeCostAnalysis` owns selection for routing, individual synthesis, and
  constant-run fusion. Choose the cheapest supported homogeneous recipe; keep
  the established preference order on ties. This does not search circuits that
  mix entangler families or optimize hardware duration or error.
- Cache validated Weyl coordinates by matrix and seed, independent of the
  entangler. Keep failure entries and operand orientations distinct. Full
  synthesis still validates construction of native factors.
- Runtime Pauli fusion compares complete run costs. The two-generator
  square-root iSWAP circuit uses two gates. Three independent runtime generators
  use four; a full symbolic Cartan solver is outside this change. Binding first
  enables the numerical solver's three-gate bound.
- Use exact principal-branch identities for local frames and scaled square-root
  expressions for small angles. The formulas use the arithmetic and
  trigonometric operations supported by symbolic exporters. OpenQASM exports
  floating-point absolute value through its existing scalar expression path.

## Validation

- All 1,383 tests pass across the affected C++ utility, transformation,
  decomposition, synthesis, compiler, OpenQASM, and jeff round-trip suites.
- All 377 MLIR Python and Qiskit-target tests pass on Qiskit 2.6.0. The 27 runtime
  synthesis and composition cases also pass on Qiskit 2.5.1. The staged package
  uses the build's QDMI driver through `MQT_CORE_QDMI_DRIVER`.
- The symbolic pair formula passes 24,461 numerical samples and 1,121
  export-then-bind samples on both Qiskit versions, including zero, tiny,
  boundary, and large angles. Worst matrix errors are below `1e-15`.
- Repository lint and whole-file C++ lint pass. Builds use Clang 23 with
  `CCACHE_DISABLE=1` because the local compiler cache produced invalid output.
- Independent numerical and design reviews found no remaining actionable
  issues. The final review removed a redundant copy of cached fusion factors.

## Measurements

An external probe checked 20,000 deterministic locally dressed canonical
matrices with one and nine candidate families. Counts and tie selection matched
independent per-family analysis. Five alternating rounds pinned to CPU 0 gave
median CPU times of 115.68 ms for nine-family selection and 796.57 ms for repeated
per-family analysis. One-family selection took 97.75 ms versus 93.26 ms. The
coordinate cache made two allocations totaling 18,944 bytes, independent of
family count; warm queries allocated nothing. These are affinity-only routine
measurements, not whole-compiler speedups. The shared and local numerical
payloads remain bounded at about 288 KiB and 18 KiB.
