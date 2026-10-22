# Benchmark audit

Status: applied and validated locally. Date: 2026-10-22. Baseline: PR #2788 at
`c7a73a904827690969514bc39ac7fdbb8561bf60`. The starting worktree was clean.

## Scope and result

`mqt.core.bench` defines validated benchmark instances, emits structured quantum
programs, and evaluates sampled results. This audit assumes local generation and
sparse counts, including the documented large-width limits. It does not assume
an execution service or benchmark hardware performance.

The audit covered all 16 families in `include/mqt-core/bench`, `src/bench`,
`bindings/bench`, and `mlir/bench`; their native, MLIR, and Python tests;
generated Python stubs; the CLI launcher; build registration; and benchmark
documentation. It traced instance specifications through validation, manifests,
generation, serialization, and evaluation. PR #2788's array inputs, cycle-edge
grouping, and independent QAOA state oracle are included in the baseline.

Two numerical defects and four useful consolidation opportunities were found and
fixed. Production code is 489 lines smaller, including the new helper; no
dependency was added. No additional generation defect was confirmed. Public
types, signatures, strict parsing, canonical schemas, semantic case IDs, and
circuit definitions remain unchanged.

## Findings

### 1. Preserve small multiplexer probabilities

Applied in `src/bench/Multiplexer.cpp`, `Multiplexer::probability`.

The old formula accumulated a control fraction near one, then evaluated a cosine
near its zero. For 64 all-one control bits and target zero, it returned
`2.0325535182104675e-52` instead of `3.9307967343631768e-58`. Both values are
representable; an absolute-error test would miss this large relative error.

Compute target-zero amplitudes from the complemented binary fraction and use one
sine formula for both targets. For `c` control bits,
`1 - sum(b_i * 2^-i) = 2^-c + sum((1-b_i) * 2^-i)`. This avoids subtraction near
one and removes the sine/cosine branch. The small absolute probabilities limit
the practical impact, but the reference API promises these supported widths. Its
mathematical model and case identity do not change.

`Multiplexer.PreservesSmallProbabilitiesNearBothEndpoints` checks relative error
at both endpoints. The exact regression fails against baseline sources and
passes against the corrected sources.

### 2. Avoid premature underflow in shared fidelity evaluation

Applied in `src/bench/EvaluationUtils.hpp`, `detail::evaluate`.

`sqrt(observed * ideal)` can lose the product before taking its square root. On
this ARM host, `long double` has the same 53-bit mantissa as `double`. Two
equally sampled, valid even-parity outcomes of a 1075-qubit X-basis GHZ
benchmark produced zero fidelity instead of the representable `2 * denorm_min`.

Use `sqrt(observed) * sqrt(ideal)`. The metric and supported inputs are
unchanged. Results below the final `double` range may still round to zero.
`GHZ.PreservesRepresentableFidelityForSubnormalProbabilities` uses exact
equality so a floating-point tolerance cannot accept zero. It fails against the
baseline on this host and passes with the fix.

### 3. Share the repeated Python generation and JSON bindings

Applied in `bindings/bench/register_benchmark.hpp` and all 16 family bindings.

Each family repeated the same six methods: generation, three metadata
properties, and two strict JSON parsers. A change to that shared contract
previously required 16 matching edits. One private template now binds those
methods; each family supplies its existing typed parsers.

The helper retains the lazy MLIR import, signatures, defaults, method order,
return types, and docstrings. Constructors, options, probability functions, and
evaluation methods stay explicit because their contracts differ. The QPE custom
constructor uses C++20 `std::construct_at`.

Existing family-specific tests cover Python serialization. A separate all-family
matrix was removed because it duplicated those tests and required another family
list to maintain. BV and Grover test only manifest parsing in Python; QFT and
QPE test only instance-specification parsing. Python source forwarding and error
translation have no dedicated test. Native tests cover both parser paths and
source-qualified errors. Regenerated stubs have no API diff.

### 4. Use the family registry for internal emitter declarations

Applied in `mlir/bench/programs/Programs.h`.

The header repeated all family names and emitter declarations already owned by
`BenchmarkFamilies.inc`. Expanding the existing registry removes that second
registration point, including the new QAOA entry. Public `generate` declarations
and their documentation remain explicit.

A narrow lint exception covers class names in the declaration macro, which
cannot be parenthesized. The contributor guide now describes registry-owned
emitter declarations and the shared binding helper.

The audit also added asymmetric BV sampling cases for both circuit methods. The
previous generation fixtures used a palindromic hidden string, which could not
expose reversed bit order. Existing structural checks remain intact.

### 5. Reuse the QFT phase recurrence

Applied in `mlir/bench/programs/QFTUtils.cpp`, `phaseAdditionLoop`.

The loop separately emitted `phase/2 + pi*bit`, already implemented by
`advancePhaseCorrection`. Calling that helper preserves the floating-point
operations, types, and loop shape without adding an interface. Bit extraction
now precedes the independent halving operation. Existing arithmetic and
phase-sensitive tests cover the consumers.

### 6. Centralize the parameter-object schema constraints

Applied in `src/bench/JSON.cpp`, `baseInstanceSpecificationSchema`.

Every family separately required an object and rejected additional properties.
The existing schema helper now supplies those two shared constraints. All
family-specific fields, defaults, limits, nested objects, and conditional
schemas remain explicit. All 16 serialized schemas match the baseline byte for
byte, including QAOA.

## Consolidations rejected or deferred

- Native outcome validation, shot overflow checks, and distribution metrics
  already share `EvaluationUtils.hpp`. QAOA cut quality and Shor factor
  verification require different evaluation contracts.
- Grover and weak-measurement Grover have similar diffusion code but different
  structured and unrolled forms. A shared gate-list helper would change that
  distinction; no demonstrated benefit justifies the rewrite.
- BV and QFT addition share a small reversed-bit tensor construction pattern.
  Carry extension differs, and no existing helper owns both uses. A new helper
  file would add an interface for too little common code.
- Output-measurement loops encode different register offsets and layouts. Keep
  these explicit rather than adding builder options for a few lines.
- JSON parsing retains direct family-specific validation and diagnostics. A
  generic schema or option framework would add machinery and risk changing
  strictness. Manifest comparison must retain numeric-kind distinctions.
- The CLI's custom sibling-file creation handles literal percent signs in
  directory names. LLVM's unique-file model substitutes every percent sign, so a
  direct replacement would break a covered path. Keep exclusive creation,
  no-overwrite publication, cleanup, and manifest-last ordering.
- Existing independent circuit oracles, numerical limits, and negative cases
  remain. Similar-looking tests across families do not establish redundancy.
- Minor parity-count and QPE zero-case reductions were left out; they do not
  remove a shared maintenance burden.

## Validation and limits

- Release build with LLVM/MLIR 23.1.0 and assertions enabled: 162 native
  benchmark tests, 53 MLIR benchmark tests, and the CLI CTest passed.
- Python 3.14 benchmark suite: 82 tests passed after removing the 16-family
  serialization matrix.
- Both numerical regressions failed against baseline sources and passed against
  corrected sources. An external assertion probe checked 257,052 QPE and
  multiplexer reference values, normalization, numerical endpoints, and shared
  count validation.
- All 16 schemas matched baseline bytes. A fresh-process check confirmed that
  importing benchmarks leaves MLIR unloaded and generation loads it lazily.
- Stub regeneration passed with no generated-file changes. Repository lint and
  explicit lint on new files passed.
- Whole-file C++ lint checked all 25 changed/new C++ files. Two diagnostics were
  corrected, and both affected files passed their rerun.

Reproduce the build and runtime checks with:

```sh
cmake --build --preset release --target mqt-core-bench-test mqt-core-mlir-unittests-benchmark mqt-core-bench
build/release/test/bench/mqt-core-bench-test
build/release/mlir/unittests/bench/mqt-core-mlir-unittests-benchmark
ctest --preset release -R '^mqt-core-mlir-benchmark-cli$'
uv run --no-sync pytest test/python/bench
uvx nox -s stubs
uvx nox -s lint
```

For uncommitted C++ changes, the configured `cpp-linter` was run with
`--files-changed-only=false --lines-changed-only=false` and an explicit
changed-file whitelist, using the `build/cpp-lint` database prepared by the nox
session.

No performance benchmark, hardware execution, other operating system, or full
executable documentation build was run. Compiler backends were exercised by the
benchmark tests, not exhaustively audited.
