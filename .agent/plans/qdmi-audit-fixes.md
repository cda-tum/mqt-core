# QDMI contract fixes and simplification

Status: complete.

## Outcome and scope

Target compilation preserves advertised operation names and ordered placements
in native QIR and OpenQASM payloads. Imported scalar outputs remain distinct
from QIR status returns. Provider handles retain ownership through allocation
failures; DDSIM translates exceptions at allocating C entry points, and job IDs
reject embedded NUL bytes.

Qiskit keeps CU3 distinct from CU, merges compatible aliases without losing
placements or calibration, snapshots retained submission parameters, and
disables near-constant rounding for QASM3. The unused private configuration
constructor and redundant test setup are removed. The test diff removes 204
lines overall while retaining ownership, calibration, numerical, concurrency,
and recovery checks.

The affected implementation is in `mlir/lib/Compiler`,
`mlir/lib/Conversion/QCToQIR`, `mlir/lib/Target/OpenQASM`, `src/qdmi`, and
`python/mqt/core/plugins/qiskit`. Public result APIs remain unchanged.

## Decisions

`CompilerTarget::findOperation` reuses capability matching to select native
names from source semantics, parameters, and ordered physical sites. QIR keeps
canonical calls internally because cleanup and resource analysis understand
those names; serialization applies native names to a clone. Incoming internal
name markers are discarded before lowering. DDSIM normalizes only its exact
native aliases at the existing JIT boundary.

The shared compiler/CLI QIR exporter reads the typed native-operation attribute
without rebuilding the target model. One symbol-use scan of the module body
preserves referenced QIS declarations, including address-taken functions. An
inconclusive scan keeps all declarations. The QDMI adapter moves serialized
bitcode directly into its payload; the public byte-vector API remains available.

The importer records source scalar-result provenance as `mqt.output_name`.
QC/QCO transformations retain it, and shared QIR lowering rejects it for target
payloads before discarding source metadata. An explicitly authored `i64` status
return remains supported.

A shared driver adoption path owns provider handles immediately after creation
or retrieval. RAII covers wrapper and map allocation. Small local C boundary
guards translate allocation failures to QDMI status codes. Allocation fault
injection uses external assert-based probes rather than adding a global
allocator override to the test suite.

DDSIM writes dense and sparse amplitudes into caller storage instead of
retaining extracted vectors. Only the sparse entry count is cached. The sparse
change removes a 24 MiB size-query allocation for the measured 20-qubit state.
Upstream now supplies the dense-cache removal and shot aggregation moves.
Repeated reads traverse the DD again; callers should reuse a retrieved result.
CSV parsing moves its existing storage. The DDSIM guide demonstrates a NumPy
view of the existing raw-result bytes, avoiding Python complex objects without
introducing another result API.

Qiskit preserves the per-operation extension hook. A fresh Target captures each
alias before equal instructions and placements are merged. The first alias
supplies overlapping calibration; global support supplies global calibration.
Unequal instructions under one Target name are rejected. Placement data is read
once per operation, and name-only validation skips qubit-index lookup. Provider
operation-name queries remain fresh; there is no backend-lifetime cache.

## Validation

On upstream `d973aefb5c`, the affected release targets passed 2,593 native tests
with one existing SC skip. The native scope includes QDMI client, runtime,
driver, registry, DDSIM, SC, compiler, OpenQASM translation, QC/QCO round trip,
MQT/QIR IR, Base/Adaptive QIR conversion, QIR JIT/runtime, QCO DD utilities,
gate decomposition, and native synthesis. The three CLI integration tests also
passed; target QIR output uses the advertised `r`, `cz`, and `measure` names.

All 1,159 selected Python tests passed with Python 3.14 and Qiskit 2.6:

```sh
PYTHONPATH="$PWD/test/python" uvx nox -s tests-3.14 -- -n 4 --no-cov \
  test/python/qdmi test/python/plugins test/python/test_mlir.py \
  test/python/test_qco_dd.py test/python/test_mlir_qiskit_target.py
```

Repository lint, whole-file C++ lint against `origin/main` on all 23 selected
files, and `git diff --check` pass. The original revision (`03ab9d878`) also
passed the complete executable Sphinx build and generated HTML link checks. Its
changed documentation was inspected in Chrome, including screenshots, console,
and network requests. The preview's only failed request was the browser's
optional `/favicon.ico` request. Independent design and ownership reviews found
no remaining actionable issue.

External allocation probes verify exactly one provider free after wrapper/map
allocation failures, DDSIM allocation-error status returns, and allocation-free
dense state retrieval. Compiler stress probes confirm that alias pruning no
longer repeatedly scans the module. These are local diagnostic workloads on a
shared ARM64 host, not end-to-end throughput benchmarks.

## Limits

Native OpenQASM requires representable operation names and physical qubits;
measurement and reset use their language keywords. Built-in `U` has incompatible
phase semantics and is rejected instead of adding compensation machinery. The
`sqrt_iswap` synthesis shorthand needs a matching `xx_plus_yy` capability for
native payload export; existing tests assert rejection while retaining their
Jeff, Qiskit native-cost, and phase checks. Qiskit QASM2 still approximates
near-constant values because its exporter has no supported precision switch;
QASM3 is the preserving path.

Hardware/cloud execution, live Slurm, Windows/macOS, TSAN, and hosted CI are not
part of this local validation.
