# Exception-free API audit

Status: integration fixes applied. Baseline: PR #2545 rebased on main
`62125acad`.

## Scope

The remaining PR migrates native APIs and diagnostic boundaries. LLVM/MLIR
dependencies, compiler Program APIs, DDSIM process isolation, typed gate
construction, JSON single-parse evaluation, and independent DD/QIR fixes are
already on main. Their behavior must survive the result-type conversion.

## Integration findings

- **DDSIM output lifetime:** collect output after JIT destruction, including
  independently compiled sampling peers. `WorkerMain.cpp` retains main's scope
  guard and explicit peer destruction. The existing
  `QIROutput.CapturesTypedRecordsAndValidatesBuffers` test checks one destructor
  record per shot, regardless of the granted worker count.
- **DD allocation settings:** `Package::create` rejects zero vector and matrix
  capacities before construction. Internal `MemoryManager::create` requires a
  positive capacity and asserts it. Both settings are exercised by
  `RejectsInvalidConfigurationBeforeAllocation`; no second result factory or
  duplicate propagation is needed.
- **DD serialization and fidelity:** adapt main's write-failure, imaginary-unit,
  and measurement-width tests to result APIs. Keep text/binary vector/matrix
  writes, all six signed and case variants of the imaginary unit, the exact
  `size_t` width boundary, scalar states, valid reordering, and invalid indices.
  Remove the narrower overlapping fidelity test and fold complex parser cases
  into its existing table.
- **JIT lifetime:** one test combines constructor quantum-state initialization
  with classical-allocation lifetime and destructor output. It creates the
  session on another thread and runs it after that thread exits, detecting
  accidental fallback-runtime ownership as well as incorrect runtime binding.

## Remaining diagnostic boundaries

- The QDMI driver retains unwinding through its C ABI allocation-recovery
  boundary. Its allocation-injection test checks `QDMI_ERROR_OUTOFMEM`, a null
  output handle, and successful retry. Embedded LLVM support archives do not
  exchange `llvm::Error` objects across shared-library boundaries.
- Python compiler target queries and submission retain `ValueError` where
  specified. Missing required provider symbols retain `QDMI_ERROR_FATAL`.
- QCO sampling workers capture diagnostic metadata and replay diagnostics in
  worker order. Serial and parallel division-by-zero tests retain the same
  category and message without leaking to stderr.

## Coverage and limits

JSON family tests retain round trips, semantic case IDs, resolved parameters,
schemas, and count evaluation. Shared tests cover duplicate keys, malformed
input, numeric limits, source diagnostics, and altered manifests. DD tests
retain malformed and truncated serialization, nonfinite weights, references,
qubit limits, and throwing streams. Mutable QIR globals and ABI restrictions
remain supported and tested.

See the [execution plan](../plans/exception-free-core.md) for current validation
and outstanding platform and performance checks.
