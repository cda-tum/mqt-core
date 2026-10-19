# OpenQASM classical arrays

Status: complete.

## Scope and decisions

The frontend supports global, fixed-size, one-dimensional arrays of `bool`,
`int`, `uint`, `float`, and compile-time-quantized `angle` values. It reuses
scalar casts, negative-index normalization, and definite-initialization
analysis. Empty arrays are valid. Runtime indices must be in bounds; the
frontend emits no runtime checks. Views, whole-array copies, multidimensional
arrays, and whole-array input or output remain unsupported.

Typed stack storage and upstream SROA/mem2reg avoid a separate array runtime. DD
execution retains rank-zero slots because promotion can produce them. QIR
converts classical memory and index types together in its final stage; checking
all operand types earlier would reject valid array accesses. Base validates
measurement ordering before its restricted instruction subset. Only Base needs
conversion-time canonicalization for that validation; Adaptive uses the existing
shared cleanup pipeline.

OpenQASM, jeff, and Qiskit exports require array storage to disappear. Qiskit
export preserves its existing scalar-only folding contract, so callers must run
`cleanup()` explicitly before exporting promotable arrays. Runtime storage stays
supported by DD execution and Adaptive QIR, subject to their capabilities.

## Validation

- Release build and CTest: 4,012 passed; one existing job-ID test skipped.
- Python compiler, loops, integer interchange, parameter binding, DD, and
  QIR-Runner suites: 531 passed on Python 3.14.
- Export regressions reimport OpenQASM, jeff, and Qiskit and check the resulting
  value with DD sampling. Existing tests retain bounds, empty-array, angle,
  parameter, allocation, and QIR capability coverage.
- The OpenQASM guide's array example, full documentation build, and local link
  checks pass. Chrome inspection covered the rendered page, console, network,
  and screenshot; only the preview server's default favicon request returned
  404.
- Whole-file C++ lint passes for all 18 changed C++ source and test files.
  Repository lint and diff-whitespace checks pass.

No performance claim is made. Hosted CI is separate from these local checks.
