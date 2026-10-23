# Array pipeline cleanup

Status: complete; implementation and local validation are complete. Hosted CI is
separate from local validation.

## Scope and decisions

Reduce repeated work in native QIR execution, frontend initialization analysis,
and OpenQASM export. Keep the supported array subset and memref representation.
Compact literal initialization remains a separate follow-up.

- `Session.cpp` runs LLVM `MemCpyOptPass` before sampling's O1 pipeline and for
  aggregate loads in state extraction. Early scalar replacement otherwise
  expands bulk copies. Keep O1: a smaller pipeline loses useful loop
  optimization, while extra early DSE repeats expensive analysis. Exported QIR
  and target capabilities remain unchanged.
- `OpenQASMSemantics.cpp` visits selected initialization runs directly. Resolve
  every offset before mutation, check sources before targets, and detach target
  state once. Visit innermost strided elements without another recursive call;
  the first callback implementation regressed dense selections. The parser
  requires positive rank and each selection matches that rank; the innermost
  callback needs no scalar recursion leaf. Callers skip fully initialized
  storage, including empty arrays.
- `QCProgram::intoOpenQASM3()` and the owned low-level translator consume input,
  including on failure. Explicit copies use the existing `copy()` API or Python
  `copy=True`. Keep the context alive through diagnostics and module
  destruction. Translation still buffers output until success.

## Validation

Existing frontend, compiler, exporter and Python tests cover initialization,
negative strides, branch isolation, ownership and round trips. Bypassing target
copy-on-write makes the branch-isolation test fail. No test cases were added.
The native release suite passes all 4,080 tests and Python 3.14 passes all 2,028
tests. C++ lint, regenerated stubs, repository lint, executable docs and
internal documentation links pass. The changed documentation pages were visually
inspected. A full Ponytail review removed unreachable visitor paths and
redundant test work. String/stream equivalence and explicit inlining rejection
still protect distinct contracts. No correctness finding remains.

Fresh same-toolchain alternating comparisons verified checksums, sampled
results, statevectors and export bytes. A 1,000-element full-copy checksum
reduced warmed native JIT CPU from median 66.2 ms to 5.2 ms and optimized LLVM
from 4,020 to 20 instructions. Sparse selected-run analysis fell from 14.6 ms to
2.9 ms; public import fell from 78.3 ms to 66.7 ms. Dense cases overlap.
Consuming export removed 127.2 MB of allocation across three 50,000-gate
imports/exports; peak live heap stayed 63.03 MB while independent process peak
RSS fell. These are workload-specific results on a shared host. Phase timings
exclude file I/O and destruction; literal-QASM end-to-end ranges overlap. Raw
harnesses and measurements remain outside the repo.

A further aggregate-load gate for sampling showed no stable benefit and was
rejected. Keep the existing completeness fast paths and low-level verifier: new
caches or unchecked entry points lack a demonstrated benefit. Completeness
checks can still scan backing words; the whole analysis is not uniformly
proportional to selection size. Compact literal initialization remains deferred.
