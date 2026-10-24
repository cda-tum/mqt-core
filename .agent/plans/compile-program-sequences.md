# Compile homogeneous program sequences

Status: complete; local validation passed.

## Outcome and scope

`compile_program` accepts homogeneous `collections.abc.Sequence` inputs and
returns an ordered list, including for empty and singleton batches. Scalar
calls, native pipelines, and submission APIs retain their behavior. The binding
and signature source are `bindings/mlir/register_mlir.cpp` and
`bindings/patterns.txt`; `python/mqt/core/mlir.pyi` is generated with
`nox -s stubs`.

## Contracts and decisions

- Accept generic sequences, preserve order and scalar precedence, and apply
  scalar output selection and ownership to each element. Source strings or paths
  can infer different formats from their contents.
- Support one Python input kind, including subclasses. Iterators, nested
  sequences, mixed kinds, and binary buffers are outside the supported subset.
- Snapshot references and validate kinds before consuming typed inputs.
  Compilation failure returns no results and can consume its processed prefix,
  including the failing input. Later inputs remain untouched; `copy=True`
  preserves all inputs.
- Compile serially against one target snapshot. An empty sequence skips target
  resolution. Keep existing program and context ownership.
- Use nanobind handles and `list_builder` to borrow tuple-owned inputs and
  transfer results into a preallocated list. Native scalars bypass the Python
  sequence ABC. Classify the first supported family once; exact Python types
  share that classification and differing subclasses use the canonical base
  checks. Input families are determined by their classes. Keep the owning tuple,
  scalar precedence, and lazy Qiskit integration; add no enum or Python cache.
- Constrained type variables reject mixed kinds; scalar and sequence overloads
  preserve output types. Qiskit target defaults precede same-family defaults
  because Qiskit's partly untyped `Target` otherwise matches `CompilerTarget` in
  ty. A closed `TypedDict` and `Unpack` share the common keyword arguments
  without accepting unknown keywords. Further shape-TypeVar experiments returned
  `Sequence` instead of `list`, union signatures lost scalar/list precision, and
  a bound union accepted mixed kinds. Keep the 28 precise overloads.

The
[typing specification](https://typing.python.org/en/latest/spec/distributing.html#syntax)
permits current typing features in stubs when their syntax parses on the oldest
supported Python. Thus `TypedDict(closed=True)` needs no runtime backport;
`type Alias = ...` would break the Python 3.11 parser and is excluded. Python
typing cannot exclude strings from `Sequence[str]`; a bare string stays scalar
at runtime, even when a caller annotates it as a sequence.

## Performance evidence and limits

The optimized Clang 23/ThinLTO comparison used Python 3.14.8, nanobind 3.1,
Qiskit 2.6.0, and the previous PR head `a084814395` as the baseline. Seven
alternating fresh-process pairs called the public entry point with `copy=True`
and seed 7. Core GHZ(2) with `QC_IMPORT` exposes binding costs: median scalar
CPU time fell from 2.69 to 2.30 microseconds; eight-element list calls fell from
20.60 to 18.95 microseconds. Median paired reductions were 14.8% and 8.1%, with
improvements in every pair. Tuple and `UserList` batches improved by 8.7% and
5.9%. These results justify the added dispatch branches.

Eight QFT(10) programs compiled to OpenQASM for a Qiskit target took 60.15 and
60.59 ms; eight Qiskit efficient-SU2(2, reps=1) imports took 5.16 and 5.24 ms.
Those results do not establish an end-to-end improvement. A QFT singleton spent
5.7 ms in target passes within a 9.4 ms public call; loop unrolling and native
synthesis took 1.6 and 1.0 ms. The native profile confirms compiler work
dominates. Input and output hashes matched across all pairs; emitted QFT IR
contained loops and the target OpenQASM used fixed physical qubits. Keep
existing pipelines and context ownership, without payload caching or shared
contexts.

Memray, including Python allocator tracing, measured 626 versus 589 allocations
and 371.5 versus 366.4 kB of allocation traffic for the GHZ batch. Peak live
tracked memory was 177.0 versus 176.2 kB. QFT allocated 82.74 MB in either
build, with identical 855.8 kB peak live tracked memory and 11.1 kB remaining
after result deletion. Process peak RSS was approximately 131 MB. Allocation
traffic, live memory, and RSS are distinct; these results establish no material
peak-memory improvement. Timing results are local to this shared host.

## Validation

The sequence tests retain distinct contracts: generic containers and order,
per-source output inference for text and paths, device compilation and
execution, ownership on success and failure, homogeneity preflight, and Qiskit
target reuse with owning input references. Existing scalar tests cover output
conversions, consumed-program rejection, option forwarding, and input parsing.

Ty LSP probes compare keyword completions, return-type hovers, and diagnostics
before and after edits (zero diagnostics, precise list return types, and common
keyword completions). Positive consumer checks target Python 3.11; negative
probes reject mixed inputs, iterators, nesting, incorrect keyword types, and
unknown keywords. Ruff checks the generated stubs.

A compiled mutation selecting every output from the first source was detected by
the retained mixed-format test. One Ponytail review covered the connected
compiler/sample/submit paths; specialist critique also checked type precedence
and stub compatibility.

After rebasing on upstream `6c0e1f65f8`, the optimized editable build and 1,181
compiler, Qiskit, QDMI-compilation, DD, and QIR-runner tests passed. The
sequence suite has 12 cases, with device payloads executed end to end. Stub
regeneration, repository lint, Python 3.11 type checks, and whole-file C++ lint
passed. The executable documentation and internal links passed; Chrome
inspection covered the walkthrough and API reference. Hosted CI remains separate
from these local results.
