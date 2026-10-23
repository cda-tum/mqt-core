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
- Use nanobind type handles and `isinstance`, preserving scalar dispatch order.
  The shared scalar importer needs no batch-specific enum. Inspect supported
  bases only for candidate sequences; keep optional Qiskit imports lazy.
- Constrained type variables reject mixed kinds; scalar and sequence overloads
  preserve output types. Qiskit target defaults precede same-family defaults
  because Qiskit's partly untyped `Target` otherwise matches `CompilerTarget` in
  ty. A closed `TypedDict` and `Unpack` share the common keyword arguments
  without accepting unknown keywords.

The
[typing specification](https://typing.python.org/en/latest/spec/distributing.html#syntax)
permits current typing features in stubs when their syntax parses on the oldest
supported Python. Thus `TypedDict(closed=True)` needs no runtime backport;
`type Alias = ...` would break the Python 3.11 parser and is excluded. Python
typing cannot exclude strings from `Sequence[str]`; a bare string stays scalar
at runtime, even when a caller annotates it as a sequence.

## Performance evidence and limits

The initial optimized Clang 23/ThinLTO build on upstream `0e0826b1bf` used
Python 3.14.8 and Qiskit 2.6.0. Seven alternating pairs compiled eight copies of
Core GHZ(32), QFT(10), repeat-until-success(3), and Qiskit efficient SU2(10,
reps=3), through the public entry point with a Qiskit target, seed 7, and
OpenQASM output. Batching saved median paired CPU time of 5.3%, 1.1%, 3.7%, and
1.7%, respectively. The shared host prevents a general wall-time claim.

QFT target passes took 12.6 ms of a 19.0 ms singleton call; the native profile
placed 83.4% of sampled cycles in MLIR. Input classification and payload
preparation were not visible hotspots. Output IR preserved dynamic loops where
required, and scalar/batch output hashes matched. These findings justify one
target import, but no enum, payload cache, or shared-context infrastructure.

Memray measured QFT allocation traffic of 82.84/83.44 MB for batch/scalar calls,
peak live tracked allocations of 0.937/0.931 MB, and 89.7/84.4 kB remaining
after results were deleted. Process peak RSS was about 130 MB. These measure
different quantities and establish neither a live-memory improvement nor a leak.

## Validation

The sequence tests retain distinct contracts: generic containers and order,
per-source output inference for text and paths, device compilation and
execution, ownership on success and failure, homogeneity preflight, and Qiskit
target reuse with owning input references. Existing scalar tests cover output
conversions, consumed-program rejection, option forwarding, and input parsing.

Ty LSP probes compare keyword completions, return-type hovers, and diagnostics
before and after edits. Positive consumer checks target Python 3.11; negative
probes reject mixed inputs, iterators, nesting, incorrect keyword types, and
unknown keywords. Ruff checks the generated stubs.

A compiled mutation selecting every output from the first source was detected by
the retained mixed-format test. One Ponytail review covered the connected
compiler/sample/submit paths; specialist critique also checked type precedence
and stub compatibility.

After rebasing on upstream `bcf97cbb6e`, the optimized editable build and 1,181
compiler, Qiskit, QDMI-compilation, DD, and QIR-runner tests passed. The
sequence suite has 12 cases, with device payloads executed end to end. Stub
regeneration, repository lint, Python 3.11 type checks, and whole-file C++ lint
passed. The executable documentation and internal links passed; Chrome
inspection covered the walkthrough and API reference. Hosted CI remains separate
from these local results.
