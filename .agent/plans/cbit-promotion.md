# Classical register promotion

Status: complete.

## Scope and contracts

The registered `promote-classical-memory` pass connects CBit to MLIR's SROA and
mem2reg interfaces. It normalizes indices before preparing wide comparisons. The
CBit dialect owns decomposition and comparison preparation; QC/QCO cleanup, Base
QIR preparation, and payload loop unrolling use the same promotion pass.

Private registers of at most 256 bits support constant-index accesses,
whole-register writes, packed reads up to 64 bits, and wider reads used only in
constant comparisons. Returned registers, arguments, dynamic accesses, and
opaque users retain storage. Output widths are independent of the promotion
bound. Each allocation owns distinct storage; register arguments may alias.

QCO branches insert promoted classical results before quantum results. MLIR owns
SSA construction and structured loop handling. Reading an undefined bit before
its first write remains undefined behavior; promotion may choose zero. The old
block-level register tracker is removed, while scoped branch-condition
propagation and bounded canonicalization remain. Allocation-based initial-read
folding preserves known zero values across unrelated quantum operations without
tracking later writes or assuming arguments cannot alias.

Qiskit uses ordinary exact-width integer expressions and scalar snapshots. Its
initialization tracker is removed under the undefined-read contract;
region-local zero allocations still reset on every entry. Switch selectors stay
limited to 64 bits, and dynamic indices retain index-width truncation. jeff's
limit on general packed integer arithmetic is unchanged.

DD sampling and Adaptive QIR record explicit Boolean results alongside CBit
registers, preserving order, repetitions, and snapshots. Other classical result
types are rejected by sampling instead of silently selecting quantum-wire
output. Statevector extraction is unchanged. Base QIR retains its measurement
output contract.

Adaptive QIR preserves Result arrays when measurement stores fuse safely.
Otherwise it uses existing Boolean storage, including runtime indices and
returned measurements whose slots are overwritten. All Boolean output loads
precede output recording. Computed scalar outputs use function results, without
temporary registers. Unsupported register calls and mixed Result/Boolean storage
merges still diagnose. Wider scalar outputs remain outside this change.

## Validation

The original integration base was `9fb299b3d3c442515907bc5a377cf20a44d57923`,
including PRs #2801 and #2802. Validation used LLVM/MLIR 23.1 and Python 3.14:

- All 4,043 native tests passed.
- The nine affected MLIR, Qiskit, and QDMI Python suites passed 1,157 tests with
  Qiskit 2.6.0. Integer interchange, Qiskit translation, and target tests also
  passed 685 cases with Qiskit 2.5.2.
- Stub generation, repository lint, executable documentation, and internal links
  passed. Rendered QIR, OpenQASM, Qiskit, and Python API documentation was
  inspected in Chrome.
- Repository C++ lint passed. Additional full-file checks include the newly
  registered promotion pass and both versioned Qiskit adapters. LLVM's installed
  `OperationSupport.h` emits analyzer lifetime warnings; no suppressions were
  added. Hosted CI and hardware execution are separate from these local checks.

Run the native suite with `ctest --preset release`. Run the Python suites with:

```sh
uv run --no-sync pytest \
  test/python/test_mlir.py test/python/test_mlir_loops.py \
  test/python/test_mlir_integer_interchange.py \
  test/python/test_mlir_qiskit_translation.py test/python/test_qirrunner.py \
  test/python/test_qco_dd.py test/python/qdmi/test_compilation.py \
  test/python/test_mlir_parameter_binding.py \
  test/python/test_mlir_qiskit_target.py
```

The checks cover these contracts:

- CBit and classical-simplification tests cover promotion eligibility, aliases,
  retained snapshots, initialization, branch result positions, zero-trip and
  nested loops. Unsupported-use assertions inspect storage operations rather
  than full printed IR.
- `test_mlir_integer_interchange.py` checks all integer predicates against
  Python integers, reversed operands, and retained wide snapshots through DD,
  OpenQASM, jeff, and Qiskit. A runtime-indexed 150-bit register keeps its
  packed read; a branch-folded static index still promotes.
- `qdmi/test_compilation.py` executes 50-, 100-, and 150-bit local-control
  programs as Base and Adaptive QIR and checks every output bit. Mixed scalar
  and register outputs exercise overwritten snapshots and repeated results.
- Existing output-phase checks remain necessary: moving a load after recording
  may preserve DDSIM output while violating QIR's profile ordering contract.
- Earlier local mutation probes changed expected snapshot outputs (`00100` to
  `00101`, and `11011` to `10010`), demonstrating that the semantic references
  detect delayed reads. These are historical checks, not final-head CI results.

Earlier shared-host measurements motivated the 256-bit cap: one branch writing
150 bits took about 4.3 ms with promotion versus 0.64 ms without it; at 256
bits, about 10.7 ms versus 1.0 ms. Promotion removed storage in those inputs but
added compiler work. These measurements preceded this integration and do not
establish an end-to-end or hardware speedup.

## QIR and exporter follow-up

Rebased on `6fffd4a68f1b24fae713bff5b49560fdd608245f`. The pre-commit failure
was MD013 on the reproduction command above; a fenced shell command avoids the
formatter joining it into one long inline span.

The QIR review uses specification tag `2.1`, revision
`a177eee5885d99e965f5cb919c8d70b0ab7a6a15`. Boolean storage initializes with one
native LLVM array store instead of `llvm.memset`. Dynamic indices no longer
imply backward branching; actual CFG cycles still determine that capability.
Base conversion rejects unmeasured output bits because static Results have no
specified initial value. Adaptive conversion preserves initialized zeros.

Executable probes check scalar snapshots, mixed outputs, runtime indices, and
zero-trip and loop-carried values against deterministic expected results.
Mutating the zero initializer changes `10` to `11`; delaying a Result read
changes `11` to `10`. These failures distinguish semantic checks from LLVM
verification. The native conversion tests retain initialization placement,
metadata, output ordering, and the Base diagnostic regressions.

Remaining export gaps are explicit: Qiskit rejects scalar results and dynamic
CBit loads; both exporters reject repeated register results; OpenQASM rejects
general arithmetic wider than 64 bits. Materializing Boolean scalar results in
fresh output registers permits Qiskit export, with scratch bits still visible.
Snapshots, zero-trip and nested loops, and 150-bit bitwise operations export
correctly. General QIR array copies and math still emit intrinsics outside the
profile instruction lists; DDSIM execution is not proof of portability.

Open PR #2591 covers classical array storage in OpenQASM, not Qiskit. PR #2806
changes frontend width limits. PR #2626 covers complete QDMI outputs, not native
Qiskit output materialization; its scalar-output guard must preserve Boolean
support when integrating with this change. None closes the other gaps above.

The final native run passes all 4,060 tests. The nine Python suites above pass
1,157 tests after rebasing; the final QIR changes pass all 75 compilation and
QIRRunner cases. Repository lint and whole-file C++ lint pass. Existing loop
tests retain real-loop capability checks; the dynamic-index adapter test now
requires successful compilation without either loop capability.
