# PennyLane QDMI through Catalyst's native device interface

Status: implementation and local validation pass. Hosted validation remains in
progress.

## Outcome and scope

Ordinary PennyLane QNodes and compiled Catalyst QNodes share gate translation,
MQT target compilation, QDMI submission, and finite-shot readout. The Python
`Device` retains registration, configuration, upstream preprocessing, result
shapes, and the shared batch/recovery API. Ordinary QNodes require neither
Catalyst nor implicit `qjit` compilation.

```text
ordinary QNode -> PennyLane transforms -> QCProgram.from_pennylane
              -> compile_program -> submit_program / try_submit_programs
qjit QNode -> Catalyst runtime -> MLIR extension's MQTQDMIFactory
           -> native gate recording -> QCProgram -> compiler/QDMI adapter
both paths -> logical-wire readout -> requested PennyLane results
```

`QCProgram.from_pennylane(tape, *, wires=None)` imports a preprocessed terminal
tape without a provider session. It appends computational-basis readout for all
mapped wires; the caller prepares measurement bases and interprets the results.
The returned public `QCProgram` supports existing transformations and exports.
The device uses the existing `CompiledProgram`, `compile_program`, and
`submit_program` APIs, plus the generic `try_submit_programs` helper. There is
no private Python recorder or prepared-program type.

The implementation lives in `mlir/lib/Target/Catalyst/`,
`bindings/mlir/Target/PennyLane/`, and `python/mqt/core/plugins/pennylane/`. The
compiler synthesizes and routes against provider metadata. QIR and OpenQASM 3
targets are supported; existing compiler contracts own format selection and
physical legality. OpenQASM 2-only targets are unsupported.

Ordinary finite-shot observables, parameter-shift gradients, custom wires,
broadcasting, shot vectors, deferred measurements, and recovery remain
available. The importer itself does not accept symbolic mid-circuit measurements
or conditional gates: ordinary device preprocessing defers them first.

## Durable decisions

- Group the Python bridge under `bindings/mlir/Target/PennyLane` and
  `mqt::bindings::pennylane`, matching Qiskit's target layout. Its declaration
  header exposes only compiler and nanobind types; Catalyst headers stay in the
  isolated implementation. The native runtime belongs to `Target/Catalyst`.
- Link the static `MQTCompilerCatalystAdapter` into the existing MLIR extension,
  which exports `MQTQDMIFactory`. No separate device DSO or Catalyst runtime
  library is needed. Vendor the four required headers and license under
  `vendor/catalyst/0.15.0/`, with provenance and hashes. This follows the
  existing extension ownership and removes runtime library discovery. Check
  Catalyst >=0.15,<0.16.0a0 before returning the factory: the runtime has no ABI
  handshake, and 0.14 differs in virtual signatures and table layout.
- Reuse `QCProgramBuilder`, `StandardGate`/`emitStandardGate`, `QCProgram`,
  `CompiledProgram`, and the QDMI compiler adapter. The gate table maps source
  names and conventions. Negative controls use X conjugation; PennyLane global
  phase has the opposite sign from Core's convention.
- Native `SetBasisState` accepts one initial preparation with runtime bits and
  arbitrary wire subsets. It emits X gates and rejects later preparation or
  resets. Keep `initial_state_prep=false`: that upstream flag also promises
  general `SetState`, which is unsupported.
- Native `MatrixOperation` uses Core's dense-unitary verification and synthesis
  for at most two total wires, including controls. Adjoints and negative
  controls share the named-gate implementation. Advertise `QubitUnitary` only in
  the qjit capability section so ordinary PennyLane retains decomposition of
  larger matrices. The public tape importer also accepts native matrices.
  General controlled matrices exceeding two wires cannot yet be synthesized;
  reject them before recording. Native Hamiltonian support remains out of scope.
- Finalize at the first terminal measurement and cache one full logical readout
  per circuit and shot count. Release resets circuit state because Catalyst
  pools devices inside a compiled invocation. Gates after terminal measurement
  and stale qubit/observable IDs are errors.
- Native Identity/Z products reduce to parity. Upstream transforms diagonalize
  other observables and split bases. Native expectation/probability primitives
  preserve non-capture compiled parameter-shift; lowering everything to samples
  would break Catalyst's differentiation checks.
- Emit classical bit i for logical wire i. Decode QDMI's descending classical
  bit strings once, then select requested wires. Physical placement must not
  reorder output. Counts-only results provide a sample multiset without temporal
  shot order. Dense compiled counts/probabilities remain exponential; native
  checks cannot protect allocations already made upstream.
- A private context owns its QDMI device/session and custom parameters,
  including copied binary values. Compiled kwargs contain only an opaque numeric
  context ID. A synchronized weak registry resolves it; live Python/native
  devices retain strong ownership. Registry locks never span provider calls.
  Concurrent callbacks require the provider to support concurrent calls on the
  shared session, as documented for direct QDMI usage; recorder state belongs to
  each separately leased Catalyst device.
- Preserve `PennyLaneJob` and the shared QDMI `Batch` scheduler. Compile all
  tapes before submission. `try_submit_programs` validates compatible compiled
  programs and returns `None` if the provider declines native batching before
  submission. Grouping, shot-copy boundaries, recovery, tracking, and retries
  remain eager frontend responsibilities.
- Keep exceptions and RTTI at the Catalyst/nanobind boundary while using the
  compiler's result-based error APIs internally. Do not let exceptions escape
  the QDMI C interface. Build the binding outside unity builds to avoid LLVM
  feature-detection macros changing KJ's exception/RTTI configuration in the
  existing Jeff importer. Honor the existing `LLVM_REQUIRES_RTTI` target
  requirement when applying shared compiler flags; disabling RTTI breaks
  exception matching across the Catalyst boundary on macOS.

## Upstream contracts and supported capture

Research inspected PennyLane `9792b3ea46a7b41ab709357b6efc63110e2ccdac`
(`0.46.0-dev129`) and Catalyst `dd4f8d5f290efae3a828b7c5a5a8d5fe8a854b95`
(`0.16.0-dev121`), alongside PennyLane 0.45.1 and Catalyst 0.15.0. These
remained the current upstream heads on 2026-10-18. The vendored headers are from
Catalyst v0.15.0 commit `3126674625638fee8413d5828b24d2b074a0aeb4`. Inspected
development virtual signatures match; header compatibility alone is not runtime
validation.

The [custom-device guide][custom-devices] distinguishes ordinary Python
execution from native compiled execution. `get_c_interface()` alone does not
redirect ordinary QNodes. The discontinued experimental `eval_jaxpr` device path
is not a replacement.

Both capture modes support native gates, computational-basis
samples/counts/probabilities, and Identity/Z/product expectations and variances.
The current [capture pipeline][capture] replaces some Python preprocessing with
empty passes. For other observables, apply existing upstream compiler passes:

```python
@qp.qjit(capture=True)
@catalyst.passes.diagonalize_measurements
@qp.transform(pass_name="split-non-commuting")
@qp.qnode(device, shots=1000)
def circuit(angle):
    qp.Hadamard(0)
    qp.S(0)
    qp.RY(angle, 1)
    return qp.expval(0.5 * qp.Y(0) + qp.Z(1))
```

Local stable/development experiments execute Y/product/Hamiltonian and
noncommuting expectations using this recipe. Splitting currently accepts
expectation measurements only; diagonalization alone supports variance.
`qp.transforms.decompose` with the native `gate_set` separately handles `Rot` in
both stacks. Composing gate decomposition with measurement passes fails in the
inspected development graph-decomposition pass, although the same circuit works
in stable Catalyst. Use non-capture compilation for that combination; remove
this qualification when the upstream passes compose successfully. Upstream
skipped-pass warnings can remain even with explicit compiler passes.

Captured parameter-shift differentiation fails in PennyLane's
[`_qnode_jvp`][capture-jvp], including on `lightning.qubit`; use `capture=False`
for compiled gradients. Native gradients, analytic execution, and variance
gradients are unsupported. Catalyst requires fixed contiguous integer wires, and
its current shot-vector support is samples-only without capture and absent with
capture. Compiled counts and probabilities allocate dense result buffers; use
samples for wide registers.

Core can compile and execute complete programs with mid-circuit measurements and
classical feedback, as verified with an OpenQASM 3 measurement/conditional
program on DDSIM. Catalyst instead
[loads each measurement result][measure-lowering] immediately before executing
its next branch. A terminal-recording device cannot return an unresolved
symbolic bit through that callback. Catalyst also
[rejects deferred MCM][deferred-rejection]. Supporting compiled feedback as one
QDMI job requires an importer for complete Catalyst IR before this lowering;
claiming another capability or buffering the callback cannot supply it. A full
PennyLane symbolic-MCM importer is likewise a separate compiler extension.

`QuantumDevice` has no multi-program submission callback. Independent compiled
QNodes reach separate synchronous terminal-measurement calls. Catalyst's
`async_qnodes=True` can run them concurrently but cannot combine them into one
QDMI job. Eager tapes use the generic native batch submission API.

Compiled code depends on a live in-process context and cannot restore a saved
executable in another process. Keep the Python device alive while using compiled
functions. Compiled calls do not use Python batch recovery, retries, or tracker
updates. Generic Catalyst seeds are unsupported; provider-specific job
parameters, including DDSIM's seed, remain available.

## Dependencies and validation

Core retains Python `>=3.11`. PennyLane is optional and bounded to
`>=0.45.1,<0.46.0a0`; the bound admits development builds when explicitly
requested while excluding the next release series. Catalyst is installed
separately by users. Its `>=0.15,<0.16.0a0` test dependency uses Python/platform
markers for available wheels: Python below 3.15, Linux x86_64/aarch64, and macOS
arm64. Normal `tests` and `minimums` sessions include it on those platforms;
other platforms skip compiled tests through the shared availability helper. The
weekly upstream workflow and `catalyst-dev` session exercise development
Catalyst and PennyLane with Python 3.12.

Tests under `test/python/plugins/qdmi_pennylane/` cover ordinary native
execution, compiled correlated readout, runtime basis preparation and matrices,
wire order, pooled reuse, both capture modes, observable lowering, error
recovery, and non-capture expectation/probability gradients. Native tests cover
result validation, context lifetime, and generic compiler batching. Existing
compiler tests own synthesis/routing coverage.

```sh
ctest --preset release -R 'CatalystQDMI|CompilerQDMIAdapter'
uvx nox -s tests-3.12 -- test/python/plugins/qdmi_pennylane
uvx nox -s minimums-3.12 -- test/python/plugins/qdmi_pennylane
uvx nox -s catalyst-dev
```

Where TestPyPI lacks native wheels, `CATALYST_WHEEL` can select an official
artifact. The inspected ARM64 artifact from upstream run `38949132648`
(`f545c9e9`, dev120) differs from inspected Catalyst HEAD only by its version
bump. Development runtime probes used that artifact with PennyLane dev129;
released probes used Catalyst 0.15.0 with PennyLane 0.45.1.

Current local validation on Linux aarch64:

- Stable and development stacks each pass 55 integration cases, including
  runtime basis preparation and matrices. An installed wheel on standard CPython
  3.15 without Catalyst passes 47 cases and skips the compiled module. The
  latest wheel also passes 1,888 Python consumer tests with Qiskit 2.6.0. The
  Qiskit 2.5.2 compatibility and development Catalyst suites pass 757 cases
  together. Both nox session command contracts and the merged lockfile validate.
- The full release suite after rebuilding with the corrected RTTI flags passes
  4,041 tests, with the existing `QueryJobId` case skipped. This includes four
  native Catalyst tests and 19 compiler adapter tests. Whole-file C++ analysis
  checks all eight changed implementation/test files with no findings, and stub
  regeneration passes on the latest source.
- Hosted regular Python tests, including macOS, and Codecov checks pass on
  `1e1bb1bc3`, confirming the RTTI correction for the earlier exception-handling
  failure. Those results do not establish hosted success for the subsequent
  rebase and binding relocation; final hosted validation remains pending.

Installed-wheel probes pass with both Catalyst/Core import orders on stable and
development stacks. They execute the exported factory, check the capability
file, and confirm that no separate adapter library is loaded. The wheel includes
the vendored license and contains no adapter DSO.

The full executable documentation build and internal link checks pass after the
native feature additions and RTTI fix. Chrome DevTools inspection verified
desktop/mobile layout, the importer output, all new API anchors, QAOA figures,
and the new preparation/matrix scope paragraph. Content assets loaded
successfully; the local preview's only failed request was its missing favicon.
The exact capture recipe above also passes on both rebuilt installed
stable/development stacks. No real hardware or performance speedup is claimed.

[custom-devices]: https://docs.pennylane.ai/projects/catalyst/en/latest/dev/custom_devices.html
[capture]: https://github.com/PennyLaneAI/catalyst/blob/dd4f8d5f290efae3a828b7c5a5a8d5fe8a854b95/frontend/catalyst/from_plxpr/device_utils.py
[capture-jvp]: https://github.com/PennyLaneAI/pennylane/blob/9792b3ea46a7b41ab709357b6efc63110e2ccdac/pennylane/workflow/_capture_qnode.py#L285-L296
[measure-lowering]: https://github.com/PennyLaneAI/catalyst/blob/dd4f8d5f290efae3a828b7c5a5a8d5fe8a854b95/mlir/lib/Quantum/Transforms/ConversionPatterns.cpp#L903-L935
[deferred-rejection]: https://github.com/PennyLaneAI/catalyst/blob/dd4f8d5f290efae3a828b7c5a5a8d5fe8a854b95/frontend/catalyst/qfunc.py#L85-L91
