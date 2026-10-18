# PennyLane QDMI through Catalyst's native device interface

Status: implemented and locally validated on
`refactor/pennylane-catalyst-device`.

## Outcome and scope

A private shared library, `mqt-core-catalyst-qdmi`, implements Catalyst's
`QuantumDevice` contract and owns recording, MQT target compilation, QDMI
submission, and finite-shot readout. The small Python `Device` retains
registration, configuration, upstream preprocessing, eager replay, result
shapes, and the existing batch/recovery API. Ordinary QNodes require neither
Catalyst nor `qjit`. The Python OpenQASM serializer is removed.

```text
ordinary QNode -> PennyLane transforms -> native recorder
qjit QNode -> Catalyst runtime -> MQTQDMIFactory -> native recorder
native recorder -> QCProgram -> MQT target compiler -> QDMI -> readout
```

The compiler synthesizes and routes against provider metadata. QIR and OpenQASM
3 targets are supported; OpenQASM 2-only targets are excluded. Existing compiler
target/submission contracts own format selection and physical legality.

Ordinary finite-shot measurements, Hamiltonians, parameter-shift gradients,
custom wires, broadcasting, shot vectors, and recovery remain supported.
Compiled execution uses `qjit(capture=False)`, fixed contiguous integer wires,
terminal measurements, and finite shots. Analytic execution, native gradients,
interactive mid-circuit feedback, dynamic allocation, and saved executables that
can be loaded in another process are outside this change. Ordinary deferred
measurements remain available.

## Durable decisions

- Reuse `QCProgramBuilder`, `StandardGate`/`emitStandardGate`, `QCProgram`,
  `CompiledProgram`, and the QDMI compiler adapter. The gate table only maps
  source names and conventions. Negative controls use X conjugation; PennyLane
  global phase has the opposite sign from Core's convention.
- Finalize/compile at the first terminal measurement and cache one full logical
  readout per circuit and shot count. Release resets circuit state because
  Catalyst pools devices inside a compiled invocation. Reject gates after
  preparation and stale qubit/observable IDs.
- Native Identity/Z products reduce to parity. Upstream transforms diagonalize
  other observables and split bases. Native expectation/probability primitives
  preserve compiled parameter-shift; lowering everything to samples would not.
- Emit classical bit i for logical wire i. Reverse QDMI's descending classical
  bit strings once; select requested wires afterward. Physical placement must
  not reorder output. Counts-only results provide a sample multiset without
  temporal shot order. Dense counts/probabilities remain exponential; native
  checks cannot protect allocations already made upstream. Prefer samples for
  wide circuits.
- Contexts own their provider handle and custom parameters, including copied
  bytes. Compiled kwargs contain only an opaque numeric `context` ID. A weak
  synchronized registry resolves it; live native devices/programs retain strong
  ownership. Registry locks never span provider calls. Adapter locking makes no
  promise about unrelated calls through an injected provider handle.
- Preserve `PennyLaneJob` and the shared QDMI `Batch` scheduler. Immutable
  native programs replace Python payloads; grouping, shot-copy boundaries,
  recovery, tracking, and retries remain eager frontend responsibilities.
  Compiled calls are synchronous. Generic Catalyst seeding is unsupported;
  provider-specific job parameters, including DDSIM's seed, remain available.
- This private ABI boundary retains exceptions/RTTI for nanobind. No LLVM/MLIR
  objects cross it. Hide static dependencies, and link no Catalyst runtime or
  compiler library. Install the pinned headers' Apache license with the adapter.
- Compile the binding bridge outside unity builds: LLVM feature-detection macros
  would otherwise change KJ's exception/RTTI detection in the existing Jeff
  import translation unit. Keep this isolation local to the new bridge.

## Versions and evidence

Research inspected Core `9ca32dd3e4df86fc313eca1c326adbf6ec27b7f9`, PennyLane
`9792b3ea46a7b41ab709357b6efc63110e2ccdac` (`0.46.0-dev129`), and Catalyst
`dd4f8d5f290efae3a828b7c5a5a8d5fe8a854b95` (`0.16.0-dev121`), alongside
PennyLane 0.45.1 and Catalyst 0.15.0, current on 2026-10-17.

The [custom-device guide][custom-devices] distinguishes ordinary Python
execution from native compiled execution. `get_c_interface()` alone does not
redirect ordinary QNodes; do not implicitly add `qjit` or implement the
discontinued experimental `eval_jaxpr` path. The inspected
[capture pipeline][capture] omits some splitting/diagonalization needed here;
revisit the explicit non-capture limitation when upstream implements those
transforms.

Headers are pinned to Catalyst v0.15.0 commit
`3126674625638fee8413d5828b24d2b074a0aeb4`. Inspected development virtual
signatures match, including `NamedOperation`'s seventh optional string-vector
argument. Header compatibility alone is not runtime validation.

PennyLane remains `>=0.45.1` without an upper pin. Install Catalyst separately.
The `test-catalyst` dependency group has its own Python `>=3.11,<3.15` range for
current JAX wheel constraints; Core and the universal lock remain `>=3.11`.
Ordinary tests on newer Python do not acquire Catalyst. Stable Catalyst runs in
Linux CI; development Catalyst/PennyLane run in weekly upstream checks.

## Validation

Two native regressions cover provider-result validation/owned binary parameters
and registry/lifetime handling. Existing compiler tests own synthesis/routing
coverage. Real DDSIM cases replace serializer-specific tests with gate
semantics. Compiled tests cover correlated terminal results, wire ordering,
pooled reuse, observable lowering, error recovery, and expectation/probability
gradients.

```sh
cmake --preset release -DBUILD_MQT_CORE_BINDINGS=ON -DBUILD_MQT_CORE_SHARED_LIBS=ON
cmake --build --preset release --target mqt-core-mlir-unittests-catalyst
ctest --preset release -R CatalystQDMI
uvx nox -s catalyst
uvx nox -s catalyst-dev
uvx nox -s stubs
uvx nox -s cpp-lint -- HEAD
uvx nox -s lint
```

Where TestPyPI lacks native wheels, `CATALYST_WHEEL` can select an official
artifact. The inspected ARM64 artifact from upstream run `38949132648`
(`f545c9e9`, dev120) differs from inspected Catalyst HEAD only by its version
bump. Validate it with PennyLane dev129 and test the released stack separately.
Verify both import orders and installed capabilities/library/license contents.

Local evidence on Linux aarch64:

- Both native regressions pass, including real DDSIM output before injected
  malformed-provider cases. Whole-file C++ analysis reports no findings.
- Installed-wheel PennyLane suite: 55 passed without Catalyst (compiled module
  skipped), 59 passed with Catalyst 0.15.0/PennyLane 0.45.1, and 59 passed with
  Catalyst 0.16.0-dev120/PennyLane 0.46.0-dev129. Both Core/Catalyst import
  orders execute ordinary and compiled QNodes successfully in both stacks.
- The new `catalyst` and `catalyst-dev` nox sessions also pass end to end, each
  running all 59 integration tests. Development uses the official ARM64
  artifact.
- Existing QDMI, batch, and relevant MLIR Python consumer suites: 917 passed.
- Stub generation, repository lint, lock consistency, and the complete
  executable documentation build with local link checks pass. Chrome DevTools
  inspection covered desktop/mobile layouts, QAOA plots, console, and network
  requests.
- The wheel contains the native adapter, capability configuration, and Catalyst
  license. Metadata retains Python `>=3.11`, PennyLane `>=0.45.1`, and no
  mandatory Catalyst dependency. The shared library exports no LLVM/MLIR symbols
  and links no Catalyst runtime. Group resolution also succeeds on Python 3.11
  and 3.15.
- Net test/fixture code drops by 329 lines, including the two native tests and
  compiled regressions. Removed serializer tests are replaced by semantic device
  checks; existing compiler tests retain target synthesis/routing coverage.

Final specialist review found no remaining design blocker. Portability review
added Windows DLL staging for native test discovery. Other operating systems,
real hardware, hosted CI, and external documentation link checking were not run.
No performance speedup is claimed.

[custom-devices]: https://docs.pennylane.ai/projects/catalyst/en/latest/dev/custom_devices.html
[capture]: https://github.com/PennyLaneAI/catalyst/blob/dd4f8d5f290efae3a828b7c5a5a8d5fe8a854b95/frontend/catalyst/from_plxpr/device_utils.py
