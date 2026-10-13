# Keynote compiler changes

This inventories the changes still carried by `codex/mqsf-2026` relative to
upstream `main` at `7e3f1289e` (13 October 2026). Counts are added/deleted
source lines from `git diff --numstat`, including comments and includes. They
describe review scope, not implementation effort. Implementation totals include
C++ Python bindings. Presentation assets, capture scripts, documentation, the
generated Python enum stub, and Python presentation tests are excluded.

The compiler delta is **678 added / 72 deleted implementation lines**, plus
**667 added / 9 deleted test and test-configuration lines**. Of the
implementation, 305 added / 5 deleted lines are opt-in presentation telemetry.
The remaining 373 added / 67 deleted lines implement the four changes below.
DDSIM runtime telemetry is counted separately below; including it brings the C++
implementation total to **781 added / 81 deleted lines**. The compiler test
total includes 155 added lines of Python Qiskit round-trip tests.

## Compiler behavior

| Change                                                              | Implementation scope                                                    | Complexity and readiness                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| ------------------------------------------------------------------- | ----------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Fixed equatorial-pulse synthesis for IonQ-style GPI/GPI2            | 74 added / 10 deleted lines across seven files                          | **Medium; general implementation with semantic tests.** Adds `r-fixed` as a synthesis basis, exposes `R_FIXED` in Python, recognizes fixed π and π/2 equatorial pulses, and decomposes constant and runtime-angle rotations into that basis. This belongs in the compiler, although the provider-specific parameter and phase convention still needs integration review before a separate upstream PR.                                                                                                                                                                                    |
| Adjacent controlled-gate canonicalization                           | 77 added / 2 deleted lines in `CtrlOp.cpp`                              | **Medium; deliberately narrow, reusable optimization.** Combines adjacent matching control shells so existing canonicalizers can cancel inverse gates or merge constant rotations. Requires identical SSA qubit order and sole-unitary bodies. Rotation merging defers to target-aware synthesis when a target environment exists. It is not a general controlled-block optimizer.                                                                                                                                                                                                        |
| Region-local classical bits and source names in OpenQASM and Qiskit | 168 added / 42 deleted lines across three files                         | **Low–medium; reusable correctness and readability fixes.** Keeps OpenQASM declarations in their loop/branch scope and tolerates an unattached while-region during import. Qiskit export collects nested classical storage and inserts zero stores at each block entry when initialization requires them. Both exporters preserve scalar/scratch labels through optional MLIR `NameLoc` metadata. Declared register names win over debug hints; colliding or empty hints receive generated names. This does not add general dynamic-bound or classical-indexing support to Qiskit export. |
| Explicit compiler target on the CLI                                 | 54 added / 13 deleted lines across `mqt-cc.cpp` and the Python bindings | **Low–medium; general mechanism, provisional public interface.** Adds `--target=#mqt.compilation_target<…>` using the existing validated target representation and `CompilerTarget.__str__()` to serialize that representation from Python. This lets the recorder compile against queried/snapshotted capabilities without fabricating a live device. It is mutually exclusive with `--qdmi-device`, and uses the same payload and compilation path. A standalone upstream change should document the option and review its CLI ergonomics.                                              |

Relevant implementation:

- [Target basis detection](../../mlir/lib/Compiler/Target.cpp),
  [Euler synthesis](../../mlir/lib/Dialect/QCO/Transforms/Decomposition/Euler.cpp).
- [Controlled-gate canonicalization](../../mlir/lib/Dialect/QCO/IR/Modifiers/CtrlOp.cpp).
- [OpenQASM import](../../mlir/lib/Dialect/QC/Translation/OpenQASMToQCEmitter.cpp)
  and
  [export](../../mlir/lib/Dialect/QC/Translation/TranslateQCToOpenQASM3.cpp),
  and [Qiskit export](../../bindings/mlir/qiskit/QiskitExport.cpp).
- [Compiler CLI](../../mlir/tools/mqt-cc/mqt-cc.cpp) and
  [Python target bindings](../../bindings/mlir/register_mlir.cpp).

The fixed-pulse basis participates in the existing Euler tests, including bound
runtime parameters. A dedicated target test checks full-unitary equivalence and
native-gate conformance, including phase. Controlled-gate tests cover inverse
pairs, relative phase in controlled rotations, and distinct axes that must not
merge. OpenQASM/Qiskit round trips check local initialization, while-loops,
source-name collisions, empty names, and diagnostic locations. The Qiskit
exporter changes contribute 88 added / 17 deleted lines; the target bindings
contribute 12 added lines, split between the synthesis enum and serialization.

The CLI tests cover accepted/rejected combinations, malformed target attributes,
and successful target compilation. A small additional test-only adjustment
refreshes QIR CLI diagnostics after upstream wording changed; it does not relax
validation or add compiler functionality.

## Presentation telemetry

These are intentional keynote patches with regression coverage,
**not a proposed stable tracing API**. They are disabled unless their
environment variable is set.

| Trace                         | Implementation scope                                                                                                                | What it records and its limits                                                                                                                                                                                                                                                                                                                                                                  |
| ----------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `MQT_MQSF_MAPPING_TRACE`      | 219 added / 5 deleted lines in `Mapping.cpp`                                                                                        | Actual trial layouts, forward/backward passes, selected trial, routed gates/SWAPs, and a bounded search sample. The search records only the first search per routing state and retains at most 256 generated-node descriptions; truncation is reported. Trial records are emitted after parallel refinement to prevent interleaving. JSON/stderr schema and overhead are presentation-specific. |
| `MQT_MQSF_SYNTHESIS_TRACE`    | 86 added lines in `TargetSynthesis.cpp`                                                                                             | Accepted two-qubit fusions: exact before/after IR, operation indices, composed matrix, and native basis use. The recorder disables MLIR multithreading; concurrent writers and a durable public trace format are outside its contract.                                                                                                                                                          |
| DDSIM program and shot traces | 103 added / 9 deleted implementation lines across `Device.cpp` and `WorkerMain.cpp`; 213 added / 2 deleted test/configuration lines | Real program completions, QDMI/JIT timing, worker progress, and indexed results. Adaptive execution retains native parallel workers and samples in 64-shot recording batches. Timestamps are batch-completion observations, not individual-shot execution timestamps. Recording overhead is included.                                                                                           |

Mapping tests compare traced/untraced output and replay the emitted routing
steps. Synthesis tests compare full unitaries, exact unchanged final IR, and
highlight indices. DDSIM tests compare seeded traced/untraced ordered results,
validate every indexed arrival and worker partition, and exercise unwritable
trace paths. The CI fix in this round makes that test honor the runtime's actual
worker count: requesting four workers does not override a two-CPU runner or an
explicit worker budget. Production sampling is unchanged by that fix.

## Capture assumptions and work that is not a compiler fix

- Target models and naming/parameter conversions live in
  [capture_devices.py](capture_devices.py). The three hardware examples compile
  against SC QDMI mock targets assembled from provider metadata. DDSIM executes
  the resulting programs. This is not execution or calibration validation on
  IQM, IBM, or IonQ hardware.
- IonQ GPI/GPI2 are represented internally as fixed-angle equatorial rotations;
  the recorder exports the corresponding native gate definitions and accounts
  for their phase convention. Provider metadata calls its entangler `zz`; the
  presentation uses the equivalent `rzz` angle convention. These adapters are
  presentation integration code, not a new provider implementation.
- The structured syndrome and iterative-QPE examples use portable adaptive QIR.
  Their captures do not establish arbitrary structured-program support on the
  three hardware targets.
- The new nested-loop iQPE needs **no native compiler patch**. Its eight-round
  outer loop contains a runtime-dependent controlled-U power loop, reset,
  measurement, and floating-point feedback. Both loops survive into executable
  adaptive QIR. Qiskit export currently cannot represent this combination of
  dynamic bounds, classical indexing, and runtime feedback; its diagram is
  derived from the submitted OpenQASM source instead.
- These counts deliberately exclude fixes already merged upstream. They are an
  inventory of the remaining branch delta, not an attribution of all work that
  made the keynote possible.

## Suggested upstream split

The four compiler behavior changes can be reviewed independently, each with its
own semantic tests. The local-bit and controlled-gate fixes have the smallest
review scope. Fixed-pulse synthesis merits a dedicated target/convention review.
The explicit-target CLI deserves a separate API/documentation decision.

Keep the three trace mechanisms on the keynote branch unless a concrete public
telemetry use case justifies a shared design. Their captured results are tested,
but their environment-variable interfaces, output schemas, bounds, and timing
cost are intentionally tailored to this presentation.
