# Click-driven MQSF workflow

Status: implementation, fresh captures, native and local browser validation, and
offline packaging complete; draft-preview publication is in progress.

## Goal and scope

Rework the keynote around presenter-controlled reveals, a compact authentic
AFQMC payload, and the common compiler/runtime architecture. The deliverable
remains a self-contained static HTML presentation and offline ZIP. Native work
and Slurm runs produce evidence before packaging; presentation playback never
submits jobs.

The supported scientific example is a small, explicitly bounded molecular active
space. Both quantum-shadow and Hartree–Fock trial results are retained. Measured
program sizes, gate counts, layouts, timestamps, and outcomes own the visuals.
Schematic flows and interpolated movements must not imply additional
observations.

## Decisions

- Use nine slides: title; ecosystem/architecture/products; AFQMC introduction;
  AFQMC execution; target compilation; compiler internals; structured programs;
  adaptive runtime; combined conclusion.
- Every narrative change is click-triggered. A click can launch a bounded smooth
  replay; it cannot automatically advance to the next narrative step. Backward
  navigation restores a deterministic completed state. Reduced motion shows it
  immediately.
- Preserve one exact submitted AFQMC payload across source, compiler stages,
  device targets, and execution verification. Select H2 after semantic
  validation for compact circuits. Do not tune the experiment to force a
  favorable result.
- Use the upstream Slurm playground and capture actual allocation and worker
  evidence. Local multiprocessing alone is not evidence of Slurm execution.
- Keep actual independent-stage artifacts, complete compact inputs, and visibly
  marked display folds for long generated code. Full source remains in the ZIP.
- Keep QDMI the community standard; show MQSC products and maintained device
  implementations with shared-interface badges and originating institutions.
- Separate observed API calls from inferred architecture paths. Only animate
  per-shot branching or intermediate search states when instrumentation records
  them. No real hardware jobs are in scope.

## Work remaining

- [x] Validate H2 preparation, batch execution, and Slurm CPU propagation.
- [x] Capture Grover size measurements, realistic RUS, compact syndrome syntax,
  and sufficient real compiler telemetry for the requested illustrations.
- [x] Integrate slide reveals, reusable architecture, three execution pillars,
  six compilation tiles, and compiler/runtime zooms.
- [x] Remove replaced scenes, stale code/CSS, and inaccurate presenter notes.
- [x] Validate native semantics, evidence consistency, offline navigation, every
      FullHD build and transition, console/network, and packaged docs.

## Validation

Use the repository's native release preset, focused presentation and affected
native tests, presentation `build.py` and `check_browser.py`, complete docs, and
repository lint. Inspect rendered output with Chrome DevTools. A successful
build alone does not establish visual correctness. Record latest results here
when the integrated version is ready.

## Recorded results

Rebased onto `1133ad276`, including the shared Slurm playground and the new
routing boundary data structure. One Slurm quantum allocation submitted 4,096
programs with 256 shots each; its successful completion released four real srun
ranks on two compute-node containers. All containers share one physical host.

The H2 pair-encoded preparation reduces the representative primitive circuit
from 66 to 47 gates (CNOT 15 to 6), with exact equivalence on the two prepared
basis states. This does not replace the general DoubleExcitation unitary. Only
two of 4,096 payloads repeat, so a payload cache is not warranted.

Final shadow-trial energy is -0.96532648 Ha, versus -0.91359614 Ha for the
Hartree-Fock trial and -0.98347273 Ha for the exact reference. Both full curves,
walker errors, and systematic-error limitations remain. No favorable seed or run
was selected to tune the display.

The compiler replays actual sweep operations and bounded A* visits. All three
native circuits pass independent measured-operator equivalence (maximum residual
2.9e-15), and unchanged QIR executes on DDSIM. The one accepted IQM two-qubit
fusion combines CX and SWAP, lies at the iSWAP local-equivalence vertex, and
uses two CZ gates. This is one recorded event, not a synthetic point cloud.

The compact syndrome and corrected literature RUS examples compile and execute.
RUS measures 0.804 for an exact 0.8 outcome probability. Region-local classical
declarations expose an upstream emitter limitation; top-level scratch bits are
an explicit supported-source constraint. A native reproducer remains in local
build evidence rather than broadening the presentation patch.

Browser packaging now projects only consumed fields. It removes unused IR
highlighting, duplicate executions, raw provider metadata, and unused classical
walker frames. Complete evidence bytes remain unchanged in the ZIP. Replaced
application/routing renderers and unused CSS were deleted. Evidence validation
binds compiler revision, canonicalization hashes, and Weyl matrices to their
producing targets; stale captures fail instead of silently updating hashes.

Local validation: optimized native build; 584 affected native tests; 64 Python
tests plus five subtests; complete executable docs; repository lint and
whole-file C++ lint for all three changed native files. Chrome and Firefox pass
all 70 builds, 11 finite replays, 711 circuit windows, 39 feedback dependencies,
66 animation-layout samples, nine print pages, cancellation, and reduced motion.
Chrome DevTools inspection found no layout failures across the 70 states,
console warnings or errors, or runtime network dependencies.

The final mapping-only upstream rebase adds a fresh optimized build and 317
passing mapping/compiler-pipeline tests. Whole-file C++ lint passes again;
compiler captures are regenerated with that installed binary. The earlier Slurm
recording retains its own revision: exact submitted bytes bind the two phases,
so refreshing compilation does not rewrite execution history.

The self-contained HTML is 3.63 MB, reduced from 31.04 MB earlier in this
iteration. The offline ZIP contains 53 members, including 43 path-only SVG
assets and the complete original evidence. Every packaged source hash matches.
Physical laptop, projector, and clicker rehearsal remains outside local checks.
