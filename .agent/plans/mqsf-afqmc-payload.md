# AFQMC payload and presentation simplification

Status: complete. The canonical payload, refreshed scientific evidence and
smaller offline presentation pass the checks below.

## Purpose and scope

The talk follows the exact OpenQASM 3 payload submitted by Core's PennyLane
adapter for shadow snapshot 0 through every target's compiler stages and DDSIM
execution. It does not compile the classical NumPy/PySCF AFQMC propagation. The
existing six-qubit LiH case and scientific contracts remain unchanged. See
[the presentation README](../../presentations/mqsf2026/README.md) for the
capture and packaging commands and evidence limits.

## Progress

Implementation, bounded reuse measurements, full capture refresh and independent
review are complete. Physical laptop/projector/clicker rehearsal remains outside
this software validation.

## Surprises & Discoveries

The prior representative used PennyLane's separate OpenQASM 2 serializer,
expanding RXX rotations before the displayed compiler pipeline. The actual
adapter already emits OpenQASM 3.0 and preserves RXX. The recorder now takes
snapshot 0's returned job payload and format directly. The existing `_samples`
instrumentation remains valid after the adapter's internal `_shots_or_counts`
return-type change; no compatibility wrapper is needed.

All 2,048 payloads are distinct. The producer shares a trial preparation and a
66-angle shadow schedule, but different zero angles specialize differently.
Current QC/QCO Python bindings provide copy and parameter binding, but no public
append or function-composition API. Reusing a compiled circuit indiscriminately
would change the program.

The browser previously embedded about 30 MB of snapshot payloads, raw shots,
shadow data and per-walker energy/weight arrays that it never used. Packaging
now keeps these observations only in the unchanged downloadable evidence.
Removing unused illustration functions and CSS leaves live scene output
unchanged.

## Decision Log

Use `Job.get_program(program_index)` as the canonical source. Keep the narrow
pre-submission payload comparison, successful-attempt checks, one native job
with sequential indices, and exact QDMI/PennyLane shot, count and wire-order
comparisons. The builder ties snapshot 0, its hash and all target stage chains
together. No synthetic source replaces the submitted program.

Retain no reuse optimization. A new composition API or symbolic workaround is
not justified by the measured cost. No generic cache, Clifford pass, terminal
Pauli correction, source-string surgery or new dependency is introduced.

Execution capture requires every available executable export to succeed.
Execution and device recorders write only complete results, using atomic file
replacement. They no longer accept partial execution as success. Files declared
unsupported by the compiler remain outside this execution requirement.

The full scientific dataset remains in `captures/demo.json.gz` and the ZIP's
`evidence.json.gz`. Only the browser view is smaller. Original program text is
preserved there, including code beyond the highlighted 200-line excerpts; no
unused second compressed copy is embedded in the page.

## Reuse measurements

Baseline: Core `d2b9c25db137fba530ac59787590fb82fd5030ec`, rebased on
`974e4fdc8561bd92b241c207546331ba9b4deb3a`; Clang 23 with ThinLTO. Four actual
snapshots (0, 1, 17, 2047), one warmup and seven measured repetitions, one
NumPy/BLAS thread, recorded Emerald SC model, compiler seed 7, three mapping
trials, two refinement iterations, lookahead 8, 4 MiB search memory, and 256
DDSIM shots. These are exploratory local measurements, not speedup claims. First
SDK imports cost 5.21 s and subsequent initialization 19 ms; neither is included
below.

Times are median milliseconds [minimum–maximum]. End-to-end sums producer,
conversion, compiler and DDSIM-client stages, excluding semantic validation,
artifact serialization and classical chemistry. Native submit/wait is contained
within DDSIM-client time and includes ordinary worker startup.

| Stage                   |            Snapshot 0 |            Snapshot 1 |            Snapshot 17 |         Snapshot 2047 |
| ----------------------- | --------------------: | --------------------: | ---------------------: | --------------------: |
| Preparation             |      0.98 [0.48–1.01] |      0.55 [0.51–0.57] |       0.56 [0.54–0.58] |      0.55 [0.54–0.55] |
| PennyLane decomposition |      4.75 [2.29–5.67] |      2.33 [2.30–2.85] |      2.37 [2.30–22.79] |      2.35 [2.32–2.87] |
| Adapter conversion      |      2.30 [1.15–2.35] |      1.15 [1.13–1.17] |       1.16 [1.14–1.23] |      1.15 [1.13–1.16] |
| OpenQASM import         |      2.88 [1.36–2.96] |      1.85 [1.53–1.90] |       1.87 [1.82–1.91] |      1.83 [1.59–1.86] |
| QC → QCO                |      4.92 [2.32–4.96] |      2.55 [2.49–2.58] |       2.59 [2.49–2.73] |      2.53 [2.50–2.57] |
| Optimization            |      6.04 [2.88–6.07] |      3.00 [2.89–3.04] |       3.08 [3.01–3.15] |      2.99 [2.92–3.04] |
| Target compilation      |   25.47 [13.03–26.79] |   14.90 [14.45–16.29] |    15.11 [14.89–16.71] |   13.79 [13.17–15.03] |
| QIR lowering            |   24.46 [17.26–27.86] |   17.78 [13.96–21.36] |    16.61 [13.89–18.85] |   15.30 [14.43–16.50] |
| DDSIM client            |   67.08 [53.77–77.76] |   57.65 [56.34–70.82] |    69.88 [56.94–70.98] |   65.21 [53.07–70.20] |
| Native submit/wait      |   56.03 [42.81–66.82] |   46.95 [45.63–60.02] |    58.96 [46.16–60.31] |   54.43 [42.29–59.51] |
| End-to-end              | 140.17 [95.74–153.16] | 101.50 [98.85–119.88] | 112.86 [103.08–121.04] | 104.70 [93.39–111.32] |

OpenQASM import accounts for 1.7–2.1% of the end-to-end median, below the 18–57
ms total spread. This and the missing composition API support keeping
independent compilation. The parameterized-helper and Clifford experiments were
not warranted. Ad hoc probe code remains outside tracked sources.

## Validation and semantic limits

The four probes reproduce exact submitted bytes from the pinned producer and
current adapter, covering 19, 24, 18 and 26 zero shadow angles. Independent
PennyLane 64×64 matrices and explicit bit-order conversion verify QC and
untargeted QCO including global phase to below 6.2e-16 absolute error. All 64
logical probabilities after target compilation agree below 8.0e-16, with
initial/final placement and measurement destinations checked explicitly.

Target compilation may discard terminal Z frames before measurements; its
unmeasured matrix is therefore not required to equal the input matrix. After
undoing both placement permutations, `U_native U_original†` is diagonal with
unit-modulus entries (off-diagonal error below 1.12e-14, modulus error below
4.8e-15). This establishes computational-basis distribution equivalence for
every input separately from phase-sensitive unitary equivalence.

| Snapshot | Optimized CX / RXX | Native CZ / R | QCO bytes | Native bytes | QIR bytes |
| -------- | -----------------: | ------------: | --------: | -----------: | --------: |
| 0        |            93 / 24 |     122 / 196 |    31,249 |       63,149 |    38,917 |
| 1        |            93 / 20 |     142 / 230 |    30,557 |       70,722 |    42,502 |
| 17       |            93 / 23 |     145 / 235 |    31,256 |       71,897 |    43,372 |
| 2047     |            93 / 19 |     117 / 188 |    30,660 |       60,936 |    37,427 |

These representation counts are not simulator speedups. Probe seed 7 does not
seed the separate PennyLane chemistry measurement stream. The science keeps
LiH/STO-3G at 2.4 Å, CAS(2,3), 2,048 snapshots × 256 shots, 128 walkers, 480
steps, dtau 0.02, seed 17 and four CPU processes. No hardware jobs are
submitted. Reset/adaptive overrides are demonstration capabilities, not evidence
that the physical IQM endpoint supports arbitrary loops.

## Outcomes & Retrospective

The full capture contains 2,048 native programs and 524,288 shots with exact
payload, order and count checks. Every device pipeline begins with snapshot 0's
submitted bytes. The four-program smoke and four phase/order probes passed;
energy curves and uncertainty recompute from all saved walker observations.

The HTML shrank from 66.48 MB to 23.52 MB while the ZIP preserves complete
scientific evidence. Removed 671 lines of unused illustrations/CSS, 2,047 unused
display records, duplicate highlighting and unused compressed source.
Before/after checks found identical live SVG output and geometry/styles across
all 28 build states. No speculative reuse implementation remains.

Validation passed: optimized native build; 384 compiler/synthesis tests; 51
presentation tests plus five subtests; documentation; repository lint and
whole-file C++ lint of the retained presentation patches. Offline Chrome and
Firefox each passed 12 slides, 28 builds, 90 animation layout samples, 5,195
circuit windows, 215 feedback dependencies, result synchronization, first paint,
restart/cancellation, reduced motion and 12 print pages. Chrome DevTools
inspection of the final page found no layout issues across 28 states, no console
errors and no runtime network dependencies. Independent review verified all 50
installed native artifacts, provenance and scientific records.

The data still demonstrates an ideal local simulator and a bounded active space.
Hardware execution, general composition APIs, large-space AFQMC and a full
uncertainty analysis remain outside scope.
