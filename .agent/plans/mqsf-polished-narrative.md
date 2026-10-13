# MQSF presentation refinement

Status: October 13 revision implemented and locally validated. Publication
through draft PR #2685 includes a final hosted-package comparison.

## Goal and scope

Implement the detailed slide feedback while preserving actual compiler and
runtime evidence, clicker operation, offline packaging, and FullHD readability.
Split the opening into ecosystem, shared architecture, and products. Keep
persistent visual anchors through the AFQMC and compiler chapters. Add brief
chapter separators and a staged, illustrated feature overview before the close.

## Decisions

- Every narrative change waits for a click. Sequential path reveals belong to
  one bounded animation, with cancellation and reduced-motion support.
- Reuse stable SVG identities across related scenes. Circuit diagrams
  distinguish controls, target gates, multi-qubit rotations, measurements,
  classical feedback, and scoped control flow. Full programs take precedence
  over short excerpts; formatting may change whitespace, never instructions or
  semantics.
- Run 4,096 AFQMC programs with 1,024 shots each, on ideal DDSIM. Capture the
  preparation, compilation, per-program completion, and result retrieval. Four
  Slurm container nodes demonstrate distribution on one physical host.
- Use an actual shorter imaginary-time run, retaining both full curves and their
  scientific limitations. Do not crop or choose measurements to improve a claim.
- Improve controlled-gate canonicalization and local classical declarations at
  their owning layers. Preserve phase, control polarity, SSA wire identity, and
  region initialization. Mapping layouts and synthesis block animations must be
  tied to actual compiler output; diagrams of search concepts are marked
  schematic.
- Keep data captures separate from static packaging. Preserve the revision and
  hashes of every original run, connected through the exact submitted payload.

## Delivered

- [x] Title, three opening slides, device cards, principles, and matching close.
- [x] Scientific AFQMC illustration, complete source, synchronized three-column
      replay, 1,024-shot QIR batch, and four-node Slurm evidence.
- [x] Persistent circuit/device compiler layout, routing and native synthesis,
      readable QIR views, and downloadable complete evidence.
- [x] Complete syndrome source, region-local bits, adaptive QIR control flow,
      synchronized source/circuit/IR highlights, and Grover growth animation.
- [x] Three-column adaptive execution log/sequence/histogram and illustrated
      feature overview with verified support claims.
- [x] Consistent typography, spacing, arrowheads, numbering and transitions;
      remove replaced scenes/styles and update notes/checks.
- [x] Native and presentation validation, every browser state and transition,
      including forward/backward navigation, print, and reduced motion.

## Validation

The October 12 changes fixed stale QIR CLI diagnostic expectations while
preserving the validation cases. The October 13 changes add source-name
preservation coverage and refresh the presentation evidence.

Use release-clang-ipo for native changes, focused semantic tests and scientific
checks, full presentation tests, repository and whole-file C++ lint, and Chrome
DevTools. Test forward/back navigation, restart/cancellation, reduced motion,
print layout, and every 1920x1080 reveal. Compare hosted ZIP members to local
bytes and check the final commit's required CI. Physical projector/clicker
rehearsal remains necessary.

## October 12 Results

- Seventeen slides, 73 presenter states, 34 bounded replays, and two explicitly
  illustrative loops. Scientific and corporate specialists reviewed the result.
- Re-recorded 4,096 QIR programs with 1,024 shots each and 240 imaginary-time
  steps on four Slurm compute nodes. Compiler, completion and retrieval events
  are independently timestamped. Imaginary time is labeled separately.
- Reused the same input for all three hardware targets. Emerald and IBM need one
  SWAP; IonQ needs none. Actual accepted fusion events drive native views.
- Native suite: 4,033 registered tests, zero failures, one normal skip. Qiskit
  translation: 444 passing tests. Presentation: 72 tests and five subtests.
- Controlled-gate canonicalization and region-local classical declaration fixes
  are isolated in signed commits for reuse. The failing CI diagnostic checks now
  match the existing CLI behavior.
- Removed the obsolete stack renderer and overridden styles. Complete evidence
  remains in the offline ZIP; the browser loads only its display projection.
- Chrome DevTools covers every settled reveal, fractional opening frames,
  scientific playback, console and network. Both browser engines check runtime
  evidence, control dependencies, circuit windows and print output. Physical
  projector and clicker rehearsal remains outside these automated checks.

## October 13 Refinement

- Horizontal frontend/compiler/orchestrator/backend architecture; combined timed
  reveals, blue/orange paths, all four provider cards retained in the product
  view.
- Stable reference nodes and clocks, fixed footer center, no nested duplicate
  transforms, consistent official TUM and Slurm vector scaling.
- Continuous recorded AFQMC wall clock, per-program preparation completion,
  timestamped energy/walker frames from a fresh four-node run.
- Persistent compiler stage cards, placement-only lead-in, one-click IBM/IonQ
  retargeting, QIR continuation and exact native-call omission markers.
- Remove the two compiler-internals slides and RUS builds; add AFQMC and MQT
  Core chapter separators; restore paper partners and proper Grover math.
- Preserve best-effort scalar/local-bit source labels at import/export while
  prioritizing explicit register names and retaining diagnostic locations.
- Validate all 56 states, fractional playback, stable-element regression cases,
  code/evidence fidelity, both browsers, and the final hosted offline ZIP.

### October 13 Validation

- Signed native name-preservation patch: 453 Qiskit tests, 522 OpenQASM CTest
  cases, optimized build/install, and whole-file C++ lint pass.
- Both browser engines pass all 56 states, 26 finite replays, 156 sampled
  animation layouts, 709 circuit windows, 39 feedback dependencies, and 17
  printed pages. Shared-reference and clock checks fail with the old motion
  engine and pass with the fix.
- Independent Chrome DevTools review inspected all slides and midpoint builds at
  1920x1080. Follow-up layout sweeps confirm the opening text, card shuffle, QPE
  caption, and closing title corrections.
- Final AFQMC run: 4,096 programs, 4,194,304 shots checked against native
  histograms, two completed Slurm jobs, four compute nodes, and a continuous
  293.176-second application timeline. No real hardware jobs were submitted.
- Three target compilations preserve the measurement instrument to within
  2.9e-15; each unchanged QIR output also runs for 512 DDSIM shots.
- Presentation tests pass: 73 tests and five subtests. Stub regeneration
  produces no tracked changes. Repository lint passes.
