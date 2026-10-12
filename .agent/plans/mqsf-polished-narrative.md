# MQSF presentation refinement

Status: implemented. Final publication checks are recorded on draft PR #2685.

## Goal and scope

Implement the detailed slide feedback while preserving actual compiler and
runtime evidence, clicker operation, offline packaging, and FullHD readability.
Split the opening into ecosystem, shared architecture, and products. Keep
persistent visual anchors through the AFQMC and compiler chapters. Add brief
chapter separators and a staged, illustrated feature overview before the close.

## Decisions

- Every narrative change waits for a click. Sequential path reveals belong to
  one bounded animation. Only the explicitly illustrative compiler caricatures
  may loop, with cancellation and reduced-motion support.
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
- [x] Persistent circuit/device compiler layout, routing and blockwise
      synthesis, complete IR views, and scoped scientific caricatures.
- [x] Compact syndrome/RUS sources, region-local bits, full adaptive QIR,
      synchronized source/circuit/IR highlights, and Grover growth animation.
- [x] Three-column adaptive execution log/sequence/histogram and illustrated
      feature overview with verified support claims.
- [x] Consistent typography, spacing, arrowheads, numbering and transitions;
      remove replaced scenes/styles and update notes/checks.
- [x] Native and presentation validation, every browser state and transition,
      including forward/backward navigation, print, and reduced motion.

## Validation

The existing CI fails only its QIR CLI output test on Linux, macOS, Windows, and
coverage builds. Updating its two stale diagnostic strings reproduces and fixes
that failure locally. Keep the actual validation cases intact.

Use release-clang-ipo for native changes, focused semantic tests and scientific
checks, full presentation tests, repository and whole-file C++ lint, and Chrome
DevTools. Test forward/back navigation, restart/cancellation, reduced motion,
print layout, and every 1920x1080 reveal. Compare hosted ZIP members to local
bytes and check the final commit's required CI. Physical projector/clicker
rehearsal remains necessary.

## Results

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
