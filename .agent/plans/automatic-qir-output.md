# QIR output and dynamic sampling

Status: implementation, review, and local validation complete.

## Contract

DDSIM captures QIR text when integer/double recording is referenced or global
constructors/destructors may record outside the shot frames. Other QIR jobs
retain compact counts and expand shots on request. Shot order is unspecified.
The public `sample()` function uses QCO and never enables QIR text. Explicit C++
JIT streams remain complete; `sampleCounts` suppresses text while sampling.

Eligible histogram-only QIR splits shot populations at measurements, resets, and
releases. Sibling paths replay deterministic entry-point prefixes. Equal
normalized reset states merge because global phase cannot affect the histogram.
After LLVM O1, eligibility excludes ambient memory, native effects, pointer
observations, unresolved helpers, and lifecycle hooks. Other supported QIR
retains per-shot execution. Worker protocol version remains 4.

## Evidence and decisions

Six existing benchmark inputs were checked through public QDMI jobs against
independent analytical references. Grouping removes the large per-shot cost for
iterative QPE and repeat-until-success. A reset-heavy probe justifies merging
equivalent reset outcomes. Mutation probes detect omitted hidden measurements,
lost lifecycle records, and deliberately wrong distributions.

Keep Adaptive QIR first in format preference. OpenQASM avoids LLVM JIT setup and
wins several small jobs, while QIR wins repeat-until-success and accepts some
current compiler inputs rejected by the OpenQASM exporter. Neither route wins
universally. Do not add format heuristics or a general execution layer.

Counts-only recording saves retained memory; grouping also reduces execution and
allocation churn. Full numeric output remains proportional to recorded volume.
Replay can still approach per-shot work when outcomes do not share paths. Seeded
results repeat within a strategy, not across algorithms.

## Completion

- [x] Conditional recording and existing compact-count transport.
- [x] Bounded grouped sampling with conservative fallback.
- [x] Representative performance, emitted IR, phases, and memory measurements.
- [x] Independent semantic references and targeted mutation probes.
- [x] One Ponytail review; no actionable findings or justified further deletion.
- [ ] Final native/Python suites, whole-file C++ lint, repository lint, and
      docs.
- [x] Signed commits verified; update the existing PR after final inspection.

Machine-local benchmark inputs, binaries, scripts, logs, and reports are kept
outside the repository under the `core-dynamic-parity-20261024` and
`core-qir-histogram-20261024` experiment directories. They pin the measured
revision and source diff rather than attributing old measurements to a later
rebased commit.

The upstream cpp-linter 1.15 local client no longer writes the result file
expected by the nox wrapper. Inspect the reported diagnostic counts separately
from that wrapper failure. Hosted CI status is reported with the PR update.
