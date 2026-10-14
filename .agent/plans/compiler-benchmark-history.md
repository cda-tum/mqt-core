# Weekend compiler benchmark history

Status: implemented locally. DGX activation is deferred.

## Outcome and decisions

`benchmarks/runner.py batch` fetches main without updating the reviewed harness
checkout. It measures missing first-parent commits from the inclusive v4.0.0
release, pinned to `8d9bcfb031fa0fe2769a852cdf8d66017d24abe8`. The same SHA is
the fixed reference, measured at batch start and interleaved with each
candidate. Empty batches still measure the reference. No GitHub-hosted worker or
Actions runner registration is needed.

Validated observations are durable completion records. Failed candidates do not
prevent later candidates from running and remain eligible for retry. A failed
initial reference stops the batch. Saved batch records retain attempts and the
main history boundary. Each observation keeps its raw samples, commit IDs,
benchmark definitions and source hash, machine and toolchain configuration,
generator-source identity, and harness commit.

A series identity covers hardware/OS, CPU allocation and available power
settings, compilers, LLVM, Python, asv, uv, benchmark source, build
configuration, and the fixed reference. Build caches are also separate per
series, preventing reuse of wheels built with an old toolchain. Generator-source
changes suppress comparison deltas and break trend lines.

Publication appends all complete groups and finished batch manifests to
`benchmark_results`, never overwriting historical samples. Imports preserve
unpublished local successes and reject conflicting history. Upload and RTD
refresh can be retried separately from measurement. Per-group benchmark
snapshots keep older suites readable after the global asv definitions change.

The Furo dashboard groups compatible observations into main history, preserves
missing points, and shows repeated reference measurements in time order. Manual
pairs stay outside the main-history plots. The asv explorer includes fetched
main history even when it is newer than the fixed harness checkout. RTD imports
data at build time; no independently hosted API or database service is required.

`benchmarks/systemd/` supplies a user service, Saturday 02:00 Europe/Berlin
timer, release-pinned settings, and example toolchain environment. A weekday
condition prevents starts after a weekday resume. Installation and credentials
are operator actions documented in `benchmarks/README.md`; no remote setup was
performed.

## Validation and limits

- Compiler, runner, dashboard decoder, and docs-link regression tests pass,
  including real local Git publication, retries, missing commits, interruptions,
  suite changes, cache separation, and publication beyond the harness revision.
- Node tests cover history ordering, repeated references, gaps, manual pairs,
  configuration grouping, and incompatible inputs.
- Explicit new-file hooks, full repository lint, and asv discovery pass.
- Browser checks confirm the real snapshot loads and synthetic history fixtures
  retain gaps, reference samples, matching paired baselines, and failure
  details.
- Documentation HTML builds successfully. The complete docs session still fails
  on 238 pre-existing links in `cpp/doxygen_crawl.html`; dashboard links pass.
- DGX native builds, measurement stability, credentials, and live publication
  remain untested. systemd activation must be validated on Linux; this host is
  macOS. Abrupt power loss may omit a final batch manifest, but completed
  observations remain checkpoints for the next run.
