# Dedicated compiler benchmark runner

Status: complete for local paired measurements and build-time RTD publication.
Weekend batching and reference diagnostics are implemented in
[the history extension](compiler-benchmark-history.md). An independently served
data feed remains outside the current scope.

## Outcome and decisions

`benchmarks/runner.py` runs a baseline/candidate comparison on a dedicated Linux
machine. The command needs a clean checkout, an explicit physical-machine label
and CPU, and configured native LLVM and C/C++ compilers. It records the actual
toolchain and OS. The DGX Spark requires native ARM64 builds; it has not yet run
this prototype. GitHub-hosted measurements and Actions runner registration are
outside this design.

Each comparison uses four interleaved asv rounds of five repeats. A process lock
prevents overlapping invocations in one checkout. CPU affinity is checked in
benchmark setup because asv can continue after an affinity-setting failure.
Compilation time, two-qubit count, and depth use Core's existing generators and
compiler APIs. Changes are informational, with no performance gate.

An independent `publish` command validates the latest complete pair, preserves
raw samples on the `benchmark_results` branch, and optionally requests an RTD
latest build. Publication never replaces an existing group, includes no partial
runs, and can be retried after a successful upload followed by an RTD failure.
Only JSON data is imported for RTD. The dashboard remains in the Core docs;
release versions keep their snapshot. No remote settings or data were changed.

## History extension

The [weekend history implementation](compiler-benchmark-history.md) extends
publication to all completed observations and introduces a stable series
identity, release-pinned reference, resumable batches, and a systemd timer. RTD
continues to import the data branch at build time.

## Validation

- Focused runner, compiler, dashboard, and docs-link tests: 59 passed.
- asv discovery, explicit hooks for new files, and full `uvx nox -s lint`
  passed.
- Browser checks verified comparison selection and machine provenance without
  console errors. The RTD preview retains the real developer-machine snapshot.
- The publication tests use a local bare Git repository and exercise retries,
  history preservation, conflicting samples, partial runs, and RTD failures.
- Native measurements on the DGX and remote publication have not been run.
- The full docs session generated HTML successfully, then reported 238 existing
  Doxygen crawl links; no dashboard links failed. The session therefore fails
  its final whole-site link check.
