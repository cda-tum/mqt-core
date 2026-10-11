# Shared HPCQC playground for QDMI workflows

Status: implementation complete. Browser rendering review remains blocked by the
unavailable Chrome connection; macOS VM execution is not validated locally.

## Scope and decisions

`examples/slurm` provides login submission, scalable classical and quantum
access nodes, and persistent slurmdbd/MariaDB accounting. Every QAN can compile
circuits and reach all configured devices. One environment and catalogue serve
all implementations. Partitions place jobs; licenses limit concurrent device
access. Device services enforce authorization.

The single-device examples derive their device ID from `SLURM_JOB_LICENSES` and
request one unit of a local license. Applications using several devices receive
explicit IDs and open independent sessions through the generic built-in driver
API. The playground monitor probes status in a bounded child process and updates
license-only reservations. Scheduler policy stays outside the installed package.
The released C++/Python Slurm adapters and the proposed native checker are
removed; the PR describes the API migration.

Site monitor credentials and workload credentials remain separate. Device output
is suppressed before loading libraries. Failed probes leave admission closed,
and systemd cleans descendants when a monitor exits. Availability is a snapshot
for the monitor identity. Controller communication failures can leave the
previous scheduler state active and require administrator attention.

The scientific application remains separate. The guide demonstrates ordinary
Slurm job dependencies for quantum measurement and classical propagation.

## Performance and portability

The monitor and command-line entry points use Python 3.15 lazy imports for
modules that are unused on their startup paths. Earlier supported Python
versions keep eager import behavior. Device imports stay within the quiet worker
and its exception boundary; optional adapter initialization remains unchanged. A
local flake8-lazy audit informed these choices without adding a CI dependency.

On Linux ARM64 with Python 3.15, alternating warm subprocess measurements gave
median DDSIM probe startup of 27.2 to 22.3 ms (40 samples per version), and CLI
help of 29.8 to 23.1 ms (22 samples). These measure startup, not device-service
latency or end-to-end cluster throughput.

For Apple Silicon development, the guide uses a native Linux VM with rootful
Docker. Builds, checkouts, and runtime files stay on its Linux filesystem. The
runner rejects macOS and mismatched-architecture wheels before creating the
cluster. The same cluster files serve Linux hosts and the documented VM path;
Docker Desktop and macOS hardware have not been tested.

## Validation

Rebased on MQT Core main `974e4fdc`. The runner/monitor suite passes 19 cases,
and the CLI suite passes 14 cases on Python 3.11, 3.14, and 3.15. Repository
lint and the complete Sphinx build pass, including all 18 executable notebooks
and internal-link checks. Generated HTML contains syntax-highlighted examples
and an accessible topology figure; the SVG was previewed locally. Chrome
introspection was unavailable, so a visual review of the rendered pages remains
outstanding.

The real cluster passed with three classical nodes and two QANs using an exact
main wheel at `974e4fdc`: login submission, license-derived device selection,
placement, DDSIM/SC execution, contention, failed-probe admission, recovery,
environment export, and accounting after service recreation. The full run
completed in 55 seconds.

The package/API removal at `c9be8c8f` previously passed the Clang release build
with IPO, 591 QDMI CTest cases with one existing skip, and 378 installed-wheel
Python tests with one skip. Generated bindings and full-file C++ lint passed.
The startup and documentation changes do not alter those native interfaces.

Device CI pins the playground revision separately from its MQT Core main package
pin. Authenticated Emerald mock and SV1 workloads test native and wheel modes,
including loaded-library checks. Their repositories provide package, native, and
executable-documentation validation. Timeout and controller-failure coverage
remains in the runner suite; real admission, recovery, device isolation,
credentials, and cleanup are exercised in the cluster.

Dynamic-node recreation requires removing obsolete node records when retaining
controller state.
