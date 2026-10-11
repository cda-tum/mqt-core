# Shared HPCQC playground for QDMI workflows

Status: implementation and local validation complete. Docker Desktop on macOS
has not been run locally.

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
versions keep eager import behavior. QDMI imports are lazy too, first resolved
within the quiet worker and its exception boundary in the playground's Python
runtime. Optional adapter initialization remains unchanged. A local flake8-lazy
audit informed these choices without adding a CI dependency.

On Linux ARM64 with Python 3.15, alternating warm subprocess measurements gave
median DDSIM probe startup of 27.2 to 22.3 ms (40 samples per version), and CLI
help of 29.8 to 23.1 ms (22 samples). These measure startup, not device-service
latency or end-to-end cluster throughput.

Docker builds MQT Core from the checked-out source for its target architecture,
using the official setup-mlir installer. Standard uv requirement lines select
additional PyPI, Git, or local-source packages. The completed environment is
copied into the runtime image; compilers, toolchains, and sources stay in
builder stages. Native/wheel device checks reuse independent device compilation.

Linux Docker Engine and macOS Docker Desktop use the same Compose setup. The
privileged container, host cgroup namespace, and cgroup-v2 mount match Docker
Desktop's documented systemd setup. Enhanced Container Isolation is incompatible
with this access. Native Linux ARM64 validation does not prove execution on Mac
hardware.

Fork and Dependabot CI runs without credentials. Device repositories retain
meaningful native and Python checks and visibly skip authenticated jobs. A
maintainer can dispatch Slurm tests from upstream main against the reviewed,
immutable PR head SHA. The workflow checks the SHA before passing credentials;
GitHub environments alone do not bypass fork secret restrictions. Full
credentialed CI and executable notebooks run on trusted branches. The manual
entry point becomes usable after its workflow reaches main.

## Validation

Rebased on MQT Core main `974e4fdc`. The runner/monitor suite passes 16 cases
after removing host-wheel checks, and the CLI suite passes 14 cases on Python
3.11, 3.14, and 3.15. Repository lint and the complete Sphinx build pass,
including all 18 executable notebooks and internal-link checks. Chrome DevTools
inspection covers the Core, IQM, and Braket guides at desktop and mobile widths,
including console and network checks. The topology uses presentation-style
pictograms and the official QDMI mark; measured label bounds and screenshots
confirm that its text fits. Examples use syntax highlighting and named file
captions.

The source-built cluster passed with three classical nodes and two QANs: login
submission, license-derived device selection, placement, DDSIM/SC execution,
contention, failed-probe admission, recovery, environment export, and accounting
after service recreation. The full run completed in 254 seconds including MQT
Core compilation. A shared environment installed local IQM source, a pinned
Braket Git revision, and a PyPI package; all 33 packages passed dependency
checks and 23 catalogue devices were found. Independent DDSIM and SC sessions
reported `IDLE`. The final runtime's monitor help and DDSIM probe also passed
with networking disabled.

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

Both device repositories passed authenticated native and wheel workloads and
fresh execution of all five documentation notebooks. IQM fork CI passed 175
native unit tests and 98 Python tests, with seven existing live-test skips. Both
repositories passed lint and reviewed-SHA workflow policy probes.

Dynamic-node recreation requires removing obsolete node records when retaining
controller state.
