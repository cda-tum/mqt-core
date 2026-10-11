# Shared HPCQC playground for QDMI workflows

Status: complete; scheduling policy and the single-license examples live in the
shared playground.

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

## Validation

The example cleanup passed 17 runner/monitor cases, repository lint, and the
complete Sphinx build with all 18 notebooks and internal links checked. The real
cluster passed with three classical nodes and two QANs using an unmodified MQT
Core main wheel (`5ea425f2`): login submission, license-derived device
selection, placement, DDSIM/SC execution, contention, failed-probe admission,
recovery, environment export, and accounting after service recreation.

The package/API removal at `c9be8c8f` passed the Clang release build with IPO,
591 QDMI CTest cases with one existing skip, and 378 installed-wheel Python
tests with one skip. Generated bindings and full-file C++ lint passed. The
example cleanup does not change those native or installed Python APIs.

Device CI pins the playground revision separately from its MQT Core main package
pin. Its authenticated workloads exercise the Emerald mock and SV1 in native and
wheel modes, including loaded-library checks. The device repositories provide
their own package, native, and executable-documentation validation.

Tests for the removed adapters/checker and duplicate static or mocked scenarios
are omitted. Timeout and controller-failure checks remain in the runner suite;
real admission, recovery, device isolation, credentials, and cleanup are tested
in the cluster. Dynamic-node recreation requires removing obsolete node records
when retaining controller state.
