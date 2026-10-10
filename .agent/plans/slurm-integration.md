# Shared HPCQC playground for QDMI workflows

Status: complete; the playground uses the existing QDMI APIs and supports a
shared quantum partition.

## Scope and decisions

`examples/slurm` provides login submission, scalable classical and quantum
access nodes, and persistent slurmdbd/MariaDB accounting. Every QAN can compile
circuits and reach all configured devices. One environment and catalogue serve
all implementations. Partitions place jobs; licenses limit concurrent device
access. Device services enforce authorization.

Applications receive explicit device IDs and open independent sessions through
the generic built-in driver API. The playground monitor probes status in a
bounded child process and updates license-only reservations. Scheduler policy
stays outside the installed package. The released C++/Python Slurm adapters and
the proposed native checker are removed; the PR describes the API migration.

Site monitor credentials and workload credentials remain separate. Device output
is suppressed before loading libraries. Failed probes leave admission closed,
and systemd cleans descendants when a monitor exits. Availability is a snapshot
for the monitor identity. Controller communication failures can leave the
previous scheduler state active and require administrator attention.

The scientific application remains separate. The guide demonstrates ordinary
Slurm job dependencies for quantum measurement and classical propagation.

## Validation

The Clang release build with IPO passed 591 QDMI CTest cases with one existing
skip and no failures. The installed candidate wheel passed 378 Python tests with
one skip; the runner/monitor suite passed 21 cases. Generated bindings,
full-file C++ lint, repository lint, and the complete Sphinx build passed. All
18 executable notebooks and internal documentation links were checked.

The real cluster passed with three classical nodes and two QANs using both an
unmodified main wheel and the candidate wheel: login submission, quantum
placement, device execution, capacity contention, failed-probe admission,
recovery, environment export, and accounting after service recreation.

IQM and Braket passed their package/adapter tests against current MQT Core main
and authenticated native/wheel Slurm workloads in the shared quantum partition.
The runs used the Emerald mock and SV1, including loaded-library checks. Device
CI pins the playground revision separately from its main package pin.

The removed tests belonged to the removed adapters/checker or repeated static
container checks. Actual workload, admission, credential, and cleanup coverage
remains in the owning integration suites. Dynamic-node recreation requires
removing obsolete node records when retaining controller state.
