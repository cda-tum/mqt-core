# Use QDMI devices with Slurm

A Slurm job can compile a circuit, submit it to a quantum device through QDMI,
and process the results in one software environment. Each device has a
cluster-wide Slurm **license named after its QDMI device ID**. Requesting that
license limits how many jobs can use the device at once. A site availability
monitor keeps new jobs pending while the device is unavailable.

| Choose…          | To control…                               | Example                                            |
| ---------------- | ----------------------------------------- | -------------------------------------------------- |
| A partition      | Where the application runs                | `--partition=quantum` selects quantum access nodes |
| A device license | Which device the job intends to use       | `--licenses=mqt.ddsim.default` requests DDSIM      |
| QDMI credentials | Whether the device service permits access | The user's IQM token or AWS profile                |

Licenses coordinate jobs within this cluster. They neither authorize device
access nor reserve capacity at a remote service.

## Submit your first job

The administrator provides Slurm clients, a shared quantum software environment,
and a device catalogue. Log in to the cluster and inspect its partitions and
licenses:

```bash
sinfo
scontrol show licenses
```

Save the following as `job.sh`, alongside your application, and submit it with
`sbatch job.sh`. The `quantum` partition name is a site choice; the
{doc}`workflow playground <slurm_cluster>` uses this name for its quantum access
nodes.

```bash
#!/bin/bash
#SBATCH --partition=quantum
#SBATCH --licenses=mqt.ddsim.default
#SBATCH --time=00:05:00
set -eu

srun python workload.py
```

This `workload.py` executes a Bell circuit on DDSIM:

```python
import os

from mqt.core.qdmi import ProgramFormat
from mqt.core.qdmi import builtin_driver as qdmi_driver

device_id = os.environ["SLURM_JOB_LICENSES"].removesuffix(":1")
device = qdmi_driver.open_device(device_id)
program = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0], q[1];
measure q -> c;
"""
job = device.submit_job(program, ProgramFormat.QASM2, 1024)
if not job.wait(60):
    raise TimeoutError("The quantum job did not finish within 60 seconds")
print(job.get_counts())
```

The example requests **one unit of one local device license**. It reads that
allocation from
[`SLURM_JOB_LICENSES`](https://slurm.schedmd.com/sbatch.html#OPT_SLURM_JOB_LICENSES).
Applications using several devices request their licenses together, accept
explicit device IDs, and open an independent driver session for each device.
Each session initializes only its selected device.

The same `device` can be passed to a {doc}`Qiskit <qdmi_backend>` or
{doc}`PennyLane <pennylane_device>` adapter. C++ applications open a device with
{cpp-api:func}`qdmi::builtin_driver::openDevice`.

### Use an authenticated device

Authenticate through the device implementation's usual mechanism before
submission. Slurm
[exports the submission environment by default](https://slurm.schedmd.com/sbatch.html#OPT_export);
credential files must also be readable by the job on its compute node. IQM and
AWS settings can coexist in the same environment.

Request the remote device's license, for example:

```bash
sbatch --licenses=iqm.emerald.mock iqm-job.sh
squeue --me
```

`iqm-job.sh` must request the site's quantum partition and run an application
compatible with that device. An administrator's successful availability check
does not establish your authorization. Workloads still need to handle expired
credentials, device failures, and remote queueing.

## Set up the cluster environment

These steps are for cluster administrators. The
{doc}`playground <slurm_cluster>` supplies a working deployment to experiment
with before adapting it to a site.

1. Install MQT Core and the required device implementations in one versioned,
   administrator-owned environment on the login and quantum access nodes. Expose
   its executables through the default `PATH` or a software module.
2. Put shared, non-secret settings in `/etc/mqt-core/qdmi.json`. The built-in
   driver combines installed device manifests with this catalogue; see
   {doc}`configuration` for the format and precedence.
3. Make user credential files and application inputs available to jobs, using a
   shared filesystem or matching per-node paths and numeric user/group IDs. Keep
   site monitoring credentials separate from user credentials.

Where several software environments are offered, users select a module before
`sbatch` or load it in their batch script. The module supplies the paths and
settings; users do not maintain a virtual environment for each device. See the
[GWDG module guide](https://docs.hpc.gwdg.de/software_stacks/module_basics/index.html).
Environment variables set for a Slurm daemon do not configure user jobs.

## Admit jobs while devices are operational

Register the device IDs and their concurrency limits in `slurm.conf`:

```ini
Licenses=mqt.ddsim.default:2,iqm.emerald.mock:1,amazon.braket.sv1:2
```

A license count limits concurrent Slurm allocations, not the number of circuits
an application can submit. Set it according to the site's workload policy and
service capacity.

The playground runs one administrative monitor per device. It probes QDMI in a
separate process and updates a
[license-only reservation](https://slurm.schedmd.com/reservations.html):

| Probe result       | Scheduling behavior                                           |
| ------------------ | ------------------------------------------------------------- |
| `IDLE` or `BUSY`   | Release the reservation; jobs can acquire the device license  |
| Failure or timeout | Keep all units reserved; new jobs wait with reason `Licenses` |
| Monitor stopped    | Reserve all units until monitoring resumes successfully       |

`BUSY` means the device is operational; the remote service may queue a submitted
job. The reservation affects new allocations. Running jobs and other device
licenses continue unaffected, and waiting jobs occupy no compute nodes.

Run the monitor with site credentials and the workload's QDMI environment,
outside Slurm daemons. Initialize reservations before opening the queue. The
`examples/slurm` systemd units enforce that order, bound probes, restart failed
monitors, and close admission when a monitor stops. Probe output is suppressed
because device diagnostics can contain credentials; service failures appear in
the journal.

:::{important}
Availability is a snapshot for the monitor's identity. A device can fail after
allocation. Controller communication errors require administrator attention: a
failed reservation update can leave the last scheduler state in effect.
:::

This is a deployment policy built from Slurm licenses, reservations, environment
export, and the ordinary QDMI API. The monitor and its settings live in the
playground and can be adapted to the site's requirements.

## Configure compute resources

The playground uses Slurm 25.11 or newer and cgroup v2. Use matching Slurm
versions across hosts. Login nodes need Slurm clients; compute nodes need
`slurmd` and the workload environment. The controller needs device libraries
only if it also hosts the monitors. Add `slurmdbd` for persistent accounting;
static local licenses do not require it.

For CPU and allocated-memory constraints, configure:

```ini
# slurm.conf
ProctrackType=proctrack/cgroup
TaskPlugin=task/cgroup,task/affinity
JobAcctGatherType=jobacct_gather/cgroup
SelectType=select/cons_tres
SelectTypeParameters=CR_CPU_Memory
```

```ini
# cgroup.conf
CgroupPlugin=autodetect
ConstrainCores=yes
ConstrainRAMSpace=yes
ConstrainSwapSpace=yes
```

`autodetect` selects the host's cgroup implementation; cgroup v1 is deprecated.
Set node resources, memory defaults, partitions, accounts, and limits for the
site. Keep Slurm authentication, such as Munge, separate from device
credentials. See the
[Slurm administration guide](https://slurm.schedmd.com/quickstart_admin.html)
and [cgroup configuration](https://slurm.schedmd.com/cgroup.conf.html).
