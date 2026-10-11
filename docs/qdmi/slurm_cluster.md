# A playground for hybrid quantum-classical workflows

Use this cluster to prototype scientific workflows that combine classical
computation with QDMI devices. Submit jobs from a login node, place quantum
stages on dedicated {term}`quantum access nodes <quantum access node>` (QANs),
and inspect their resource use and outcomes through Slurm accounting. Each QAN
has access to all connected QDMI devices and provides CPU and memory for circuit
compilation and quantum job submission. Device implementations share one
software environment; DDSIM, IQM, and Amazon Braket are examples of connections
it can provide.

```{figure} ../_static/slurm-playground.svg
:alt: Login submission through Slurm to classical or quantum access nodes, with shared QDMI devices and persistent accounting.

Separate classical and quantum stages while sharing one QDMI environment.
```

The
[cluster sources](https://github.com/munich-quantum-toolkit/core/tree/main/examples/slurm)
provide these hosts:

| Host                     | Role                                                             |
| ------------------------ | ---------------------------------------------------------------- |
| `login`                  | User shell, Slurm clients, and shared software environment       |
| `controller`             | Scheduling and device availability monitors                      |
| `node` (two by default)  | Classical computation in the default `compute` partition         |
| `qan` (two by default)   | Compilation and access to all devices in the `quantum` partition |
| `accounting`, `database` | Slurm accounting and persistent MariaDB storage                  |

Each classical node advertises two CPUs and 2 GiB of scheduled memory; each QAN
advertises four CPUs and 8 GiB. These containers share the Docker engine's
resources; scaling nodes does not add physical capacity. Jobs run as `mqt-dev`
(UID/GID 10000), with cgroup v2 enforcing allocations. One Python environment
managed by uv at `/opt/venv` contains MQT Core and the installed device
implementations. The administrator maintains the catalogue; users supply their
credentials. See {doc}`slurm` for deployment on an existing cluster.

## Start the cluster and log in

Use Docker Engine on Linux or Docker Desktop on macOS, with Docker Compose.
Apple Silicon builds use native Linux ARM64. Docker builds MQT Core from your
checkout, downloads its LLVM/MLIR toolchain with
[setup-mlir](https://github.com/munich-quantum-software/setup-mlir), and
installs the workload environment with uv. Build tools and source trees stay
outside the runtime image.

The cluster requires a rootful Linux Docker engine with cgroup v2 and privileged
containers for systemd and Slurm resource isolation. Its Compose configuration
uses
[Docker Desktop's systemd setup](https://docs.docker.com/desktop/release-notes/#430),
including the host cgroup namespace. Enhanced Container Isolation must be
disabled for this playground. For small demonstrations, allocate at least four
CPUs and 12 GiB of RAM to Docker; size workloads against these shared resources.
Use this playground for trusted development workloads.

From the MQT Core checkout:

```bash
sh examples/slurm/prepare.sh
docker compose -f examples/slurm/compose.yml up --build -d --wait
docker compose -f examples/slurm/compose.yml exec --user 10000:10000 login bash --login
```

The image supplies one environment on `PATH`; jobs need no activation or package
installation. Subsequent builds reuse the downloaded toolchain and uv caches.
The following commands run in the login shell:

```bash
sinfo
scontrol show licenses
sbatch --wait --partition=quantum --licenses=mqt.ddsim.default --time=00:05:00 \
  --output=/jobs/slurm-%j.out \
  --wrap='srun python /workspace/test/slurm/bell_job.py'
sacct -X --format=JobID,JobName,Partition,State,ExitCode,Elapsed,AllocTRES%80
```

This executes a Bell circuit through DDSIM in the `quantum` partition. Results
appear in `/jobs`, shared by all nodes and backed by `build/slurm/jobs` on the
host. Put application inputs, scripts, and outputs there, or add a shared
application mount. The MQT Core checkout is available read-only at `/workspace`.

(multiple-device-implementations)=

## Connect quantum devices

Add packages to `MQT_CORE_SLURM_PACKAGES` using
[uv requirements](https://docs.astral.sh/uv/pip/dependencies/). For example,
save this as `devices.txt` to build both device implementations from their
current sources:

```{code-block} text
:caption: devices.txt

iqm-qdmi[qiskit] @ git+https://github.com/iqm-finland/QDMI-on-IQM.git@main
amazon-braket-qdmi[qiskit] @ git+https://github.com/munich-quantum-software/amazon-braket-qdmi-device.git@main
```

Then rebuild the shared environment:

```bash
export MQT_CORE_SLURM_PACKAGES="$(cat devices.txt)"
docker compose -f examples/slurm/compose.yml up --build -d --wait
```

Replace `main` with a full commit SHA to fix the source revision. For published
packages, use names such as `iqm-qdmi[qiskit]` and optional version constraints.
To build a local checkout, set `MQT_CORE_SLURM_WORKLOAD` to its absolute parent
directory and use a container path such as `/workload/QDMI-on-IQM[qiskit]` in
the requirements. Source packages are compiled inside Docker for its
architecture.

All implementations and adapters share `/opt/venv`; MQT Core's Qiskit and
PennyLane adapters are included. Installed packages supply device manifests.
Targeted driver sessions open only their selected device, so sessions for
several implementations can coexist.

Before startup, configure concurrency limits and their accounting records in
`build/slurm/slurm.conf`:

```ini
Licenses=mqt.ddsim.default:2,mqt.sc.default:1,iqm.emerald.mock:1,amazon.braket.sv1:2
AccountingStorageTRES=license/mqt.ddsim.default,license/mqt.sc.default,license/iqm.emerald.mock,license/amazon.braket.sv1
```

Put shared endpoints and device presets in `build/slurm/qdmi.json`, mounted at
`/etc/mqt-core/qdmi.json`. These settings extend installed manifests; see
{doc}`configuration`. Keep credentials outside this catalogue and the image.

Monitors use site credentials, independently of job credentials. Put their
references in the private `build/slurm/qdmi.env` file, using systemd
`EnvironmentFile` syntax, for example:

```ini
IQM_TOKENS_FILE=/run/credentials/iqm.json
AWS_SHARED_CREDENTIALS_FILE=/run/credentials/aws
AWS_PROFILE=site-monitor
```

Mount those credential files read-only into the controller through a Compose
overlay. Users authenticate on the login node through each implementation's
usual mechanism. Slurm exports their environment to jobs; credential files must
also be readable by the user on the relevant QAN. Environment variables set for
a daemon do not configure user jobs.

Use `--partition=quantum` for every connected quantum device and request its
license, for example `--licenses=iqm.emerald.mock` or
`--licenses=amazon.braket.sv1`. The single-device examples read the device ID
from `SLURM_JOB_LICENSES`. Every QAN runs the same environment and can open
sessions to several devices. Adding a device requires its implementation,
catalogue entry, license, and credentials; it does not require a new partition
or node type.

Classical nodes use an internal network; QANs, the login node, and the
monitoring controller also have outbound connectivity. Partitions place jobs and
licenses limit concurrency. Device authentication and site network policy govern
access to remote APIs.

## Chain quantum and classical stages

A quantum-assisted AFQMC application can measure a trial state on a QAN, save
its results to `/jobs`, and propagate classical walkers on compute nodes. From
the login shell, submit the application's two batch scripts:

```bash
measurement=$(sbatch --parsable --partition=quantum --licenses=iqm.emerald.mock measure.sh)
sbatch --partition=compute --dependency=afterok:"$measurement" propagate.sh
```

The scripts choose their CPU and memory requirements and agree on an output
path. The classical stage starts only after successful measurement and holds no
quantum license. Application code, scientific validation, and additional
scientific dependencies belong with the application.

For tightly coupled iterations, use a
[heterogeneous allocation](https://slurm.schedmd.com/heterogeneous_jobs.html)
that includes both node types. Slurm permits licenses on its first component and
retains them for the allocation. An `srun` step stays within its allocation; it
cannot move an ordinary compute job into a QAN partition.

## Inspect and operate the cluster

`slurmdbd` records jobs in MariaDB, including user, account, partition, state,
exit code, elapsed time, and allocated license TRES. The `research` account and
`mqt-dev` association are created at startup. Database and controller state
persist in named volumes. `sacct` can therefore inspect completed jobs after
service restarts. Job environments and scripts are not archived. Applications
should save software versions, input digests, and remote job IDs alongside
results, without credentials.

The controller reserves every configured device license before opening the
partitions. Each `qdmi-availability@ID:COUNT.service` checks its device every 30
seconds, with a ten-second probe timeout, and releases the reservation after
success. Stopping a monitor closes its license; failed checks keep new jobs
pending without interrupting running work. From an administrator shell on the
controller:

```bash
systemctl stop qdmi-availability@mqt.ddsim.default:2.service
systemctl start qdmi-availability@mqt.ddsim.default:2.service
```

Enter that shell from the host with
`docker compose -f examples/slurm/compose.yml exec controller bash`. The monitor
maintains the `qdmi-unavailable-ID` reservation; do not create a conflicting
reservation. Inspect both `Free` and `Reserved` in `scontrol show licenses`,
because reserved tokens can still appear in `Free`. A successful check is a
snapshot for the monitor's identity, not a promise of user authorization or
prompt execution at the remote service.

Scale classical and quantum access nodes from the host:

```bash
docker compose -f examples/slurm/compose.yml up -d --wait --scale node=4 --scale qan=3
```

The cluster accepts up to 128 registered nodes. Drain nodes and wait for their
jobs before scaling down. Slurm retains stopped dynamic nodes until they are
removed with `scontrol delete NodeName=...` or the cluster is reset.

`docker compose -f examples/slurm/compose.yml down` stops the cluster while
preserving accounting. To discard the cluster, including its accounting history:

```bash
docker compose -f examples/slurm/compose.yml down --volumes
rm -r build/slurm
```

Images and build caches remain. For independent clusters, choose a Compose
`--project-name`, set `MQT_CORE_SLURM_RUNTIME` to an absolute directory, and
pass that directory to `prepare.sh`.

## Integration tests

`test/slurm/run_integration.py` tests login submission, partition placement,
license contention, unavailable-device admission and recovery, job environment
export, and accounting using the bundled DDSIM and SC implementations.

The IQM and Braket repositories run their own authenticated simulator workloads
in the shared quantum partition, in native and wheel installation modes. They
supply credentials and an optional native build through a Compose overlay and
setup script. Native libraries are compiled independently of the MQT Core image
layer, and Python adapters share the common environment in either mode. IQM uses
the Emerald Resonance mock and Braket uses SV1; these tests can incur service
charges.
