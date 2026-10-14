# Compiler benchmarks

This suite measures compilation time, native two-qubit gate count, and native
gate depth using the structured benchmarks in `mqt.core.bench`. A static MQT
dashboard adds circuit and connectivity filters, revision comparisons, three
charts, a measurement table, and a JSON download. The original asv explorer
remains available under `/asv/`.

## Corpus and metric definitions

The suite uses Core's GHZ (star topology, Z readout), standard QFT (period
exponent `qubits // 2`), and static Bernstein–Vazirani (all-one hidden string)
generators. Each runs with 4, 8, and 16 total qubits on fixed all-to-all and
line targets. Bernstein–Vazirani reserves one of these qubits for its flag, so
its hidden string and output contain `qubits - 1` bits. No external circuit
dataset or live device is required.

Targets support U, CX, and measurement. The Base QIR payload contract requires
static quantum loops to be unrolled. Compilation stops at native QC; QIR
lowering, serialization, execution, and quality analysis are excluded.

| Metric                                 | Definition                                                                                                                                             |
| -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `Compilation.time_compile`             | Wall time to copy generated QC input, convert to QCO, compile for the target, convert back to QC, and dispose of the result.                           |
| `CircuitQuality.track_two_qubit_count` | Native two-qubit gate count in the compiled QC program.                                                                                                |
| `CircuitQuality.track_depth`           | Longest dependency path of U/CX gates, each with unit duration. Terminal measurements add no depth. Barriers synchronize wires without adding a layer. |

Generation, target construction, and Python imports happen outside the timed
region. Every timing invocation compiles a fresh copy. Quality benchmarks use
the same generator parameters and compilation path, then inspect the output with
the existing QC gate counter and Qiskit depth implementation. No Qiskit
transpilation is performed. Unsupported native gates or nonterminal measurements
fail the quality benchmark instead of yielding misleading depths.

Dynamic benchmark variants need a separate definition of executed gate count and
depth and are outside this prototype. Core's benchmark manifest and case ID
describe the semantic input; generator changes can still change its IR. Keep the
generated IR fixed or version the suite when comparing compiler-only changes
across generator revisions.

## Run locally

Use Python 3.13, a recent uv release compatible with the repository lockfile,
and the LLVM/MLIR version required by Core (23.1.0 for the initial prototype).
Set `MLIR_DIR` to the installed distribution's `lib/cmake/mlir` directory. Run
these commands from the repository root:

```sh
uv sync --python 3.13 --locked --only-group dev --only-group benchmark
uv sync --locked --inexact --no-dev --no-build-isolation-package mqt-core \
  --config-setting=cmake.build-type=Release
uv run --no-sync python -m pytest -n 0 test/compiler_benchmarks
node --test test/compiler_benchmarks/dashboard.test.cjs
uv run --no-sync asv check -E existing
uv run --no-sync asv machine --yes
uv run --no-sync asv run -E existing --quick --show-stderr
```

The quick run checks all cases but saves no measurements. The standard workflow
builds the requested compiler revisions in isolated asv environments, records
individual timing samples, and generates both dashboards:

```sh
uvx nox -s compiler-benchmarks
uvx nox -s benchmark-dashboard
```

Open <http://127.0.0.1:8080>. Keep the serving terminal open; Ctrl+C stops it.
Use `uvx nox -s benchmark-dashboard -- serve --port 8081` if the port is
occupied. Serving uses Python's standard HTTP server, binds only to localhost,
and does not rebuild or rerun benchmarks.

To keep the preview independent of a terminal session, start the installed
Python environment in the background from the repository root:

```sh
nohup .venv/bin/python -m benchmarks.dashboard serve > .asv/preview.log 2>&1 < /dev/null &
```

The serving process must remain running for the local URL to work. No login
service or restart after a machine reboot is installed.

Raw results live under `.asv/results/`; the complete static website is
`.asv/html/`, with the MQT entry point at `index.html` and asv at
`asv/index.html`. These files are ignored by Git and survive deletion of
`build/`. asv's disposable compiler environments remain in `build/asv/env/`.
Machine metadata lives in `~/.asv-machine.json`. Back up the results directory:
generated HTML is not a replacement for the measurement database.

To rebuild the website from existing results without compiling or measuring:

```sh
uvx nox -s benchmark-dashboard -- build
```

The equivalent direct commands are `python -m benchmarks.dashboard build` and
`python -m benchmarks.dashboard serve`, using an environment containing asv. Do
not point `asv preview` at the MQT overview: it serves the nested asv site.

The overview compares only matching machines, environments, case parameters, and
benchmark versions. Failed or missing values remain unavailable instead of
becoming zero. A missing or zero baseline does not produce an infinite percent
change. Timing cells expose the sample count and interquartile range in their
tooltips; the JSON download includes individual samples. One revision produces
an explicit baseline-only view. Filters can also be selected through WebMCP when
the browser supports it.

## Compare revisions

The configuration tracks this checkout's `HEAD`. asv can build each requested
revision into an isolated uv environment, using the checked-in benchmark suite:

```sh
uvx nox -s compiler-benchmarks -- 'HEAD~2..HEAD' --interleave-rounds
uv run --no-sync asv continuous BASE_REVISION HEAD
uvx nox -s benchmark-dashboard -- build
```

The Nox benchmark session defaults to `HEAD^!` and forwards positional arguments
to `asv run`. `asv continuous` compares a specific pair and can exit nonzero
when measurements worsen; this is not necessarily a build failure. Rebuild the
dashboard afterward to display its saved results. asv's regression page remains
available for longer histories; two points alone are not a calibrated gate.

Replace `BASE_REVISION` with a commit that provides the current Core benchmark
and compiler APIs. Keep LLVM/MLIR, the C++ compiler, Python, Qiskit, and build
flags fixed. asv does not install LLVM/MLIR or automatically lock all build
dependencies to `uv.lock`. The configuration fixes Qiskit and requests a Release
wheel; archive environment details alongside any retained comparison.

Mapping currently defaults to seed 42 and a trial count derived from the
available CPUs. Compare runs on the same machine with the same CPU affinity;
different trial counts may affect both time and circuit quality. The suite uses
two process rounds with three samples per timing case. These settings make a
small benchmark suite, not a calibrated regression gate. Build and lint work
should finish before collecting timings. Local runs do not publish results.

## Weekend history on a dedicated Linux machine

The DGX command fetches `origin/main` and measures every missing first-parent
commit, oldest first. The supplied settings start
**at the 4.0.0 release, inclusive**, and use the same commit as the fixed
reference: `8d9bcfb031fa0fe2769a852cdf8d66017d24abe8` (`v4.0.0`). Commits inside
merged branches are excluded; their merge commit is measured. GitHub-hosted
runners do not collect measurements.

Use a clean, dedicated checkout at `~/mqt-core-benchmarks` containing this
benchmark implementation. Keep that reviewed harness revision fixed; the batch
fetches source history without pulling changes into the harness checkout. Update
the harness deliberately after review. Install native LLVM/MLIR 23.1.0, Clang,
and uv. DGX Spark requires ARM64 binaries. The native build and timing stability
must be validated on the DGX before relying on its results.

Copy the worker settings outside the checkout and adjust the toolchain paths:

```sh
mkdir -p ~/.config/mqt-core-benchmarks
cp benchmarks/systemd/worker.json ~/.config/mqt-core-benchmarks/worker.json
cp benchmarks/systemd/worker.env ~/.config/mqt-core-benchmarks/worker.env
chmod 600 ~/.config/mqt-core-benchmarks/worker.env
```

Select a fixed logical CPU in `worker.json`; `0` is an example, not a claim that
it is the best DGX core. Keep the machine label, CPU, power settings and
toolchain fixed. Reserve a quiet measurement period without other compute jobs.
Affinity pins the benchmark process but does not isolate shared caches, memory,
or power. The runner checks affinity in the benchmark process. The mapper then
sees one available CPU and uses its default seed 42.

From the dedicated checkout, load the environment and install the locked worker
dependencies once. These commands do not register an Actions runner:

```sh
set -a
. ~/.config/mqt-core-benchmarks/worker.env
set +a
uv sync --locked --only-group benchmark --python 3.13.7
uv run --no-sync python -m benchmarks.runner batch \
  --settings ~/.config/mqt-core-benchmarks/worker.json
```

The reference is measured at batch start, even when no new main commits exist.
Every candidate is measured together with a fresh reference, using four
interleaved rounds of five samples per timing case. Reference builds, candidate
builds, and measurements run sequentially. An exclusive lock prevents two
commands in the same checkout from overlapping.

Complete observations are checkpointed immediately. A failed candidate remains
eligible for retry while later candidates continue. A failed initial reference
stops the batch. A failed or gracefully interrupted batch exits unsuccessfully;
its status and attempts are saved under `.asv/results/batches/`. The next batch
skips complete candidates from the same series. Raw asv files, machine
configuration, commit IDs, benchmark source version, per-observation benchmark
definitions, generator-source hashes, and individual samples remain under
`.asv/results/`. Each observation has a unique ID; reruns never replace earlier
samples. Abrupt power loss can prevent a final batch manifest, but complete
observations remain checkpoints for the next run.

A series identifies the machine, OS, CPU allocation and available CPU power
settings, compilers, LLVM, Python, uv, asv, build configuration, benchmark
source, and reference. Changing these settings starts another series and
remeasures its missing history from the configured start. Build isolation still
resolves Core's build dependencies for each revision; the runner does not claim
that all build dependencies across releases are identical. Inspect the recorded
environment and reference drift when interpreting a change.

The dashboard displays main history, missing measurements, batch failures, and
repeated reference measurements in time order. It does not automatically
normalize timings or apply a regression gate. Changed generator sources suppress
comparison deltas and break trend lines because the input may have changed. The
fixed-reference table helps distinguish machine drift from compiler changes.

### Publish data and refresh RTD

Add `--publish` to the batch command, or publish retained results separately:

```sh
uv run --no-sync python -m benchmarks.runner publish
```

Publication appends **all complete observations and batch records** to the
data-only `benchmark_results` branch on `origin`. Configure Git identity and
credentials that allow this branch write. No force push is used. Existing sample
files are immutable; partial measurements are excluded. A failed upload can be
retried without repeating the measurements. Keep a backup of `.asv/results/`;
generated HTML does not replace raw results.

Set `RTD_API_TOKEN` in the local environment file to request a fresh `mqt-core`
RTD `latest` build after publication. Without this token, the next normal RTD
build imports the data. If RTD fails after a successful upload, retry `publish`;
it does not duplicate data. Credentials stay outside the repository.

The dashboard stays in the Core docs. RTD imports JSON from `benchmark_results`
at build time and renders static assets; no separate repository, database
server, or measurement job on RTD is required. Release docs retain their saved
snapshot.

### Enable the Saturday timer later

The supplied user service expects `~/mqt-core-benchmarks`, uv at
`~/.local/bin/uv`, and the configuration files above. Adjust these paths on the
DGX. Validate one manual batch and its upload before enabling the timer:

```sh
mkdir -p ~/.config/systemd/user
cp benchmarks/systemd/mqt-core-benchmarks.* ~/.config/systemd/user/
systemd-analyze --user verify ~/.config/systemd/user/mqt-core-benchmarks.service \
  ~/.config/systemd/user/mqt-core-benchmarks.timer
systemd-analyze calendar 'Sat *-*-* 02:00:00 Europe/Berlin'
systemctl --user daemon-reload
systemctl --user enable --now mqt-core-benchmarks.timer
systemctl --user list-timers mqt-core-benchmarks.timer
```

The timer starts each Saturday at 02:00 Europe/Berlin. A condition prevents
weekday starts after suspend; runs have a 36-hour timeout. Missed schedules do
not run immediately after reboot. Enable user lingering with
`loginctl enable-linger "$USER"` if the service must run while logged out; this
may require administrator approval. Keep the DGX awake during the reserved slot.
Timer and credential installation are operator actions, not part of a local
dashboard build.

Inspect a run with `journalctl --user -u mqt-core-benchmarks.service` and its
batch JSON. To pause scheduling, use
`systemctl --user disable --now mqt-core-benchmarks.timer`. This does not stop
an already active service. The direct `batch` command remains available on any
day.

For a one-off paired experiment outside the main history, use
`python -m benchmarks.runner run --machine LABEL --cpu CPU --baseline SHA`. This
measures the current checkout against that ancestor. Omit `--baseline` to use
the previous saved candidate, or `HEAD^` on the first run.

## Read the Docs

The dashboard is embedded in `docs/compiler_benchmarks.md` and inherits the Furo
documentation theme, including its navigation and light/dark colors.
`uvx nox --non-interactive -s docs` generates the overview and original asv
explorer from the checked-in `benchmarks/results/` snapshot before Sphinx runs.
Sphinx copies the static assets through `html_extra_path`. The RTD configuration
runs this session, fetches full Git history, and uploads its HTML output; it
needs no additional deployment service.

For `latest`, the RTD pre-build step imports `results/` from `benchmark_results`
into the snapshot directory. Before that branch exists, it keeps the checked-in
snapshot. Other documentation versions and local builds keep their saved data.
The importer accepts only regular JSON files and does not execute branch
content.

The page is `compiler_benchmarks.html` within each RTD documentation version.
Its data and the asv explorer live in `compiler-benchmarks/`, using relative
URLs so that version prefixes continue to work. The measured commit IDs are
independent of the documentation version. RTD renders saved results; it does not
execute performance benchmarks on its documentation build machines.

To refresh the documentation snapshot after a reviewed benchmark run:

```sh
cp -R .asv/results/. benchmarks/results/
uvx nox --non-interactive -s docs
```

Keep `benchmarks.json`, machine metadata, and revision result files together.
The initial snapshot contains revisions `626e8b5e` and `2caec49d`, built with
LLVM/MLIR 23.1.0 and Apple Clang 21.0.0 in Release mode, using Python 3.13 and
Qiskit 2.5.2 on one machine. It demonstrates the workflow and is not a
production regression gate. Raw files retain timing samples and the benchmark
source hash.
