---
file_format: mystnb
kernelspec:
  name: python3
mystnb:
  number_source_lines: true
---

# PennyLane interface for gate-based QDMI devices

The {py:mod}`mqt.core.plugins.pennylane` module connects PennyLane to gate-based
QDMI devices through a native C++ implementation of Catalyst's device interface.
Ordinary QNodes and compiled Catalyst programs share gate translation, MQT
target compilation, QDMI submission, and readout decoding. Ordinary `@qp.qnode`
calls continue to work without Catalyst or `@qp.qjit`.

The selected device must provide target metadata usable by MQT Core's compiler,
accept QIR or OpenQASM 3 programs, and return finite-shot computational-basis
results. OpenQASM 2-only devices are unsupported. The examples below use the
local [DD-based simulator device](ddsim_device.md) included with MQT Core and
require no credentials or remote resources.

Install MQT Core with the optional PennyLane dependency into the active
environment:

```bash
uv pip install "mqt-core[pennylane]"
```

## Quickstart

A Bell-state circuit illustrates device discovery by stable ID, circuit
conversion, execution, and finite-shot result reconstruction.

```{code-cell} ipython3
import pennylane as qp

bell_device = qp.device("mqt.ddsim.default", wires=2, job_parameters={"custom1": 7})


@qp.qnode(bell_device, shots=1000)
def bell_state():
    qp.Hadamard(0)
    qp.CNOT(wires=[0, 1])
    return qp.counts(wires=[0, 1])


bell_counts = bell_state()
assert sum(bell_counts.values()) == 1000
assert set(bell_counts) <= {"00", "11"}
print({str(key): int(value) for key, value in sorted(bell_counts.items())})
```

Only the computational-basis states $00$ and $11$ have nonzero probability, up
to finite-shot fluctuations in their relative frequencies.

## Native compilation

PennyLane preprocesses ordinary quantum tapes using the plugin's capability
configuration. Its transforms decompose unsupported operations, split
incompatible measurement bases, diagonalize observables, and expand parameter
broadcasting. The Python device replays the resulting gates into the native
recorder and reconstructs PennyLane measurement results from its samples.

The recorder builds MQT circuit IR. MQT Core then synthesizes gates and routes
the circuit for the selected QDMI target. Programs therefore need not already
use the provider's native gate names or physical qubit placement. Invalid or
incomplete target metadata and unsupported target configurations fail before a
job is created. Every program in an ordinary execution batch is prepared before
submission begins.

The compiler prefers Adaptive QIR binary/text, then OpenQASM 3, then Base QIR
binary/text, according to the formats advertised by the device. Routing
preserves logical classical output positions. Samples and requested wire subsets
follow PennyLane's wire order, independently of physical placement.

QDMI waits and sample/count retrieval release the Python GIL. If ordered samples
are unavailable, the adapter can reconstruct a sample multiset from counts; this
fallback does not recover temporal shot order.

## Catalyst compilation

Install the additional dependency on a supported Catalyst platform:

```bash
uv pip install "mqt-core[pennylane]" "pennylane-catalyst>=0.15"
```

Catalyst supports Linux x86_64/aarch64 and macOS arm64. Ordinary PennyLane
execution remains available on other MQT Core platforms. See the
[Catalyst installation guide](https://docs.pennylane.ai/projects/catalyst/en/stable/dev/installation.html)
for platform requirements.

Explicit `qp.qjit` compilation calls the same native device directly:

```python
import pennylane as qp

compiled_device = qp.device("mqt.ddsim.default", wires=2)


@qp.qjit(capture=False)
@qp.qnode(compiled_device, shots=1000, diff_method="parameter-shift")
def compiled_expectation(angle):
    qp.RY(angle, 0)
    qp.CNOT(wires=[0, 1])
    return qp.expval(qp.Z(0))


value = compiled_expectation(0.4)
```

Use `capture=False` with current released and development Catalyst versions.
Their capture pipeline does not yet implement all device preprocessing needed by
this integration. Ordinary QNodes are never implicitly compiled with Catalyst.

Compiled execution requires contiguous integer wire labels starting at zero and
finite shots. Native expectation and probability operations support Catalyst's
parameter-shift differentiation. Native device gradients, analytic execution,
interactive mid-circuit measurement, and variance gradients are unsupported.
PennyLane's ordinary deferred-measurement workflow remains available as
described below. Catalyst's own restrictions also apply: counts use arrays
instead of Python dictionaries, shot vectors are limited to sample measurements,
and parameter broadcasting requires an explicit supported transform. See
[Catalyst's limitations](https://docs.pennylane.ai/projects/catalyst/en/stable/dev/sharp_bits.html).
Compiled counts and probabilities allocate $2^{n}$ entries for $n$ measured
wires before calling the device. Use samples for large measured registers.

The Python device retains an in-process native context containing the QDMI
session and job parameters. Compiled code contains only an opaque context ID;
credentials are not copied into compiler IR. Keep the device alive while using
compiled functions. Saved executables cannot restore this context in another
process. Each compiled measurement call blocks until its QDMI result is
available; repeated terminal measurements of the unchanged circuit share that
readout. Compiled calls do not use the Python batch recovery, retry, or tracker
interfaces. Catalyst-level random seeds are unsupported; provider-specific job
parameters remain available.

## End-to-end use case: finite-shot MaxCut QAOA

Consider MaxCut on the fixed graph with $E=\{(0,1),(0,2),(1,2),(2,3)\}$.

```{code-cell} ipython3
from collections import Counter
import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
```

```{code-cell} ipython3
:tags: [remove-input]

%config InlineBackend.figure_formats = ['svg']

plt.rcParams.update(
    {
        "axes.facecolor": "#f8f9fb",
        "figure.facecolor": "#f8f9fb",
        "savefig.facecolor": "#f8f9fb",
        "text.color": "#202124",
        "axes.labelcolor": "#202124",
        "axes.edgecolor": "#5f6368",
        "xtick.color": "#3c4043",
        "ytick.color": "#3c4043",
    }
)
```

```{code-cell} ipython3
:tags: [hide-input]
:mystnb: {image: {alt: "Four-node MaxCut graph with edges 01, 02, 12, and 23."}}

graph = nx.Graph([(0, 1), (0, 2), (1, 2), (2, 3)])
positions = {
    0: (-1.0, 0.75),
    1: (-1.0, -0.75),
    2: (0.25, 0.0),
    3: (1.35, 0.0),
}

figure, axis = plt.subplots(figsize=(5.5, 3.3))
nx.draw_networkx(
    graph,
    pos=positions,
    ax=axis,
    node_color="#4c78a8",
    edge_color="#5f6368",
    font_color="white",
    node_size=850,
    width=2,
)
axis.set_title("Four-node MaxCut instance")
axis.set_axis_off()
figure.tight_layout()
```

The cost Hamiltonian is the negative cut size, and the mixer is the sum of
Pauli-X operators. The ansatz applies Hadamard gates followed by evolution under
each Hamiltonian, parameterized by $\gamma$ and $\beta$. Terms within each
Hamiltonian commute, so one Trotter step suffices.

```{code-cell} ipython3
cost_hamiltonian = sum(
    0.5 * (qp.Z(u) @ qp.Z(v) - qp.Identity(u)) for u, v in graph.edges
)
mixer_hamiltonian = sum(qp.X(wire) for wire in graph.nodes)


def ansatz(parameters):
    for wire in graph.nodes:
        qp.Hadamard(wire)
    qp.ApproxTimeEvolution(cost_hamiltonian, parameters[0], 1)
    qp.ApproxTimeEvolution(mixer_hamiltonian, parameters[1], 1)


qaoa_device = qp.device("mqt.ddsim.default", wires=4, job_parameters={"custom1": 7})


@qp.qnode(qaoa_device, shots=1000, diff_method="parameter-shift")
def cost(parameters):
    ansatz(parameters)
    return qp.expval(cost_hamiltonian)


@qp.qnode(qaoa_device, shots=1000)
def sample(parameters):
    ansatz(parameters)
    return qp.sample(wires=range(4))
```

The calculation below evaluates the initial parameter-shift gradient and
performs four gradient-descent updates. Each objective value is an independent
finite-shot estimate; the resulting sequence is therefore not expected to be
monotonic.

```{code-cell} ipython3
parameters = qp.numpy.array([0.5, 0.5], requires_grad=True)
optimizer = qp.GradientDescentOptimizer(stepsize=0.15)
jobs_before = qaoa_device.submitted_jobs
started = time.monotonic()

objective_values = [float(cost(parameters))]
initial_gradient = np.asarray(qp.grad(cost)(parameters), dtype=float)

for _ in range(4):
    parameters = optimizer.step(cost, parameters)
    objective_values.append(float(cost(parameters)))

samples = np.asarray(sample(parameters), dtype=np.int8)
submitted_jobs = qaoa_device.submitted_jobs - jobs_before
elapsed = time.monotonic() - started

print(f"Initial gradient: {initial_gradient}")
print(f"Final parameters: {np.asarray(parameters)}")
print(f"QDMI jobs submitted: {submitted_jobs}")
print(f"Elapsed time: {elapsed:.3f} s")
```

Parameter-shift expands one gradient evaluation into several shifted tapes.
Compatible tapes share a native multi-program QDMI job when the provider
supports it. Otherwise the device submits independent jobs. All submissions in
one PennyLane batch happen before waiting for ordered results, allowing
asynchronous QDMI implementations to execute them concurrently.

The sampled bit strings determine candidate bipartitions. The cut value is the
number of graph edges whose endpoints have different bit values.

```{code-cell} ipython3
def bitstring(sample_row):
    return "".join(str(int(bit)) for bit in sample_row)


def cut_value(candidate):
    return sum(
        candidate[first] != candidate[second] for first, second in graph.edges
    )


sample_counts = Counter(bitstring(row) for row in samples)
best_bitstring = max(
    sample_counts,
    key=lambda candidate: (cut_value(candidate), sample_counts[candidate], candidate),
)
best_cut = cut_value(best_bitstring)

print(f"Best sampled partition: {best_bitstring}")
print(f"Cut edges: {best_cut} of {graph.number_of_edges()}")
```

The following panels show the noisy objective estimates, the empirical
bit-string distribution, and the highest-cut partition observed in the final
sample. Orange edges cross that partition.

```{code-cell} ipython3
:tags: [hide-input]
:mystnb: {image: {alt: "QAOA objective estimates, final counts, and best sampled graph partition."}}

figure, axes = plt.subplots(3, 1, figsize=(7, 10))

axes[0].plot(
    range(len(objective_values)),
    objective_values,
    marker="o",
    color="#4c78a8",
)
axes[0].set(
    xlabel="Optimizer update",
    ylabel=r"$\langle H_C \rangle$",
    title="Noisy finite-shot objective estimates",
    xticks=range(len(objective_values)),
)
axes[0].grid(alpha=0.25)

ordered_bitstrings = sorted(sample_counts)
axes[1].bar(
    ordered_bitstrings,
    [sample_counts[candidate] for candidate in ordered_bitstrings],
    color="#4c78a8",
)
axes[1].set(
    xlabel="Bit string",
    ylabel="Observed count",
    title="Final sampled distribution",
)
axes[1].tick_params(axis="x", rotation=60)

node_colors = [
    "#4c78a8" if best_bitstring[node] == "0" else "#e45756"
    for node in graph.nodes
]
edge_colors = [
    "#f28e2b"
    if best_bitstring[first] != best_bitstring[second]
    else "#9aa0a6"
    for first, second in graph.edges
]
edge_widths = [
    3.2 if best_bitstring[first] != best_bitstring[second] else 1.5
    for first, second in graph.edges
]
nx.draw_networkx(
    graph,
    pos=positions,
    ax=axes[2],
    node_color=node_colors,
    edge_color=edge_colors,
    width=edge_widths,
    font_color="white",
    node_size=850,
)
axes[2].set_title(f"Best sampled cut: {best_cut} edges")
axes[2].set_axis_off()

figure.tight_layout()
```

## Direct generic construction

Stable entry points such as `mqt.ddsim.default` provide the simplest
construction. A device integration package can register a
{py:class}`~mqt.core.plugins.pennylane.device.QDMIDevice` subclass under the
stable ID of another QDMI device. Applications may also construct the generic
class directly:

```python
import pennylane as qp

from mqt.core.plugins.pennylane import QDMIDevice

device_id = "stable ID reported by the QDMI driver"
device = QDMIDevice(
    device_id=device_id,
    wires=["a", "b", "c", "d"],
    session_parameters={
        "token": "...",
        "base_url": "https://device.example",
    },
    job_parameters={
        "custom1": "device-specific job value",
    },
)
```

Automatic retries are disabled (`max_retries=0`). To retry confirmed failed jobs
up to three times, set the option when creating the device:

```python
device = qp.device("mqt.ddsim.default", wires=2, max_retries=3)
```

Retries create additional executions and can incur charges. Successful jobs are
reused; errors that do not confirm job failure require
[explicit recovery](#recovering-results-after-a-failure).

An integration can return an already-open device. Pass that handle directly to
the generic class. Do not repeat session parameters because the session already
exists:

```python
from mqt.core.plugins.pennylane import QDMIDevice
from mqt.core.qdmi import builtin_driver

device = QDMIDevice(
    device=builtin_driver.open_device("mqt.ddsim.default"),
)
```

Arbitrary PennyLane wire labels map to logical integer wires before native
compilation. The compiler handles gate synthesis and hardware placement.
Shot-vector copies, batches, and parameter-shift tapes retain their input order
in the returned results. The PennyLane call remains synchronous, and every
execution requires finite shots. Set shots on the QNode or use `qp.set_shots` to
override them. Devices do not accept a `shots` argument or provide a default
shot count. Omitting finite shots fails before submission.

Use `with qp.Tracker(device) as tracker:` to record accepted program executions
(`executions`), requested shots, batches, and the number of tapes in each batch
(`batch_len`). Shot-vector copies count as separate executions even when grouped
in one native job. Tracking includes accepted jobs whose execution later fails.

Mid-circuit measurements use PennyLane's deferred-measurement transform. Reset
and feedback may require unused device wires; custom wire labels are supported.
The plugin rejects programs requiring more wires than are available and rejects
postselection. Explicit `one-shot` and `tree-traversal` requests are
unsupported; the plugin does not infer native dynamic-circuit support from gate
names.

## Supported gate-level scope

The native gate set includes identity and Pauli gates; H, S, T, SX; RX, RY, RZ,
phase shift, global phase, and U gates; controlled Pauli, rotation, and phase
gates; Toffoli, SWAP, CSWAP, ISWAP, ECR; and Ising XX, YY, and ZZ rotations.
Adjoints and controls, including negative controls, use the native recorder.
PennyLane decomposes other operations to this set; MQT Core then lowers it to
the target's native operations and topology.

The adapter uses finite-shot samples for probabilities, counts, expectations,
and variances. Native Z observables and products let Catalyst retain expectation
and probability operations during differentiation; upstream transforms supply
basis changes and Hamiltonian splitting. Pulse programming, arbitrary Hermitian
observables, and analytic state or probability results are unsupported.

## Recovering results after a failure

If execution fails, the device keeps accepted jobs and successful results.
Retrieve the batch handle from `error.job` on a
{py:class}`~mqt.core.plugins.pennylane.exceptions.PennyLaneExecutionError`, or
from `device.last_job` after an interruption inside `device.execute()`. Each
`execute()` call clears this attribute until its batch is prepared. QNode
preprocessing happens before `execute()` and does not update it.

To recover a QNode's measurement values, save its executable tapes and
postprocessor **before execution**. For the Bell-state QNode above, use
PennyLane's
[`construct_batch`](https://docs.pennylane.ai/en/stable/code/api/pennylane.workflow.construct_batch.html):

```{code-cell} ipython3
tapes, postprocess = qp.workflow.construct_batch(bell_state, level="device")()
results = bell_device.execute(tapes)
counts = postprocess(results)[0]
```

If this execution stops, keep the same Python session. After resolving a
connection or result-read problem, recover the measurement values from the
retained jobs:

```{code-cell} ipython3
job = bell_device.last_job
assert job is not None  # A batch was prepared by execute().
entries = job.collect()
counts = postprocess(job.result())[0]
```

`collect()` reads accepted work without starting new executions; its entries
retain attempts, results, and errors. If needed, use `job.submit()` for
untouched inputs or `job.resubmit([i])` to replace a failed or cancelled
attempt, then retrieve the result again. Successful results are reused. See
{py:class}`~mqt.core.plugins.pennylane.job.PennyLaneJob` for uncertain
submissions and explicit cancellation.

This workflow recovers forward measurements, including broadcasts and shot
vectors. It does not restore arbitrary QNode return containers or resume an
interrupted gradient or optimizer.

### Native multi-program jobs

The device groups tapes and shot partitions with the same program format and
shot count into native QDMI jobs. Devices that reject a program list use
independent submissions without waiting between them. Results retain the
original tape and shot-partition order. `submitted_jobs` counts native jobs;
PennyLane's execution tracker counts individual program executions and shots.
Recovery retains successful programs and retries only confirmed failures.
