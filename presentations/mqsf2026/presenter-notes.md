# MQSF rehearsal notes

Allow about 35 minutes. Each click changes one narrative step. Motion within a
step ends and holds; clicking during playback advances immediately. Use the
separate notes file privately: the P shortcut displays notes on the projector.

## F1 · Introduction

Introduce the MQSC CTO/co-founder role and the Senior Researcher role at TUM
CDA. Keep the title still. The subtitle is “From the Metal to the User.”

## F2–F4 · From diverse paths to a shared stack

Begin with quantum processors, then a researcher and their classical resources.
Follow workstation → cloud-hosted quantum systems; then cloud compute and HPC.
On-premise quantum access goes through HPC. Add the other user groups. Software
must connect this growing ecosystem without a bespoke solution for every path.

Move classical compute to the bottom. Reveal Qiskit, PennyLane, OpenQASM 3, and
the SDK extension point, with paths from every user group. Then follow programs
and runtime calls into compilation, orchestration, and the shared backend
interface. Modularity, efficiency, and extensibility motivate the published MQSS
architecture.

Replace the boxes with products, from the device interface upward: QDMI device
implementations, the MQT Compiler Collection, and MQSC Orchestrator. Both upper
components use MQT Core. QDMI is an open community standard. MQSC drives much of
the implementation work; it does not own the standard. Credit TUM CDA as MQT's
origin, the MQT community, TUM CAPS/LRZ/MQV, and the wider QDMI community.
Downloads are recorded cumulative downloads, not unique users.

## F5–F6 · A practical hybrid application

Introduce quantum-assisted auxiliary-field Quantum Monte Carlo as a software
workflow, without claiming application expertise or quantum advantage. The
bounded H₂ example keeps the actual circuits readable.

Classical preparation builds the molecular problem and trial. Quantum sampling
collects classical shadows once. Classical walkers then reuse those observations
for overlap estimates. Circle area represents recorded importance weight; the
weighted reduction produces the energy estimate. Attribute the adapted Amazon
Braket example.

## F7 · Execute the hybrid workflow

Reveal the actual payload-generation code, then shrink it as the quantum batch
expands. One QDMI job contains 4,096 programs with 1,024 shots each: 4,194,304
shots in total. Trace basis generation, exact matrix validation, and PennyLane
preprocessing before program conversion. Job states and indexed results are
recorded observations; playback rescales their timing.

Reveal classical propagation: four Slurm ranks on four compute nodes, 128
walkers and 240 imaginary-time steps for each trial. These are real
Docker-playground allocations on one host, not a cluster-scaling experiment.
Wall time and imaginary time are different coordinates.

Animate the quantum-shadow trial, then the Hartree–Fock trial. Interpolation
connects recorded walker states and energy estimates; the walker rings remain
visible at zero weight. Show the complete observed range. The reference is exact
FCI within this basis; the band covers only walker uncertainty conditional on
the shared shadows. No conclusion about quantum advantage follows from these
curves.

## F8–F9 · Compile for a physical target

Keep snapshot 0's actual submitted payload. A physical device adds a topology
and native operations. Emerald supplies 54 qubits and R/CZ constraints. Follow
hardware-independent simplification, placement and routing, native synthesis,
and executable generation.

Click through initial placement, forward/backward refinement, and the final
forward route. Gate highlights and SWAP counters come from recorded operations;
visual interpolation does not add search states. Completed tiles retain visual
reminders of the pipeline.

Each target change updates the device and recorded compiler stages together.
Show IBM's public Nighthawk model and IonQ's all-to-all GPI/GPI2/RZZ model. The
same payload enters each compilation. Metadata is captured in advance, compiled
through local QDMI models, and checked by executing the resulting QIR on DDSIM.
The executable panel changes from IQM's QIR to IBM's Qiskit circuit and QPY
serialization, then IonQ's native OpenQASM 3. Qiskit draws the complete IBM
circuit with idle wires hidden; QPY retains the physical register and layout. No
hardware jobs are submitted. End with the extensible interface and abstract
compiler flow.

## F10–F12 · Why are we unrolling?

Begin only with measured Grover program sizes. At 22 qubits the structured
output is about 1.25 KB; the fully unrolled output is about 2.32 MB. The graph
uses bytes and a logarithmic axis, with no extrapolated points.

Read the compact syndrome source. Reset, measurement feedback, and counted loops
survive QC → QCO → optimization → QC → QIR. This portable example does not
undergo device mapping or native-gate synthesis. Its full QIR entry point is
shown, then reset, measurement/branch, and loop excerpts are highlighted.
Scratch bit `s` is local to the loop. Credit the “Why Are We Unrolling?”
collaborators and jeff as an exchange format for structured quantum programs.
The literature RUS V₃ example is retained in the downloadable evidence.

## F13–F14 · Execute adaptive QIR

DDSIM runs QIR 2.1 Adaptive Profile using a JIT, including the program's
classical computation. Iterative QPE uses a four-qubit U plus a recycled
control: five qubits in total, eight output bits, phase 1/3, and 8,192 shots.
U = P(π/6)⊗⁴ acts on |1111⟩. The outer loop resets the control for each bit;
the nested loop repeats controlled-U 2^(7−k) times. P(θ) applies the feedback
accumulated so far, and measurement updates θ ← θ/2 − (π/2) result[k] for the
next round. These loops and runtime calculations remain in the compiled QIR.

The diagram comes from the exact submitted OpenQASM syntax, without expanding
the loops. Qiskit export is unavailable for this program's dynamic bounds and
classical indexing. The evidence records that limit and retains the real
QC/QCO/QIR artifacts; it does not replace the program with a Qiskit surrogate.

Four native workers execute in parallel. Their progress and the histogram share
actual 64-shot batch-completion observations with the QDMI/JIT trace. Each
arrival retains its index in the returned result. The replay is slowed for
inspection and includes instrumentation overhead; it is not a throughput
benchmark. QCO measurement-branch sharing is a separate path from adaptive QIR.

## F15–F17 · Features and conclusion

Build the feature overview: OpenQASM 3.1, QIR 2.1, Qiskit, PennyLane, packaged
binaries, and developer resources. Then zoom back out to the shared
architecture. Common interfaces, compilation, and orchestration connect
different users, classical infrastructure, and devices.

F16.6 ends with the Unitary Foundation QOSS survey invitation. Leave its QR
visible long enough to scan: about 20 minutes to complete, open through 29
October 2026. Everyone who builds or uses quantum software can contribute.

“Built with the Community. Driven into practice by MQSC.”

Leave the product tower and community credits visible for discussion; mq.sc and
the social links remain in the shared footer.
