# MQSF rehearsal notes

Allow about 35 minutes. Each click changes one narrative step. Motion within a
step ends and holds; clicking during playback advances immediately. Use the
separate notes file privately: the P shortcut displays notes on the projector.

## 1 · Introduction

Introduce the MQSC CTO/co-founder role and the Senior Researcher role at TUM
CDA. Keep the title still. The subtitle is “From the Metal to the User.”

## 2 · From diverse paths to a shared stack

Begin with quantum processors, then a researcher and their classical resources.
Follow workstation → cloud-hosted quantum systems; then cloud compute and HPC.
On-premise quantum access goes through HPC. Add the other user groups. Software
must connect this growing ecosystem without a bespoke solution for every path.

Move classical compute to the bottom. Build Qiskit's program/runtime split,
compiler infrastructure, orchestration, and shared backend interface. Add
devices, classical hosting, OpenQASM, and PennyLane. Modularity, efficiency, and
extensibility motivate the published MQSS architecture.

Replace the boxes with products, from the device interface upward: QDMI device
implementations, the MQT Compiler Collection, and MQSC Orchestrator. Both upper
components use MQT Core. QDMI is an open community standard. MQSC drives much of
the implementation work; it does not own the standard. Credit TUM CDA as MQT's
origin, the MQT community, TUM CAPS/LRZ/MQV, and the wider QDMI community.
Downloads are recorded cumulative downloads, not unique users.

## 3 · A practical hybrid application

Introduce quantum-assisted auxiliary-field Quantum Monte Carlo as a software
workflow, without claiming application expertise or quantum advantage. The
bounded H₂ example keeps the actual circuits readable.

Classical preparation builds the molecular problem and trial. Quantum sampling
collects classical shadows once. Classical walkers then reuse those observations
for overlap estimates. Circle area represents recorded importance weight; the
weighted reduction produces the energy estimate. Attribute the adapted Amazon
Braket example.

## 4 · Execute the hybrid workflow

Reveal the actual payload-generation code, then shrink it as the quantum batch
expands. One QDMI job contains 4,096 programs and 1,048,576 shots. The replay
starts at submission, omitting preparation. Job states and indexed results are
observations, not simulated durations.

Reveal classical propagation: four Slurm ranks on two compute nodes, 128 walkers
and 480 imaginary-time steps for each trial. These are real Docker-playground
allocations on one host, not a cluster-scaling experiment. Wall time and
imaginary time are different coordinates.

The first curve uses quantum-shadow overlaps. The next click overlays the
Hartree–Fock trial. Show the complete observed range. The reference is exact FCI
within this basis; the band covers only walker uncertainty conditional on the
shared shadows. No conclusion about quantum advantage follows from these curves.

## 5 · Compile for a physical target

Keep snapshot 0's actual submitted payload. A physical device adds a topology
and native operations. Emerald supplies 54 qubits and R/CZ constraints. Follow
hardware-independent simplification, placement and routing, native synthesis,
and executable generation.

Click through initial placement, forward/backward refinement, and the final
forward route. Gate highlights and SWAP counters come from recorded operations;
visual interpolation does not add search states. Completed tiles retain visual
reminders of the pipeline.

Change the device card, then recompile on another click. Show IBM's public
Nighthawk model and IonQ's all-to-all GPI/GPI2/RZZ model. The same payload
enters each compilation. Metadata is captured in advance, compiled through local
QDMI models, and checked by executing the resulting payload on DDSIM. No
hardware jobs are submitted. End with the extensible interface and abstract
compiler flow.

## 6 · Inside the compiler

Open the machine room: actual OpenQASM → QC references → QCO values. Explain
quantum dependencies and the reuse of classical MLIR dialects. Windows retain
original line numbers; full generated programs are in the evidence bundle.

Show real cancellation/rotation merging, a bounded recorded A* search, and fused
two-qubit interactions in the Weyl chamber. The chamber shows measured matrix
coordinates, not a simulated synthesis-search trajectory. Finish with real QIR
and return to the abstract architecture.

## 7 · Why are we unrolling?

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

## 8 · Execute adaptive QIR

DDSIM runs QIR 2.1 Adaptive Profile using a JIT, including the program's
classical computation. Iterative QPE uses four logical qubits, eight output
bits, phase 1/3, and 8,192 shots. Controlled-U powers remain counted loops.

Four native workers execute in parallel. Their progress and the histogram share
actual 64-shot batch-completion observations with the QDMI/JIT trace. Each
arrival retains its index in the returned result. The replay is slowed for
inspection and includes instrumentation overhead; it is not a throughput
benchmark. QCO measurement-branch sharing is a separate path from adaptive QIR.

## 9 · Conclusion

Zoom back out to the shared architecture. Common interfaces, compilation, and
orchestration connect different users, classical infrastructure, and devices.

“Built with the MQT and QDMI communities — driven into practice by MQSC.”

Leave the mq.sc QR code visible for discussion.
