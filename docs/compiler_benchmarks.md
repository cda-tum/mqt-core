---
hide-toc: true
---

# Compiler benchmarks

The compiler dashboard compares compilation time, native two-qubit gate count,
and circuit depth for GHZ, QFT, and Bernstein–Vazirani programs from
{py:mod}`mqt.core.bench`. Select a circuit, connectivity, revision, and baseline
to inspect the measurements. The asv explorer and JSON download provide the
underlying results and timing samples.

```{raw} html
:file: _build/compiler-benchmark-fragment.html
```

Each measurement group identifies the machine, compiler revisions, and sample
counts. The suite uses 4, 8, and 16 qubits on U/CX all-to-all and line targets.
Measurements do not necessarily describe the documentation version you are
reading. The initial checked-in snapshot contains two revisions measured on a
development machine, with six timing samples per case.

Compilation time includes copying the input, QC/QCO conversion, target
compilation, conversion back to native QC, and result disposal. Generation and
quality analysis are excluded. Gate depth assigns one layer to U/CX gates;
terminal measurements add no depth and barriers synchronize wires.

The dedicated Linux runner measures missing main commits from **4.0.0 onward**,
with fixed CPU affinity and 20 timing samples per case. It repeatedly measures
the 4.0.0 reference alongside candidates to expose machine drift. Compatible
observations share a history series; changes to the machine, toolchain, or
benchmark suite start a separate series. Failed commits remain visible and are
retried. GitHub-hosted runners are not used for measurement.

Each observation retains raw samples and machine configuration. Changes are
informational; a timing change alone is not proof of a regression. Different
generator sources suppress deltas because they can change compiler inputs. The
initial development snapshot predates 4.0.0 and is separate from the DGX
history.

Read the Docs publishes saved results; it does not run the benchmarks. Its
`latest` build imports the published measurement history, while release versions
retain their saved snapshot. Before the first publication, `latest` also uses
the checked-in snapshot.

See the repository's
[benchmark instructions](https://github.com/munich-quantum-toolkit/core/blob/main/benchmarks/README.md)
for local runs and updating the documentation snapshot.
