# Python binding performance

Status: complete; local implementation and validation complete.

## Scope and contracts

The public benchmark, QDMI, and Qiskit interfaces keep their signatures and
supported inputs. Benchmark objects remain immutable validated inputs;
generating a program loads MLIR lazily. QDMI operation wrappers retain their
session while alive and release it when destroyed. Qiskit import preserves
packed parameter order, identity, and numeric values on the supported 2.5 and
2.6 adapters.

Only primitive instruction parameter reads change. Custom operations, control
flow, validation, and error ordering retain their existing paths. Changing a
cached Qiskit Gate without assigning it back can leave its parameters
inconsistent with the circuit. Import follows packed circuit state, as QPY
serialization does; it does not preserve Qiskit 2.5's incidental reading of a
stale numeric cache.
[Qiskit documents the assignment-back requirement](https://github.com/Qiskit/qiskit/blob/2.5.0/crates/circuit/src/circuit_instruction.rs#L90-L97).

## Decisions and evidence

Baseline: `eb8f7db150836e3c516aca82d3dce92765a2bb24`, rebuilt from its exact
binding sources with the candidate compilation settings and native libraries.
Measurements used CPython 3.14.8, Qiskit 2.6.0, nanobind 3.1.0, and Clang 23
Release builds with ThinLTO and mold on an ARM64 DGX Spark. Eight alternating
baseline/candidate pairs ran in fresh processes on the same CPU. The shared host
was not reserved. Public entry points produced identical IR or metadata
fingerprints in both cohorts. Times below are medians of process CPU time; wall
times closely matched them.

| Change and public workload                                |                 Baseline |                Candidate | Interpretation                                                                 |
| --------------------------------------------------------- | -----------------------: | -----------------------: | ------------------------------------------------------------------------------ |
| Benchmark GHZ(8), QFT(10), QAOA(8, depth=2) generation    | 0.898 / 1.079 / 1.154 ms | 0.883 / 1.064 / 1.132 ms | Small timing change; some paired intervals overlap noise                       |
| QDMI operations, 3 / 54 wrappers                          |            198 / 1535 ns |             165 / 908 ns | Repeatable warmed wrapper-query reduction                                      |
| QDMIBackend construction, SC / DDSIM                      |         0.517 / 0.259 ms |         0.519 / 0.258 ms | No clear end-to-end benefit                                                    |
| Qiskit import, 256 symbolic RX / efficient_su2(8, reps=3) |         2.118 / 1.129 ms |         1.794 / 1.030 ms | 7/8 symbolic RX pairs and 8/8 ansatz pairs improved; numeric control unchanged |

- **Benchmark handoff:** use the existing family registry and typed native
  generators, passing the existing Python receiver with `nb::find`. Validated
  immutable inputs make JSON serialization, reparsing, and an unused manifest
  unnecessary. All 16 families already have generation tests. This removes about
  200 allocation events per measured call without copying benchmark values.
  Separate encoding, parsing/construction, and manifest probes cost roughly 1,
  1.2–1.5, and 4.5–4.7 microseconds. These probes are not additive attribution
  of the complete JSON path. Native generation and destruction dominate; sampled
  native profiles show allocation, MLIR storage uniquing, operation
  registration, and context destruction. Sharing contexts is outside scope.
- **Operation pooling:** use nanobind's bounded pool only for `Operation`. Its
  destructor releases the session before wrapper storage enters the pool. Warm
  54-operation queries drop from 58 allocation events to 4; backend creation
  drops from 3422 to 3314. Target construction dominates the latter (about 98%
  of the profiled constructor), so a query speedup does not imply a backend
  speedup. First operation queries showed no meaningful cold penalty. At the
  default capacity of 128, empty 56-byte wrappers occupy at most 7168 bytes,
  plus pool bookkeeping. Site pooling is excluded: large site lists have a
  different reuse pattern and previously regressed.
- **Qiskit parameters:** read `CircuitInstruction.params` directly. The two
  changed reads remove 512 Gate constructions for 256 symbolic RX instructions
  and 128 for the ansatz on both supported adapters. UUID decoding remains
  unchanged. No cache, invalidation logic, or new retained state is needed.
  Symbolic RX allocation traffic falls from 2.58 MB to 1.68 MB; peak live traced
  memory stays about 0.7 MB. Benchmark retained memory is unchanged. Warm-pool
  allocation traces exclude storage allocated during warmup. Process peak RSS
  varied between runs; no meaningful RSS reduction is claimed.

Cold benchmark generation, including lazy module import, remained about 11–12
ms. Do not interpret these local measurements as isolated-host or CI benchmarks.
The private benchmark bridge is omitted from generated stubs; public
`generate()` annotations remain unchanged.

## Validation

- 1193 existing benchmark, QDMI, Qiskit
  translation/target/parameter/loop/integer, and Qiskit backend tests passed on
  Python 3.14/Qiskit 2.6.
- 43 focused existing compatibility tests passed on Python 3.11.17/Qiskit 2.5.2.
  Four fixed circuit fixtures have identical emitted IR across both builds and
  versions, preserve parameter UUIDs, and match bound statevectors exactly.
- Fresh-process probes cover both benchmark import orders, subclass and
  temporary receivers, and rejection of the wrong receiver type. The existing
  native session fixture verifies immediate session release for three bursts of
  258 Operation wrappers, while a separately retained operation remains usable.
- No new tests duplicate the existing end-to-end contracts. Temporary profiles
  check removed construction/allocation work without making implementation
  details permanent test requirements. One independent Ponytail review found no
  substantiated defects.
- Stub regeneration, optimized rebuild, and the final 1193-test selection
  passed. Repository lint, including Ruff and ty, passed. All-line clang-tidy
  checked all four changed C++ files without diagnostics. The local nox wrapper
  cannot read its feedback file; running its underlying cpp-linter command
  against the staged files completed successfully.

No context sharing, parameter cache, site pooling, API expansion, or general
benchmark infrastructure is part of this change. Hosted CI has not run for this
local investigation.
