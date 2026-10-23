# CPU topology layout search

Status: complete locally. Placement and routing are separate passes in the same
source file. Pattern search is opt-in; direct search remains the default.
Runtime validation, repository lint and whole-file C++ lint passed.

## Outcome and scope

`enumerateTopologyLayouts` in `Mapping/Placement.h` enumerates every injective,
non-induced layout of a circuit graph onto a target. Rows follow ascending graph
vertex IDs and contain dense target vertices. Disconnected components, idle
qubits, empty graphs and implicit all-to-all targets are supported.
`findTopologyLayout` reuses this search with a candidate-assignment budget and
retains at most one result per task. The placement pass tries it after existing
identity and path shortcuts and before the weighted greedy fallback. Calibration
scoring is outside this search.

## Decisions

- Keep the cheap identity and path shortcuts. Bound the general embedding search
  to 100,000 candidate assignments in total; fixed starting ranges and budgets
  preserve thread-count determinism. Do not enumerate all layouts in the
  compiler. Use the mapper's existing weighted interactions to preserve its wire
  ordering and supported control-flow subset. Found layouts still pass through
  routing, native synthesis and final target validation.
- Fuse edge joins and injectivity checks in a vertex-at-a-time search. Start
  with the highest-degree query vertex, then prefer vertices with more assigned
  neighbors. Extend from the shortest assigned-neighbor adjacency list and check
  remaining edges and target degree. This preserves the matching constraints of
  [Algorithm 1](https://arxiv.org/abs/2508.21287) without intermediate join
  tables.
- Use explicit search frames, a worker-local layout and used-vertex bitset, and
  the MLIR context's thread pool. Concatenate contiguous starting ranges in
  order so thread count does not change results. Small searches stay serial.
  All-to-all targets need neither adjacency lists nor pattern occurrences.
- Retain direct vertex search as the compiler default. The extended-pattern
  comparison covers successful and unsuccessful searches, cold and warm caches,
  and one, two, four and eight workers. Warm motif indexes help selected
  negative batches and chains, but successful placement remains faster with
  direct search across the measured groups. Do not promote the experimental wave
  scheduler: it helps some workloads but loses two feasible hits at budget
  10,000 and has no uniform latency advantage. The balanced-direct prototype
  remains a benchmark artifact; the motif solver is available through an
  explicit placement option.
- Expose optional four-/six-vertex chains, six-/eight-cycles and full
  two-by-three and two-by-four grids through individually lazy shared caches.
  Each group is complete or unavailable after 2,000,000 probes or 65,536
  occurrences. Existing pattern groups keep their behavior. The decomposition
  pointer overload reuses these groups without copying their occurrence tables.
- Keep iterative IR connectivity traversal. An actual recursive prototype
  matched all 14 existing fixtures. Eleven alternating timing samples found only
  about 3% differences, with neither implementation consistently faster; an
  explicit stack also avoids dependence on IR nesting depth. Pattern matching
  recursion remains bounded by template arity.
- Keep the one- and two-qubit gate precondition. Treating an undecomposed larger
  gate as a clique can require couplings its decomposition does not need; the
  existing compiler pipeline lowers these gates before mapping.
- Sort decomposition patterns by arity, edge count and type, replacing the fixed
  type list. Extract matching helpers as file-local functions, using `static` as
  required by the repository's LLVM lint policy.
- Keep enumeration within each target-pattern type serial. A shared-executor
  prototype deadlocked when work stealing re-entered the same `call_once`. The
  motif solver prepares independent types in parallel outside their
  initialization, then builds immutable indexes for that call. Its resumable DFS
  uses a 256-probe serial prefix and up to 64 root ranges, with per-task wave
  quotas growing from 1,024 to 4,096 and 16,384. Unused budgets return to the
  next wave; deterministic result selection avoids racy cancellation.
- Separate placement and routing in `Mapping.cpp`. Placement owns selection and
  materialization; candidate scoring reuses the router without mutating IR. The
  canonical explicit-target pipeline runs both passes. The mapping pass retains
  dynamic-input compatibility for existing callers.
- Expose `patterns=none|small|extended` only on PlacementPass. `none` selects
  the existing direct solver. The other presets use VF2 decomposition and
  bounded tuple joins; a miss continues with the existing layout candidates. The
  public Python options remain unchanged.
- Persist internal logical order as `mqt.placement_order` so separate passes and
  serialized placed IR retain routing tie-breaking. This is distinct from
  `mqt.layout` source provenance. Routing validates and consumes the internal
  permutation; existing physical provenance must use full target width and
  target-ordered sites before permutations can be composed. Target preparation
  rejects stale placement metadata; arbitrary optimization pipelines discard it
  with source layout metadata unless the caller promises preservation.
- Keep canonical patterns in the shared target cache. The integrated motif
  helper rebuilds oriented indexes per call; prototype measurements with fully
  retained indexes do not describe the integrated pass. Separate passes also
  repeat native-cost preparation when final routing needs it. Adjacent placed
  interactions skip that preparation.

## Validation

- Mapping unit tests: 153 passed. Direct, small-pattern and extended-pattern
  searches match a brute-force oracle for all 1,024 labelled five-vertex query
  graphs on three targets. Tests also cover budget boundaries, 1/2/4/8 workers,
  unavailable larger patterns, empty/all-to-all inputs and idle vertices.
- Separate-pass tests cover serialized IR, noncontiguous site IDs, spare sites,
  nested counted loops, rejected options and malformed metadata. Sampling and
  basis-state checks preserve semantics; default split routing matches the
  legacy combined path in the covered cases.
- Compiler unit tests: 300 passed; two device-write failure tests require
  `/dev/full` and were skipped on macOS. Existing contracts for preplaced native
  programs remain intact. The one-way-entangler test checks target compilation,
  ordered native support and layout-aware full-unitary equivalence.
- MQT IR tests: 43 passed, including placement-order serialization, attribute
  ownership and permutation validation. Generic compilation-option tests check
  metadata invalidation and explicit preservation.
- QCO utility tests: 251 passed. Across these four suites, 747 tests passed and
  two platform-dependent tests were skipped.
- The documented CLI pipeline succeeds with all three pattern presets.
- Seven synthetic end-to-end cases have identical gate and SWAP counts before
  and after the pass split. All pass IR, linearity, topology and native-support
  checks; both six-qubit cases pass full-matrix equivalence. These single-call
  checks are regression checks, not new performance measurements.
- Whole-file C++ lint (`uvx nox -s cpp-lint`) passed with LLVM 23 for all 16
  changed C++ translation units in the PR, including optional patterns and pass
  separation.
- Repository lint (`uvx nox -s lint`) and final diff checks passed.

## Performance evidence and limits

Target-compilation benchmarks used seven synthetic relabelled graph circuits
through `QCOProgram::compileForTarget`, including cleanup, mapping and native
synthesis. The baseline is the pipeline before integration. Default mapping
options on this machine select ten trials and ten threads; seed 42 was fixed. On
an Apple M1 Pro, Apple Clang 21 and LLVM/MLIR 23.1.0 release build, five
alternating batches each ran one warm-up and three timed compilations. These are
medians of the batch medians. Parsing, input copying, target construction and
thread-pool startup were excluded.

| Circuit and target                |   Before |    After | Output change        |
| --------------------------------- | -------: | -------: | -------------------- |
| Native 4×4 grid in 8×8 grid       | 7.416 ms | 4.081 ms | 100 → 48 CX gates    |
| Unrestricted 4×4 grid in 8×8 grid | 5.400 ms | 3.523 ms | 17 → 0 routing SWAPs |
| Impossible five-cycle in 8×8 grid | 1.840 ms | 1.946 ms | Same two SWAPs       |

Every result passed verification, linearity and topology/native-support checks.
The two six-qubit cases also passed layout-aware full-matrix comparison. These
are synthetic workloads, not Benchpress, and do not establish a universal
speedup or fidelity improvement. In the smaller native 2×3-in-3×3 case, CX count
stayed at 14 but total gates increased from 33 to 47 due to direction repairs.
The shortcut optimizes routing, not native gate count or calibration cost.

### Layout enumeration

Synthetic graph microbenchmarks used an Apple M1 Pro, Apple Clang 21, LLVM/MLIR
23.1.0 and a release build. Medians of three calls followed a warm-up; query
planning, search and complete output materialization were included. Target
construction and thread-pool startup were excluded. Complete result sets matched
an independent materialized hash-join reference for nine workloads at each
thread setting. The reference uses degree pruning and shared-boundary join
ordering; it is a straightforward C++ implementation, not a tuned dataframe
engine.

| Query and target                      | Hash-join reference | Serial search | Eight-thread search |
| ------------------------------------- | ------------------: | ------------: | ------------------: |
| 4×4 grid in a 22×22 grid              |           13.724 ms |      0.919 ms |            0.390 ms |
| Impossible five-cycle in a 22×22 grid |            7.698 ms |      0.493 ms |            0.188 ms |
| Nine-vertex path in a 12×12 grid      |           47.632 ms |     23.468 ms |           16.508 ms |

Output-heavy disconnected matching showed essentially no advantage over the
reference. Small inputs use serial execution even with an eight-thread context.
The all-results API retains every layout, so time and output space can be
exponential. The compiler uses the bounded API instead. These enumeration
measurements do not establish general superiority over specialized graph
solvers.

### Bounded first-layout comparison

The direct production API was compared with a first-layout motif prototype that
uses the existing VF2 matcher, query decomposition and target patterns. The
prototype keeps oriented tuples and per-column postings, without intermediate
join tables. Measurements include query planning and search. Three motif modes
include cold preparation, warm canonical patterns with rebuilt indexes, and
fully retained indexes. Target construction, pool startup and result checks are
outside timers. The environment is the same release toolchain and Apple M1 Pro
as above.

The fixed corpus contains 40 exported pre-placement Benchpress graphs (ten
circuits on four topologies) and 33 synthetic cases, including proven impossible
cycles and nonplanar graphs. Four budgets from 1,000 through 1,000,000 and one-
and eight-thread contexts give 2,336 configurations, each with a warmup and
seven samples. Five independent repetitions of the default-budget matrix
confirmed identical outcomes and the main latency differences. A brute-force
oracle checked 3,072 small query/target pairs; mixed-pattern, budget and thread
tests also passed. Every measured layout passed an independent validity check.

At budget 100,000, both solvers found all 45 feasible fixtures and returned no
layout for the 28 independently proven impossible fixtures. Excluding all-to-all
controls, cutoff diagnostics and trivial boundaries, direct search was about 4.1
times faster on eight successful real cases and 2.2 times faster on 22
successful synthetic cases than the fully warm motif implementation (serial
geometric means). The synthetic advantage was about 1.8 times with eight
threads. All real explicit targets have at most 57 sites and remain serial.

Six of those eight real successes are GHZ/Ising path graphs already handled by
the existing path shortcut on these targets. For the remaining DNN16 cases,
serial median times in microseconds were:

| Target    | Direct | Motif cold | Motif indexes warm |
| --------- | -----: | ---------: | -----------------: |
| Square    |    2.5 |       16.0 |               14.3 |
| Heavy-hex |    2.0 |       13.2 |               11.8 |

Warm indexes help some negative cases: the impossible 17-cycle in an 8×8 grid
fell from about 738 to 591 microseconds serially. The 22 unsuccessful real cases
were about 2.3 times faster by geometric mean, but summed medians decreased only
from about 44 to 36 microseconds; most were cheap degree rejections. On the four
nontrivial synthetic negatives, neither method consistently won with warm
indexes. At budget 1,000, motif search found 8/8 real and 13/22 synthetic
feasible cases, versus 6/8 and 10/22 for direct search. One tuple probe can bind
several vertices, so equal numeric budgets are not equal work. The two extra
real hits at budget 1,000 are covered by the existing path shortcut. All hits
coincide at the pipeline budget. Faster degree rejections also benefit from an
earlier check before query planning, independently of motif indexing.

Warm results assume persistent indexes across compilations. Owned index
capacities ranged from about 5 KiB on the DNN heavy-hex target to 1.14 MiB on
the 22×22 grid, excluding canonical patterns and allocator overhead. These are
kernel measurements, not new end-to-end compilation measurements. The exported
graphs can have a different wire order from the mapper; the existing path
shortcut's success here does not depend on that order. Fixed 32-way budget
partitions still waste unused quotas and can miss feasible layouts at small
budgets; this comparison leaves that policy unchanged.

### Extended patterns and parallel bounded search

The extended comparison adds 24 fixed fixtures: longer chains, rectangle
perimeters, full grid blocks, planted subgraphs and independently impossible
queries. All targets in this extension are connected, relabelled graphs with
64–400 sites. The complete corpus has 97 cases: 40 exported Benchpress graphs
and 57 synthetic cases, comprising 61 feasible and 36 proven impossible cases.
The existing VF2 matcher and decomposition are used by both motif variants.

Six methods compare the unchanged production direct search, a balanced direct
prototype, and small/extended motif searches with cold/warm indexes. Both new
solvers share the same resumable wave scheduler. The sweep measures budgets
1,000/10,000/100,000/1,000,000 and 1/2/4/8 workers: 9,312 single-query rows and
192 warm-batch rows, each with one warmup and seven samples. Three independent
repeats cover the default-budget matrix and batches. Batches have 48 calls and
share one target cache; repeated queries are explicit. They parallelize whole
queries with serial inner searches, separately from single-query parallelism.

An independent brute-force oracle validates 3,072 query/target pairs. Budget
boundaries, mixed motifs, shared-cache queries and thread-count determinism also
pass. Every timed layout is validated outside its timer. At the pipeline budget
100,000, every method finds all 61 feasible fixtures. At budget 1,000, hits are
32/44/42/38 for original direct/balanced direct/small motifs/extended motifs; at
budget 10,000 they are 60/58/56/54. Numeric probe counts are not equal work: a
tuple probe can bind several vertices.

At budget 100,000, medians of four independent run aggregates give the following
geometric means at eight workers. Single-query groups contain 38 successful and
12 impossible synthetic cases; batches contain five positive and two negative
groups. Complete tables and per-case results remain in the artifact report.

| Method                | Successful query | Negative query | Positive batch | Negative batch |
| --------------------- | ---------------: | -------------: | -------------: | -------------: |
| Current direct        |          64.9 µs |       131.6 µs | 35,359 calls/s | 14,249 calls/s |
| Balanced direct       |          25.6 µs |       212.5 µs | 43,933 calls/s | 12,269 calls/s |
| Small motifs, warm    |          50.2 µs |       256.3 µs | 24,862 calls/s | 14,847 calls/s |
| Extended motifs, warm |          81.9 µs |       216.8 µs | 23,035 calls/s | 24,606 calls/s |

Across only the 16 new successful fixtures, extended motifs are about 8% faster
than the original direct search at eight workers, but 3.6 times slower than
balanced direct and 1.7 times slower than small motifs. Their negative-batch
throughput advantage is real; successful batches favor direct search. Both
approaches scale roughly five to six times across independent queries. This does
not imply the same scaling within one search.

Fully warm extended indexes retain 1.94/6.28/23.12 MB on 8×8/12×12/20×20 grids,
versus 0.108/0.317/1.10 MB for small motifs. Serial preparation takes roughly
1.46/4.05/13.99 ms. These costs support retaining the direct pipeline default
while preserving the larger patterns and reproducible prototypes for targeted
workloads. The later pass integration exposes motif search as an opt-in; it does
not change the default based on these measurements.

These are release-build graph-search measurements on an Apple M1 Pro with Apple
Clang 21 and LLVM/MLIR 23.1.0. Worker counts are not affinity-pinned cores. Warm
motif caches retain degrees and indexes; the direct API rebuilds adjacency per
call. Preparation, retained capacities and cold calls are reported separately.
Retained index capacity excludes canonical groups and allocator overhead. No
optional cache group hits its cap in this corpus. These results do not establish
an end-to-end compiler or GPU speedup.

The benchmark harness and raw results are kept outside the repository. Its
`build.py` builds against the release libraries; `bounded_benchmark --verify`
runs the oracle and concurrency checks; `run.py` reproduces the matrix. The
handoff report links the full methodology, per-case samples, repeated comparison
tables and source hashes.
