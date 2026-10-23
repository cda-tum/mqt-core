# CPU topology layout search

Status: complete. The implementation is reduced to the integrated placement
path; pattern generation, matching, decomposition and parallel search remain.

## Scope and ownership

`PlacementPass` selects and materializes layouts; `MappingPass` routes them.
Both live in `Mapping.cpp` and share read-only candidate scoring and the routing
engine. Placement exposes `patterns=none|small|extended`; the canonical pipeline
keeps direct search (`none`) as its default. High-level Python options are
unchanged. Existing dynamic mapping callers remain supported.

The mapper's existing interaction collection owns circuit-to-graph conversion.
The unused standalone IR graph builder, analysis wrapper, and unbounded
all-layout enumeration API are omitted. `Mapping/Placement.h` provides bounded
layout search, VF2 matching, and pattern decomposition. Shared quantum-value
helpers introduced only for the removed analysis are also omitted.

## Decisions and limits

- Keep executable identity/path shortcuts before bounded search, then existing
  weighted greedy, identity, and random layout refinement on a miss. Search
  considers connectivity; calibration/native scoring remains in the router.
- Direct search fuses adjacency and injectivity constraints without intermediate
  join tables. Fixed starting ranges and quotas preserve deterministic results
  across thread counts. Retain at most one result per task.
- Both solvers use a 100,000-candidate budget. Pattern tuples can assign several
  vertices, so numeric budgets are not equal work. A miss can mean budget
  exhaustion. Pattern preparation and VF2 decomposition are outside the budget.
- Small patterns are pairs, four-cycles, and stars. Extended patterns add
  four-/six-vertex chains, six-/eight-cycles, and full 2×3/2×4 grids. Match
  larger arity first, then more edges, then type; extra target edges are
  allowed.
- Target copies share lazy canonical pattern caches. Optional groups are
  complete or unavailable after 2,000,000 probes or 65,536 occurrences.
  Unavailable groups fall back to smaller selected patterns; original groups
  remain uncapped.
- Prepare independent pattern types in parallel outside their initializers.
  Nested executor work within `call_once` can re-enter initialization and
  deadlock. Per-type enumeration stays serial; oriented indexes are built per
  search call. Pattern search shares unused quotas through deterministic waves.
- Persist `mqt.placement_order` to retain logical routing tie-breaking across
  separate passes and serialization. Routing validates and consumes it, then
  composes `mqt.layout` provenance. Existing physical provenance must cover the
  target in target site order. Target preparation rejects stale metadata;
  arbitrary optimization pipelines invalidate it unless preservation is
  promised.
- The canonical pipeline routes newly placed inputs and preserves existing
  static native programs, including supported higher-arity gates. Standalone
  routing accepts static inputs. Separate passes can repeat native-cost
  preparation when routing is needed; adjacent interactions skip that
  preparation.

## Evidence for the default

The bounded comparison used 97 fixed workloads: 40 exported Benchpress graphs
and 57 synthetic cases, including longer chains, rectangle perimeters, full grid
blocks, and proven impossible queries. It covered 1/2/4/8 workers, cold and warm
caches, four budgets, seven samples after warmup, and repeated default-budget
runs on an Apple M1 Pro with Apple Clang 21 and LLVM/MLIR 23.1.0 release builds.
Worker counts were not pinned cores. Every returned layout was validated.

At budget 100,000, all methods found all 61 feasible fixtures. Small budgets
favored different methods; the experimental balanced direct scheduler lost two
feasible hits at budget 10,000. Keep the existing deterministic direct
scheduler.

At eight workers, geometric mean successful-query latency was 64.9 µs for
direct, 50.2 µs for small patterns, and 81.9 µs for extended patterns across the
selected synthetic group. Extended patterns improved negative-batch throughput,
while successful batches favored direct search. No method won uniformly. These
pattern measurements retained oriented indexes; the integrated pass rebuilds
them per call. Warm-cache prototypes therefore do not establish integrated
compiler speedups. The harness and raw measurements remain outside the
repository.

## Validation

The reduced sources pass 754 release tests: 144 mapping, 313 compiler, 43 MQT
IR, and 254 QCO utility tests. Two `/dev/full` tests are skipped on macOS.
Repository and whole-file C++ lint pass. The three CLI preset smoke cases
produce byte-identical outputs to the preceding revision. No new performance
claim is made for this reduction.

Tests retain the independent brute-force oracle over all 1,024 labelled
five-vertex queries on three targets, budget/1/2/4/8-worker checks, successful
and impossible searches, pattern limits/shared caches, and empty/all-to-all/idle
inputs. Pass integration tests retain serialization, sparse/spare sites, counted
loops, invalid metadata, static native inputs, and layout-aware semantics. The
ten removed tests exercised only deleted unused APIs.

Relevant test binaries under `build/release/mlir/unittests` are
`Dialect/QCO/Transforms/Mapping/mqt-core-mlir-unittest-mapping`,
`Compiler/mqt-core-mlir-unittests-compiler`,
`Dialect/MQT/IR/mqt-core-mlir-unittest-mqt-ir`, and
`Dialect/QCO/Utils/mqt-core-mlir-unittest-qco-utils`. Also check the three
documented CLI presets and repository lint; see the root agent guide for build
commands.
