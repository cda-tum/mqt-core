/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include "mqt/Compiler/Target.h"
#include "mqt/Dialect/QCO/Utils/Graph.h"

#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/LogicalResult.h"

#include <cstddef>
#include <cstdint>
#include <optional>

namespace mlir {
class Operation;
class Region;

namespace qco {

/// Simple undirected interactions, represented by two directed edges.
struct RegionConnectivity {
  struct Qubit {
    Value root;
    /// Tensor slot, or nullopt for a scalar qubit.
    std::optional<int64_t> slot;
  };
  Graph graph{ArrayRef<size_t>{}};
  /// Dense vertex order: block arguments, then roots in program order.
  ///
  /// Captured values become roots on their first use. Includes idle qubits.
  SmallVector<Qubit> qubits;
};

/// Build a connectivity graph without changing IR.
///
/// Requires a single-block region that passes verification and verifyLinearity.
/// Root values remain valid only while their IR exists.
/// Follows RegionBranchOpInterface edges, visiting each reachable single-block
/// region once, without evaluating conditions or unrolling loops.
/// Supports scalar QCO and statically sized qtensors with constant indices.
/// Incoming paths at joins and loop backedges must agree on wire identities,
/// including individual tensor slots.
/// Barriers add no edges; repeated interactions add one undirected edge.
/// Diagnoses dynamic tensor accesses, quantum calls, undecomposed gates with
/// more than two qubits, and unsupported operations carrying qubits or regions.
[[nodiscard]] FailureOr<RegionConnectivity>
buildConnectivityGraph(Region& region);

/// Cached connectivity of an operation with one region, such as func.func.
///
/// Uses buildConnectivityGraph's input contract. MLIR invalidates this analysis
/// unless a pass explicitly preserves it, including the stored root values.
class RegionConnectivityAnalysis {
public:
  explicit RegionConnectivityAnalysis(Operation* operation);

  /// Return the graph, or failure if the region is unsupported.
  [[nodiscard]] const FailureOr<RegionConnectivity>& getConnectivity() const {
    return connectivity_;
  }

private:
  FailureOr<RegionConnectivity> connectivity_ = failure();
};

struct TopologyPatternMatch {
  CompilerTarget::TopologyPatterns::Type type;
  /// Graph vertices in the target pattern's template order.
  ///
  /// Equal vertex IDs across slices constrain a later layout search.
  SmallVector<size_t, 4> vertices;
};

/// Find one non-induced match using VF2 frontier and feasibility checks.
///
/// The graph must be simple, undirected (symmetric), and loop-free.
/// Returns vertices in template order, independent of graph insertion order.
/// Search depth is bounded by the pattern's arity.
[[nodiscard]] std::optional<SmallVector<size_t, 4>>
matchTopologyPattern(const Graph& graph,
                     const CompilerTarget::TopologyPatterns::Group& pattern);

/// Cover each graph edge once using available pattern types.
///
/// Prefers larger arity, then more edges, then the smaller pattern type.
/// Retains shared boundary vertices while removing matched edges from a copy.
/// Isolated vertices need no slice. Ignores groups without target occurrences.
/// Fails if the available patterns cannot cover all edges. This does not join
/// target occurrences or construct a placement.
[[nodiscard]] FailureOr<SmallVector<TopologyPatternMatch>>
decomposeConnectivityGraph(
    Graph graph, ArrayRef<CompilerTarget::TopologyPatterns::Group> available);

/// Cover graph edges using independently cached pattern groups.
///
/// Uses the same ordering and coverage contract as the contiguous-group
/// overload. Null pointers are ignored, allowing unavailable optional groups.
[[nodiscard]] FailureOr<SmallVector<TopologyPatternMatch>>
decomposeConnectivityGraph(
    Graph graph,
    ArrayRef<const CompilerTarget::TopologyPatterns::Group*> available);

/// Enumerate all injective, non-induced layouts of graph onto target.
///
/// The graph must be simple, undirected (symmetric), and loop-free. Each row
/// maps graph vertices in ascending ID order to dense target vertices, not site
/// IDs. Includes disconnected components and isolated vertices. An empty graph
/// has one empty layout; a graph that cannot fit has none. Native operation
/// availability and calibration are not considered.
///
/// Fuses edge joins and injectivity filtering in a vertex-at-a-time search,
/// without materializing pattern occurrences or intermediate joins. Independent
/// search branches use the context's thread pool; disabling context threading
/// selects serial execution. Small searches also run serially.
/// Output order is deterministic across thread counts. All results are
/// retained: enumeration can take exponential time and output space. This
/// helper does not change IR or select a layout for the compiler pipeline.
[[nodiscard]] SmallVector<SmallVector<size_t>, 0>
enumerateTopologyLayouts(const Graph& graph, const CompilerTarget& target,
                         MLIRContext& context);

/// Find one topology layout within a bounded number of candidate assignments.
///
/// Uses enumerateTopologyLayouts's graph and layout contract. The search
/// retains at most one layout per worker task and divides maxCandidates across
/// fixed starting ranges, independent of thread count. Returns nullopt when no
/// layout is found, including budget exhaustion; this does not prove
/// infeasibility.
[[nodiscard]] std::optional<SmallVector<size_t>>
findTopologyLayout(const Graph& graph, const CompilerTarget& target,
                   MLIRContext& context, size_t maxCandidates);

/// Find a bounded layout using selected topology patterns and the VF2 cover.
///
/// Uses findTopologyLayout's graph and layout contract. Canonical occurrences
/// share the target's lazy caches; oriented tuple indexes are built per call.
/// Unavailable groups are ignored. Returns nullopt if the selected groups do
/// not cover all edges or the search exhausts maxCandidates. An empty selection
/// supports only edgeless queries or all-to-all targets. Include QubitPair to
/// cover edges that do not fit larger patterns.
///
/// A candidate is one pattern tuple or isolated-vertex assignment. Preparation
/// and decomposition are outside this budget. Fixed search ranges share unused
/// quota in deterministic parallel waves; output is independent of thread
/// count.
[[nodiscard]] std::optional<SmallVector<size_t>> findTopologyPatternLayout(
    const Graph& graph, const CompilerTarget& target, MLIRContext& context,
    size_t maxCandidates,
    ArrayRef<CompilerTarget::TopologyPatterns::Type> selected);

} // namespace qco
} // namespace mlir
