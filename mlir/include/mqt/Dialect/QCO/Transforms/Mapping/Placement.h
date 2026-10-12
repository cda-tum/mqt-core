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
/// Supports scalar QCO and statically sized qtensors with constant indices.
/// Barriers add no edges; repeated interactions add one undirected edge.
/// Diagnoses nested control flow, dynamic tensor accesses, undecomposed gates
/// with more than two qubits, and unknown operations carrying qubits. Analyze
/// nested regions separately; their block arguments define local wire roots.
[[nodiscard]] FailureOr<RegionConnectivity>
buildConnectivityGraph(Region& region);

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
/// Search depth is bounded by the pattern's arity (at most four).
[[nodiscard]] std::optional<SmallVector<size_t, 4>>
matchTopologyPattern(const Graph& graph,
                     CompilerTarget::TopologyPatterns::Type type);

/// Cover each graph edge once using available pattern types.
///
/// Prefers four-cycles, then stars, then pairs. Retains shared boundary
/// vertices while removing matched edges from a copy. Isolated vertices need no
/// slice. Fails if the available patterns cannot cover all edges. This does not
/// yet join target occurrences or construct a placement.
[[nodiscard]] FailureOr<SmallVector<TopologyPatternMatch>>
decomposeConnectivityGraph(
    Graph graph, ArrayRef<CompilerTarget::TopologyPatterns::Type> available);

} // namespace qco
} // namespace mlir
