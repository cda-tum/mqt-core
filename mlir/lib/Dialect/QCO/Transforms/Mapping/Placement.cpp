/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Dialect/QCO/Transforms/Mapping/Placement.h"

#include "mqt/Dialect/QCO/IR/QCODialect.h"
#include "mqt/Dialect/QCO/IR/QCOInterfaces.h"
#include "mqt/Dialect/QCO/IR/QCOOps.h"
#include "mqt/Dialect/QCO/QCOUtils.h"
#include "mqt/Dialect/QTensor/IR/QTensorOps.h"

#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/ErrorHandling.h"

#include <algorithm>
#include <array>
#include <limits>
#include <utility>

namespace mlir::qco {

FailureOr<RegionConnectivity> buildConnectivityGraph(Region& region) {
  if (!llvm::hasSingleElement(region)) {
    return emitError(region.getLoc(),
                     "connectivity requires a single-block region");
  }
  RegionConnectivity result;
  // collectWireOrigins does not track identities within tensors.
  DenseMap<Value, SmallVector<size_t>> wires;
  llvm::SetVector<std::pair<size_t, size_t>> edges;
  constexpr auto emptySlot = std::numeric_limits<size_t>::max();

  const auto addRoot = [&](Value value) -> LogicalResult {
    const bool scalar = isa<QubitType>(value.getType());
    auto tensor = dyn_cast<RankedTensorType>(value.getType());
    if (!scalar && (!tensor || !tensor.hasStaticShape())) {
      return emitError(value.getLoc(),
                       "connectivity requires statically sized qubit tensors");
    }
    const int64_t size = scalar ? 1 : tensor.getNumElements();
    auto& indices = wires[value];
    for (int64_t i = 0; i < size; ++i) {
      indices.push_back(result.qubits.size());
      result.qubits.push_back(
          {.root = value, .slot = scalar ? std::nullopt : std::optional(i)});
    }
    return success();
  };
  const auto consume = [&](Value value) {
    auto indices = std::move(wires.at(value));
    wires.erase(value);
    return indices;
  };
  for (Value argument : region.getArguments()) {
    if (isLinearQubitType(argument.getType()) && failed(addRoot(argument))) {
      return failure();
    }
  }
  for (Operation& op : region.front()) {
    if (isa<CallOp>(op)) {
      return op.emitError("inline quantum calls before connectivity analysis");
    }
    for (Value operand : op.getOperands()) {
      if (!isLinearQubitType(operand.getType()) || wires.contains(operand)) {
        continue;
      }
      if (operand.getParentRegion() == &region || failed(addRoot(operand))) {
        return op.emitError("cannot identify an input qubit for connectivity");
      }
    }
    if (auto unitary = dyn_cast<UnitaryOpInterface>(op)) {
      if (!isa<BarrierOp>(op)) {
        if (unitary.getNumQubits() > 2) {
          return op.emitError("decompose to one- and two-qubit gates before "
                              "connectivity analysis");
        }
        if (unitary.isTwoQubit()) {
          const auto a = wires.at(unitary.getInputQubit(0)).front();
          const auto b = wires.at(unitary.getInputQubit(1)).front();
          edges.insert(std::minmax(a, b));
        }
      }
      for (Value input : unitary.getInputQubits()) {
        wires[unitary.getOutputForInput(input)] = consume(input);
      }
    } else if (isa<AllocOp, StaticOp, qtensor::AllocOp>(op)) {
      if (failed(addRoot(op.getResult(0)))) {
        return failure();
      }
    } else if (auto measure = dyn_cast<MeasureOp>(op)) {
      wires[measure.getQubitOut()] = consume(measure.getQubitIn());
    } else if (auto reset = dyn_cast<ResetOp>(op)) {
      wires[reset.getQubitOut()] = consume(reset.getQubitIn());
    } else if (auto tensor = dyn_cast<qtensor::FromElementsOp>(op)) {
      SmallVector<size_t> indices;
      for (Value qubit : tensor.getElements()) {
        indices.push_back(consume(qubit).front());
      }
      wires[tensor.getResult()] = std::move(indices);
    } else if (isa<qtensor::ExtractOp, qtensor::InsertOp>(op)) {
      auto extract = dyn_cast<qtensor::ExtractOp>(op);
      auto insert = dyn_cast<qtensor::InsertOp>(op);
      Value input = extract ? extract.getTensor() : insert.getDest();
      const auto index =
          getConstantIntValue(extract ? extract.getIndex() : insert.getIndex());
      auto indices = consume(input);
      if (!index || *index < 0 ||
          std::cmp_greater_equal(*index, indices.size())) {
        return op.emitError(
            "connectivity requires an in-bounds constant tensor index");
      }
      auto& slot = indices[static_cast<size_t>(*index)];
      if (extract) {
        if (slot == emptySlot) {
          return op.emitError("cannot extract an empty slot for connectivity");
        }
        wires[extract.getResult()] = {std::exchange(slot, emptySlot)};
        wires[extract.getOutTensor()] = std::move(indices);
      } else {
        if (slot != emptySlot) {
          return op.emitError(
              "cannot insert into an occupied slot for connectivity");
        }
        slot = consume(insert.getScalar()).front();
        wires[insert.getResult()] = std::move(indices);
      }
    } else if (isa<SinkOp, qtensor::DeallocOp>(op) ||
               op.hasTrait<OpTrait::IsTerminator>()) {
      for (Value operand : op.getOperands()) {
        wires.erase(operand);
      }
    } else if (op.getNumRegions() != 0 ||
               llvm::any_of(op.getOperandTypes(), isLinearQubitType) ||
               llvm::any_of(op.getResultTypes(), isLinearQubitType)) {
      return op.emitError("unsupported connectivity operation; analyze nested "
                          "regions separately and inline quantum calls");
    }
  }
  result.graph =
      Graph(llvm::to_vector(llvm::seq<size_t>(0, result.qubits.size())));
  for (const auto& [a, b] : edges) {
    result.graph.addEdge(a, b);
    result.graph.addEdge(b, a);
  }
  return result;
}

namespace {
using Pattern = CompilerTarget::TopologyPatterns::Type;
using Edge = std::pair<size_t, size_t>;
} // namespace

static ArrayRef<Edge> patternEdges(Pattern type) {
  static constexpr std::array<Edge, 4> CYCLE{{{0, 1}, {1, 2}, {2, 3}, {3, 0}}};
  static constexpr std::array<Edge, 3> STAR{{{0, 1}, {0, 2}, {0, 3}}};
  static constexpr std::array<Edge, 1> PAIR{{{0, 1}}};
  switch (type) {
  case Pattern::FourCycle:
    return CYCLE;
  case Pattern::Star:
    return STAR;
  case Pattern::QubitPair:
    return PAIR;
  }
  llvm_unreachable("unknown topology pattern");
}

std::optional<SmallVector<size_t, 4>> matchTopologyPattern(const Graph& graph,
                                                           Pattern type) {
  const size_t arity = type == Pattern::QubitPair ? 2 : 4;
  auto nodes = graph.getNodes();
  if (nodes.size() < arity) {
    return std::nullopt;
  }
  std::array<SmallVector<size_t, 3>, 4> neighbours;
  for (const auto& [a, b] : patternEdges(type)) {
    neighbours[a].push_back(b);
    neighbours[b].push_back(a);
  }
  SmallVector<size_t, 4> match;
  const auto adjacent = [&](size_t a, size_t b) {
    return llvm::is_contained(graph.getNeighbours(a), b);
  };
  // All templates have a connected prefix, so extend along its frontier.
  const auto search = [&](auto&& self) -> bool {
    const auto depth = match.size();
    if (depth == arity) {
      return true;
    }
    ArrayRef<size_t> candidateRange = nodes;
    for (const auto previous : neighbours[depth]) {
      if (previous < depth) {
        candidateRange = graph.getNeighbours(match[previous]);
        break;
      }
    }
    auto candidates = llvm::to_vector(candidateRange);
    llvm::sort(candidates);
    for (const auto candidate : candidates) {
      if (llvm::is_contained(match, candidate) ||
          graph.getDegree(candidate) < neighbours[depth].size()) {
        continue;
      }
      if (llvm::any_of(neighbours[depth], [&](size_t other) {
            return other < depth && !adjacent(candidate, match[other]);
          })) {
        continue;
      }
      size_t patternFrontier = 0;
      size_t patternRemaining = 0;
      for (const auto next : neighbours[depth]) {
        if (next > depth) {
          ++patternRemaining;
          patternFrontier += llvm::any_of(neighbours[next],
                                          [&](size_t n) { return n < depth; });
        }
      }
      size_t graphFrontier = 0;
      size_t graphRemaining = 0;
      for (const auto next : graph.getNeighbours(candidate)) {
        if (!llvm::is_contained(match, next)) {
          ++graphRemaining;
          graphFrontier +=
              llvm::any_of(match, [&](size_t n) { return adjacent(next, n); });
        }
      }
      // Non-induced matches may map new pattern vertices into the graph
      // frontier.
      if (patternFrontier > graphFrontier ||
          patternRemaining > graphRemaining) {
        continue;
      }
      match.push_back(candidate);
      if (self(self)) {
        return true;
      }
      match.pop_back();
    }
    return false;
  };
  if (search(search)) {
    return match;
  }
  return std::nullopt;
}

FailureOr<SmallVector<TopologyPatternMatch>>
decomposeConnectivityGraph(Graph graph, ArrayRef<Pattern> available) {
  SmallVector<TopologyPatternMatch> slices;
  // Removing edges cannot create a new non-induced match of an exhausted type.
  for (const auto type :
       {Pattern::FourCycle, Pattern::Star, Pattern::QubitPair}) {
    if (!llvm::is_contained(available, type)) {
      continue;
    }
    while (auto match = matchTopologyPattern(graph, type)) {
      for (const auto& [a, b] : patternEdges(type)) {
        graph.removeEdge((*match)[a], (*match)[b]);
        graph.removeEdge((*match)[b], (*match)[a]);
      }
      slices.push_back({.type = type, .vertices = std::move(*match)});
    }
  }
  if (llvm::any_of(graph.getNodes(),
                   [&](size_t n) { return graph.getDegree(n) != 0; })) {
    return failure();
  }
  return slices;
}

} // namespace mlir::qco
