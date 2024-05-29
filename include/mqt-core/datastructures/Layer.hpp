#pragma once

#include "Definitions.hpp"
#include "QuantumComputation.hpp"
#include "datastructures/UndirectedGraph.hpp"
#include "operations/OpType.hpp"
#include "operations/Operation.hpp"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace qc {
/**
 * @brief Class to manage the creation of layers when traversing a quantum
 * circuit.
 * @details The class uses the DAG of the circuit to create layers of gates that
 * can be executed at the same time. It can be used to create the front or look
 * ahead layer.
 */
class Layer final {
public:
  class DAGVertex;
  using ExecutableSet = std::unordered_set<std::shared_ptr<DAGVertex>>;
  using InteractionGraph = UndirectedGraph<Qubit, std::shared_ptr<DAGVertex>>;

  class DAGVertex : public std::enable_shared_from_this<DAGVertex> {
  protected:
    // if the executableCounter becomes equal to the executableThreshold the
    // vertex becomes executable
    std::int64_t executableThreshold = 0;
    std::int64_t executableCounter = 0;
    std::vector<std::shared_ptr<DAGVertex>> enabledSuccessors;
    std::vector<std::shared_ptr<DAGVertex>> disabledSuccessors;
    bool executed = false;
    Operation* operation;
    ExecutableSet* executableSet;

    /**
     * @brief Construct a new DAGVertex
     * @details The constructor initializes the vertex with the given operation
     * and the given executable set, which is shared among all vertices.
     *
     * @param op
     * @param es
     */
    DAGVertex(Operation* op, ExecutableSet& es)
        : operation(op), executableSet(&es) {}

  public:
    [[nodiscard]] static auto
    create(Operation* operation,
           ExecutableSet& executableSet) -> std::shared_ptr<DAGVertex> {
      std::shared_ptr<DAGVertex> v(new DAGVertex(operation, executableSet));
      v->updateExecutableSet();
      return v;
    }
    [[nodiscard]] auto isExecutable() const {
      assert(executableCounter <= executableThreshold);
      return (!executed) && executableCounter == executableThreshold;
    }
    [[nodiscard]] auto isExecuted() const { return executed; }
    [[nodiscard]] auto getOperation() const -> const Operation* {
      return operation;
    }

  protected:
    auto incExecutableCounter() {
      executableCounter++;
      updateExecutableSet();
    }
    auto decExecutableCounter() {
      executableCounter--;
      updateExecutableSet();
    }
    /**
     * @brief Inserts or removes the vertex from the executable set.
     * @warning May not be called from the constructor.
     */
    auto updateExecutableSet() -> void {
      if (isExecutable()) {
        if (const auto& it = executableSet->find(shared_from_this());
            it == executableSet->end()) {
          executableSet->insert(shared_from_this());
        }
      } else {
        if (const auto& it = executableSet->find(shared_from_this());
            it != executableSet->end()) {
          executableSet->erase(it);
        }
      }
    }

  public:
    auto execute() -> void {
      if (isExecutable()) {
        executed = true;
        for (const auto& successor : disabledSuccessors) {
          successor->decExecutableCounter();
        }
        for (const auto& successor : enabledSuccessors) {
          successor->incExecutableCounter();
        }
        updateExecutableSet();
      } else {
        throw std::logic_error(
            "The vertex is not executable and cannot be executed.");
      }
    }
    auto addEnabledSuccessor(const std::shared_ptr<DAGVertex>& successor) {
      enabledSuccessors.emplace_back(successor);
      ++successor->executableThreshold;
      successor->updateExecutableSet();
    }
    auto addDisabledSuccessor(const std::shared_ptr<DAGVertex>& successor) {
      disabledSuccessors.emplace_back(successor);
      --successor->executableThreshold;
      successor->updateExecutableSet();
    }
  };

protected:
  ExecutableSet executableSet;
  auto constructDAG(const QuantumComputation& qc) -> void;

public:
  Layer() = default;
  Layer(const Layer&) = default;
  Layer(Layer&&) = default;
  Layer& operator=(const Layer&) = default;
  Layer& operator=(Layer&&) = default;
  ~Layer() = default;
  explicit Layer(const QuantumComputation& qc) { constructDAG(qc); }
  [[nodiscard]] auto getExecutableSet() const -> const ExecutableSet& {
    return executableSet;
  }
  [[nodiscard]] auto
  constructInteractionGraph(OpType opType,
                            std::size_t nControls) const -> InteractionGraph;
  [[nodiscard]] auto getExecutablesOfType(OpType opType, std::size_t nControls)
      const -> std::vector<std::shared_ptr<DAGVertex>>;
};
} // namespace qc
