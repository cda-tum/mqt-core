#pragma once

#include "Definitions.hpp"
#include "dd/DDDefinitions.hpp"

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

namespace dd {
template <class Edge, std::size_t numberOfStochasticOperations = 64>
class StochasticNoiseOperationTable {
public:
  explicit StochasticNoiseOperationTable(const std::size_t nv) : nvars(nv) {
    resize(nv);
  };

  // access functions
  [[nodiscard]] const auto& getTable() const { return table; }

  void resize(std::size_t nq) {
    nvars = nq;
    table.resize(nvars);
  }

  void insert(std::uint8_t kind, qc::Qubit target, const Edge& r) {
    assert(kind <
           numberOfStochasticOperations); // There are new operations in OpType.
                                          // Increase the value of
                                          // numberOfOperations accordingly
    table.at(target).at(kind) = r;
    ++count;
  }

  Edge* lookup(std::uint8_t kind, qc::Qubit target) {
    assert(kind <
           numberOfStochasticOperations); // There are new operations in OpType.
                                          // Increase the value of
                                          // numberOfOperations accordingly
    lookups++;
    Edge* r = nullptr;
    auto& entry = table.at(target).at(kind);
    if (entry.w.r == nullptr) {
      return r;
    }
    hits++;
    return &entry;
  }

  void clear() {
    if (count > 0) {
      for (auto& t : table) {
        std::fill(t.begin(), t.end(), Edge{});
      }
      count = 0;
    }
  }

  [[nodiscard]] fp hitRatio() const {
    return static_cast<fp>(hits) / static_cast<fp>(lookups);
  }

  std::ostream& printStatistics(std::ostream& os = std::cout) {
    os << "hits: " << hits << ", looks: " << lookups
       << ", ratio: " << hitRatio() << std::endl;
    return os;
  }

private:
  std::size_t nvars;
  std::vector<std::array<Edge, numberOfStochasticOperations>> table;

  // operation table lookup statistics
  std::size_t hits = 0;
  std::size_t lookups = 0;
  std::size_t count = 0;
};
} // namespace dd
