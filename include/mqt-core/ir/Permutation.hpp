#pragma once

#include "Definitions.hpp"
#include "operations/Control.hpp"

#include <cstddef>
#include <functional>
#include <map>

namespace qc {
class Permutation : public std::map<Qubit, Qubit> {
public:
  [[nodiscard]] auto apply(const Controls& controls) const -> Controls;
  [[nodiscard]] auto apply(const Targets& targets) const -> Targets;
  [[nodiscard]] auto apply(Qubit qubit) const -> Qubit;
  [[nodiscard]] auto maxKey() const -> Qubit;
  [[nodiscard]] auto maxValue() const -> Qubit;
};
} // namespace qc

// define hash function for Permutation
namespace std {
template <> struct hash<qc::Permutation> {
  std::size_t operator()(const qc::Permutation& p) const {
    std::size_t seed = 0;
    for (const auto& [k, v] : p) {
      qc::hashCombine(seed, k);
      qc::hashCombine(seed, v);
    }
    return seed;
  }
};
} // namespace std
