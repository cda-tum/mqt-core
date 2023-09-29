#include "dd/StateGenerator.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>

namespace dd {
StateGenerator::StateGenerator(std::size_t s) : seed(s) {
  if (seed == 0U) {
    std::random_device rd;
    seed = rd();
  }
  generator.seed(seed);
  angleDistribution = std::uniform_real_distribution<double>(0.0, 2.0 * M_PI);
}

std::complex<double> StateGenerator::getRandomComplexOnUnitCircle() {
  const auto angle = angleDistribution(generator);
  return {std::cos(angle), std::sin(angle)};
}

std::pair<std::complex<double>, std::complex<double>>
StateGenerator::getRandomSingleQubitState() {
  const auto a = getRandomComplexOnUnitCircle();
  const auto b = getRandomComplexOnUnitCircle();
  const auto norm = std::sqrt(std::norm(a) + std::norm(b));
  return {a / norm, b / norm};
}

vNode* StateGenerator::generateRandomNode(const Qubit qubit) {

  const auto [a, b] = getRandomSingleQubitState();
  auto* node = vNodeMemoryManager.get();
  node->v = qubit;
  node->e[0].w = cn.lookup(a);
  node->e[1].w = cn.lookup(b);
  return node;
}

void StateGenerator::connectNodeToRandomSuccessors(
    vNode* node, std::vector<vNode*>& successors, std::vector<bool>& used) {
  auto indexDistribution =
      std::uniform_int_distribution<std::size_t>(0U, successors.size() - 1U);
  for (auto& e : node->e) {
    auto index = indexDistribution(generator);
    // if all successors have been used, just take whatever has been selected.
    if (std::all_of(used.begin(), used.end(),
                    [](const auto& b) { return b; })) {
      e.p = successors[index];
      continue;
    }
    // otherwise, loop as long as necessary to find an unused successor
    while (used[index]) {
      index = indexDistribution(generator);
    }
    e.p = successors[index];
    used[index] = true;
  }
}

vEdge StateGenerator::generateRandomVectorDD(
    const std::size_t levels, const std::vector<std::size_t>& nodesPerLevel) {

  assert(levels > 0U && "Number of levels must be greater than zero");
  assert(nodesPerLevel.size() == levels &&
         "Number of levels must match nodesPerLevel size");

  // reserve space for nodes
  std::vector<std::vector<vNode*>> nodes;
  nodes.reserve(levels);
  for (std::size_t i = 0U; i < levels; ++i) {
    nodes.emplace_back();
    assert(nodesPerLevel[i] <= (1U << (levels - i - 1U)) &&
           "Number of nodes per level must not exceed maximum");
    assert(
        (i == 0U || nodesPerLevel[i - 1U] <= 2U * nodesPerLevel[i]) &&
        "Number of nodes per level must not exceed twice the number of nodes "
        "in the level above");
    nodes.back().reserve(nodesPerLevel[i]);
  }

  // generate nodes
  for (std::size_t i = 0U; i < levels; ++i) {
    for (std::size_t j = 0U; j < nodesPerLevel[i]; ++j) {
      nodes[i].emplace_back(generateRandomNode(static_cast<Qubit>(i)));
    }
  }

  // connect nodes from top to bottom
  for (std::size_t i = levels - 1U; i > 0U; --i) {
    std::vector<bool> used(nodes[i - 1U].size(), false);
    for (auto* node : nodes[i]) {
      connectNodeToRandomSuccessors(node, nodes[i - 1U], used);
    }
  }

  return {nodes[levels - 1U][0], Complex::one};
}

} // namespace dd
