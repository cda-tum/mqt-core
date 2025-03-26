#pragma once

#include "dd/ComplexNumbers.hpp"
#include "dd/ComputeTable.hpp"
#include "dd/DDDefinitions.hpp"
#include "dd/MemoryManager.hpp"
#include "dd/RealNumberUniqueTable.hpp"
#include "dd/UniqueTable.hpp"
#include "ir/operations/Control.hpp"

#include <array>
#include <fstream>
#include <random>
#include <regex>
#include <stack>

namespace dd {

class Package;

class DDContainerBase {
public:
  struct Config {
    MemoryManager::Config memoryManagerConfig{};
    UniqueTable::UniqueTableConfig uniqueTableConfig{};
  };

  DDContainerBase(std::size_t nqubits, RealNumberUniqueTable& cUt,
                  ComplexNumbers& cn, const Config& config)
      : nqubits(nqubits), cUt(cUt), cn_(cn), mm(config.memoryManagerConfig),
        ut(mm, config.uniqueTableConfig) {}

  void reset();

  bool garbageCollect(bool force = false);

  [[nodiscard]] inline bool possiblyNeedsCollection() const {
    return ut.possiblyNeedsCollection();
  }

  inline void resize(std::size_t n) {
    nqubits = n;
    ut.resize(n);
  }

  void addStatsJson(nlohmann::basic_json<>& j,
                    bool includeIndividualTables) const;

  const UniqueTable& getUniqueTable() const { return ut; }
  UniqueTable& getUniqueTable() { return ut; }

  const MemoryManager& getMemoryManager() const { return mm; }
  MemoryManager& getMemoryManager() { return mm; }

protected:
  ComplexNumbers& getCn() { return cn_; }

  std::size_t nqubits;
  std::reference_wrapper<RealNumberUniqueTable> cUt;
  std::reference_wrapper<ComplexNumbers> cn_;

  MemoryManager mm;
  UniqueTable ut;
};

template <class Node> class DDContainer : public DDContainerBase {
public:
  friend Package;

  struct Config : public DDContainerBase::Config {
    std::size_t computeTableAddNumBuckets =
        decltype(ctAdd)::DEFAULT_NUM_BUCKETS;
  };

  DDContainer(std::size_t nqubits, RealNumberUniqueTable& cUt,
              ComplexNumbers& cn, const Config& config)
      : DDContainerBase(nqubits, cUt, cn, config),
        ctAdd(config.computeTableAddNumBuckets) {}

  void reset() {
    DDContainerBase::reset();
    ctAdd.clear();
    ctKronecker.clear();
  }

  /**
   * @brief Reset all memory managers
   * @arg resizeToTotal If set to true, each manager allocates one chunk of
   * memory as large as all chunks combined before the reset.
   * @see MemoryManager::reset
   */
  /**
   * @brief Clear all compute tables.
   *
   * @details This method clears all entries in the compute tables used for
   * various operations. It resets the state of the compute tables, making them
   * ready for new computations.
   */
  /**
   * @brief Clear all unique tables
   * @see UniqueTable::clear
   * @see RealNumberUniqueTable::clear
   */
  // TODO: docs
  bool garbageCollect(bool force) {
    const bool collect = DDContainerBase::garbageCollect(force);
    if (collect) {
      ctAdd.clear();
      ctKronecker.clear();
    }
    return collect;
  }

  /**
   * @brief Add two decision diagrams.
   *
   * @param x The first DD.
   * @param y The second DD.
   * @return The resulting DD after addition.
   *
   * @details This function performs the addition of two decision diagrams
   * (DDs). It uses a compute table to cache intermediate results and avoid
   * redundant computations. The addition is conducted recursively, where the
   * function traverses the nodes of the DDs, adds corresponding edges, and
   * normalizes the resulting edges. If the nodes are terminal, their weights
   * are directly added. The function ensures that the resulting DD is properly
   * normalized and stored in the unique table to maintain the canonical form.
   */
  Edge<Node> add(const Edge<Node>& x, const Edge<Node>& y) {
    Qubit var{};
    if (!x.isTerminal()) {
      var = x.p->v;
    }
    if (!y.isTerminal() && (y.p->v) > var) {
      var = y.p->v;
    }

    const auto result = add2(CachedEdge{x.p, x.w}, {y.p, y.w}, var);
    return getCn().lookup(result);
  }

  /**
   * @brief Compute the element-wise magnitude sum of two vectors or matrices.
   *
   * For two vectors (or matrices) \p x and \p y, this function returns a result
   * \p r such that for each index \p i:
   * \f$ r[i] = \sqrt{|x[i]|^2 + |y[i]|^2} \f$
   *
   * @param x DD representation of the first operand.
   * @param y DD representation of the second operand.
   * @param var Number of qubits in the DD.
   * @return DD representing the result.
   */
  CachedEdge<Node> addMagnitudes(const CachedEdge<Node>& x,
                                 const CachedEdge<Node>& y, const Qubit var) {
    if (x.w.exactlyZero()) {
      if (y.w.exactlyZero()) {
        return CachedEdge<Node>::zero();
      }
      const auto rWeight = y.w.mag();
      return {y.p, rWeight};
    }
    if (y.w.exactlyZero()) {
      const auto rWeight = x.w.mag();
      return {x.p, rWeight};
    }
    if (x.p == y.p) {
      const auto rWeight = std::sqrt(x.w.mag2() + y.w.mag2());
      return {x.p, rWeight};
    }

    if (const auto* r = ctAddMagnitudes.lookup(x, y); r != nullptr) {
      return *r;
    }

    constexpr std::size_t n = std::tuple_size_v<decltype(x.p->e)>;
    std::array<CachedEdge<Node>, n> edge{};
    for (std::size_t i = 0U; i < n; i++) {
      CachedEdge<Node> e1{};
      if constexpr (std::is_same_v<Node, mNode> ||
                    std::is_same_v<Node, dNode>) {
        if (x.isIdentity() || x.p->v < var) {
          if (i == 0 || i == 3) {
            e1 = x;
          }
        } else {
          auto& xSuccessor = x.p->e[i];
          e1 = {xSuccessor.p, 0};
          if (!xSuccessor.w.exactlyZero()) {
            e1.w = x.w * xSuccessor.w;
          }
        }
      } else {
        auto& xSuccessor = x.p->e[i];
        e1 = {xSuccessor.p, 0};
        if (!xSuccessor.w.exactlyZero()) {
          e1.w = x.w * xSuccessor.w;
        }
      }
      CachedEdge<Node> e2{};
      if constexpr (std::is_same_v<Node, mNode> ||
                    std::is_same_v<Node, dNode>) {
        if (y.isIdentity() || y.p->v < var) {
          if (i == 0 || i == 3) {
            e2 = y;
          }
        } else {
          auto& ySuccessor = y.p->e[i];
          e2 = {ySuccessor.p, 0};
          if (!ySuccessor.w.exactlyZero()) {
            e2.w = y.w * ySuccessor.w;
          }
        }
      } else {
        auto& ySuccessor = y.p->e[i];
        e2 = {ySuccessor.p, 0};
        if (!ySuccessor.w.exactlyZero()) {
          e2.w = y.w * ySuccessor.w;
        }
      }
      edge[i] = addMagnitudes(e1, e2, var - 1);
    }
    auto r = makeDDNode(var, edge);
    ctAddMagnitudes.insert(x, y, r);
    return r;
  }

private:
  /**
   * @brief Internal function to add two decision diagrams.
   *
   * This function is used internally to add two decision diagrams (DDs) of type
   * Node. It is not intended to be called directly.
   *
   * @param x The first DD.
   * @param y The second DD.
   * @param var The variable associated with the current level of recursion.
   * @return The resulting DD after addition.
   */
  [[nodiscard]] CachedEdge<Node>
  add(const CachedEdge<Node>& x, const CachedEdge<Node>& y, const Qubit var) {
    if (x.w.exactlyZero()) {
      if (y.w.exactlyZero()) {
        return CachedEdge<Node>::zero();
      }
      return y;
    }
    if (y.w.exactlyZero()) {
      return x;
    }
    if (x.p == y.p) {
      const auto rWeight = x.w + y.w;
      return {x.p, rWeight};
    }

    if (const auto* r = ctAdd.lookup(x, y); r != nullptr) {
      return *r;
    }

    constexpr std::size_t n = std::tuple_size_v<decltype(x.p->e)>;
    std::array<CachedEdge<Node>, n> edge{};
    for (std::size_t i = 0U; i < n; i++) {
      CachedEdge<Node> e1{};
      if constexpr (std::is_same_v<Node, mNode> ||
                    std::is_same_v<Node, dNode>) {
        if (x.isIdentity() || x.p->v < var) {
          // [ 0 | 1 ]   [ x | 0 ]
          // --------- = ---------
          // [ 2 | 3 ]   [ 0 | x ]
          if (i == 0 || i == 3) {
            e1 = x;
          }
        } else {
          auto& xSuccessor = x.p->e[i];
          e1 = {xSuccessor.p, 0};
          if (!xSuccessor.w.exactlyZero()) {
            e1.w = x.w * xSuccessor.w;
          }
        }
      } else {
        auto& xSuccessor = x.p->e[i];
        e1 = {xSuccessor.p, 0};
        if (!xSuccessor.w.exactlyZero()) {
          e1.w = x.w * xSuccessor.w;
        }
      }
      CachedEdge<Node> e2{};
      if constexpr (std::is_same_v<Node, mNode> ||
                    std::is_same_v<Node, dNode>) {
        if (y.isIdentity() || y.p->v < var) {
          // [ 0 | 1 ]   [ y | 0 ]
          // --------- = ---------
          // [ 2 | 3 ]   [ 0 | y ]
          if (i == 0 || i == 3) {
            e2 = y;
          }
        } else {
          auto& ySuccessor = y.p->e[i];
          e2 = {ySuccessor.p, 0};
          if (!ySuccessor.w.exactlyZero()) {
            e2.w = y.w * ySuccessor.w;
          }
        }
      } else {
        auto& ySuccessor = y.p->e[i];
        e2 = {ySuccessor.p, 0};
        if (!ySuccessor.w.exactlyZero()) {
          e2.w = y.w * ySuccessor.w;
        }
      }

      if constexpr (std::is_same_v<Node, dNode>) {
        dNode::applyDmChangesToNode(e1.p);
        dNode::applyDmChangesToNode(e2.p);
        edge[i] = add(e1, e2, var - 1);
        dNode::revertDmChangesToNode(e2.p);
        dNode::revertDmChangesToNode(e1.p);
      } else {
        edge[i] = add(e1, e2, var - 1);
      }
    }
    auto r = makeDDNode(var, edge);
    ctAdd.insert(x, y, r);
    return r;
  }

public:
  /**
   * @brief Increment the reference count of an edge
   * @details This is the main function for increasing reference counts within
   * the DD package. It increases the reference count of the complex edge weight
   * as well as the DD node itself. If the node newly becomes active, meaning
   * that it had a reference count of zero beforehand, the reference count of
   * all children is recursively increased.
   * @tparam Node The node type of the edge.
   * @param e The edge to increase the reference count of
   */
  void incRef(const Edge<Node>& e) noexcept {
    getCn().incRef(e.w);
    const auto& p = e.p;
    const auto inc = ut.incRef(p);
    if (inc && p->ref == 1U) {
      for (const auto& child : p->e) {
        incRef(child);
      }
    }
  }

  /**
   * @brief Decrement the reference count of an edge
   * @details This is the main function for decreasing reference counts within
   * the DD package. It decreases the reference count of the complex edge weight
   * as well as the DD node itself. If the node newly becomes dead, meaning
   * that its reference count hit zero, the reference count of all children is
   * recursively decreased.
   * @tparam Node The node type of the edge.
   * @param e The edge to decrease the reference count of
   */
  void decRef(const Edge<Node>& e) noexcept {
    getCn().decRef(e.w);
    const auto& p = e.p;
    const auto dec = ut.decRef(p);
    if (dec && p->ref == 0U) {
      for (const auto& child : p->e) {
        decRef(child);
      }
    }
  }

  /**
   * @brief Create a normalized DD node and return an edge pointing to it.
   *
   * @details The node is not recreated if it already exists. This function
   * retrieves a node from the memory manager, sets its variable, and normalizes
   * the edges. If the node resembles the identity, it is skipped. The function
   * then looks up the node in the unique table and returns an edge pointing to
   * it.
   *
   * @tparam Node The type of the node.
   * @tparam EdgeType The type of the edge.
   * @param var The variable associated with the node.
   * @param edges The edges of the node.
   * @param generateDensityMatrix Flag to indicate if a density matrix node
   * should be generated.
   * @return An edge pointing to the normalized DD node.
   */
  template <template <class> class EdgeType>
  [[nodiscard]] EdgeType<Node>
  makeDDNode(const Qubit var,
             const std::array<EdgeType<Node>,
                              std::tuple_size_v<decltype(Node::e)>>& edges,
             [[maybe_unused]] const bool generateDensityMatrix = false) {
    auto p = mm.template get<Node>();
    assert(p->ref == 0U);

    p->v = var;
    if constexpr (std::is_same_v<Node, mNode> || std::is_same_v<Node, dNode>) {
      p->flags = 0;
      if constexpr (std::is_same_v<Node, dNode>) {
        p->setDensityMatrixNodeFlag(generateDensityMatrix);
      }
    }

    auto e = EdgeType<Node>::normalize(p, edges, mm, getCn());
    if constexpr (std::is_same_v<Node, mNode> || std::is_same_v<Node, dNode>) {
      if (!e.isTerminal()) {
        const auto& es = e.p->e;
        // Check if node resembles the identity. If so, skip it.
        if ((es[0].p == es[3].p) &&
            (es[0].w.exactlyOne() && es[1].w.exactlyZero() &&
             es[2].w.exactlyZero() && es[3].w.exactlyOne())) {
          auto* ptr = es[0].p;
          mm.returnEntry(*e.p);
          return EdgeType<Node>{ptr, e.w};
        }
      }
    }

    // look it up in the unique tables
    auto* l = ut.lookup(e.p);

    return EdgeType<Node>{l, e.w};
  }

  /**
   * @brief Delete an edge from the decision diagram.
   *
   * @param e The edge to delete.
   * @param v The variable associated with the edge.
   * @param edgeIdx The index of the edge to delete.
   * @return The modified edge after deletion.
   */
  Edge<Node> deleteEdge(const Edge<Node>& e, const Qubit v,
                        const std::size_t edgeIdx) {
    std::unordered_map<Node*, Edge<Node>> nodes{};
    return deleteEdge(e, v, edgeIdx, nodes);
  }

private:
  /**
   * @brief Helper function to delete an edge from the decision diagram.
   *
   * @param e The edge to delete.
   * @param v The variable associated with the edge.
   * @param edgeIdx The index of the edge to delete.
   * @param nodes A map to keep track of processed nodes.
   * @return The modified edge after deletion.
   */
  Edge<Node> deleteEdge(const Edge<Node>& e, const Qubit v,
                        const std::size_t edgeIdx,
                        std::unordered_map<Node*, Edge<Node>>& nodes) {
    if (e.isTerminal()) {
      return e;
    }

    const auto& nodeIt = nodes.find(e.p);
    Edge<Node> r{};
    if (nodeIt != nodes.end()) {
      r = nodeIt->second;
    } else {
      constexpr std::size_t n = std::tuple_size_v<decltype(e.p->e)>;
      std::array<Edge<Node>, n> edges{};
      if (e.p->v == v) {
        for (std::size_t i = 0; i < n; i++) {
          edges[i] = i == edgeIdx
                         ? Edge<Node>::zero()
                         : e.p->e[i]; // optimization -> node cannot occur below
                                      // again, since dd is assumed to be free
        }
      } else {
        for (std::size_t i = 0; i < n; i++) {
          edges[i] = deleteEdge(e.p->e[i], v, edgeIdx, nodes);
        }
      }

      r = makeDDNode(e.p->v, edges);
      nodes[e.p] = r;
    }
    r.w = getCn().lookup(r.w * e.w);
    return r;
  }

public:
  /// transfers a decision diagram from another package to this package
  Edge<Node> transfer(Edge<Node>& original) {
    if (original.isTerminal()) {
      return {original.p, getCn().lookup(original.w)};
    }

    // POST ORDER TRAVERSAL USING ONE STACK
    // https://www.geeksforgeeks.org/iterative-postorder-traversal-using-stack/
    Edge<Node> root{};
    std::stack<Edge<Node>*> stack;

    std::unordered_map<decltype(original.p), decltype(original.p)> mappedNode{};

    Edge<Node>* currentEdge = &original;
    constexpr std::size_t n = std::tuple_size_v<decltype(original.p->e)>;
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-do-while)
    do {
      while (currentEdge != nullptr && !currentEdge->isTerminal()) {
        for (std::size_t i = n - 1; i > 0; --i) {
          auto& edge = currentEdge->p->e[i];
          if (edge.isTerminal()) {
            continue;
          }
          if (edge.w.approximatelyZero()) {
            continue;
          }
          if (mappedNode.find(edge.p) != mappedNode.end()) {
            continue;
          }

          // non-zero edge to be included
          stack.push(&edge);
        }
        stack.push(currentEdge);
        currentEdge = &currentEdge->p->e[0];
      }
      currentEdge = stack.top();
      stack.pop();

      bool hasChild = false;
      for (std::size_t i = 1; i < n && !hasChild; ++i) {
        auto& edge = currentEdge->p->e[i];
        if (edge.w.approximatelyZero()) {
          continue;
        }
        if (mappedNode.find(edge.p) != mappedNode.end()) {
          continue;
        }
        hasChild = edge.p == stack.top()->p;
      }

      if (hasChild) {
        Edge<Node>* temp = stack.top();
        stack.pop();
        stack.push(currentEdge);
        currentEdge = temp;
      } else {
        if (mappedNode.find(currentEdge->p) != mappedNode.end()) {
          currentEdge = nullptr;
          continue;
        }
        std::array<Edge<Node>, n> edges{};
        for (std::size_t i = 0; i < n; i++) {
          if (currentEdge->p->e[i].isTerminal()) {
            edges[i].p = currentEdge->p->e[i].p;
          } else {
            edges[i].p = mappedNode[currentEdge->p->e[i].p];
          }
          edges[i].w = getCn().lookup(currentEdge->p->e[i].w);
        }
        root = makeDDNode(currentEdge->p->v, edges);
        mappedNode[currentEdge->p] = root.p;
        currentEdge = nullptr;
      }
    } while (!stack.empty());
    root.w = getCn().lookup(original.w * root.w);
    return root;
  }

  ///
  /// Deserialization
  /// Note: do not rely on the binary format being portable across different
  /// architectures/platforms
  ///
  template <class Edge = Edge<Node>,
            std::size_t N = std::tuple_size_v<decltype(Node::e)>>
  [[nodiscard]] Edge deserialize(std::istream& is,
                                 const bool readBinary = false) {
    auto result = CachedEdge<Node>{};
    ComplexValue rootweight{};

    std::unordered_map<std::int64_t, Node*> nodes{};
    std::int64_t nodeIndex{};
    Qubit v{};
    std::array<ComplexValue, N> edgeWeights{};
    std::array<std::int64_t, N> edgeIndices{};
    edgeIndices.fill(-2);

    if (readBinary) {
      std::remove_const_t<decltype(SERIALIZATION_VERSION)> version{};
      is.read(reinterpret_cast<char*>(&version),
              sizeof(decltype(SERIALIZATION_VERSION)));
      if (version != SERIALIZATION_VERSION) {
        throw std::runtime_error(
            "Wrong Version of serialization file version. version of file: " +
            std::to_string(version) +
            "; current version: " + std::to_string(SERIALIZATION_VERSION));
      }

      if (!is.eof()) {
        rootweight.readBinary(is);
      }

      while (is.read(reinterpret_cast<char*>(&nodeIndex),
                     sizeof(decltype(nodeIndex)))) {
        is.read(reinterpret_cast<char*>(&v), sizeof(decltype(v)));
        for (std::size_t i = 0U; i < N; i++) {
          is.read(reinterpret_cast<char*>(&edgeIndices[i]),
                  sizeof(decltype(edgeIndices[i])));
          edgeWeights[i].readBinary(is);
        }
        result = deserializeNode(nodeIndex, v, edgeIndices, edgeWeights, nodes);
      }
    } else {
      std::string version;
      std::getline(is, version);
      if (std::stoi(version) != SERIALIZATION_VERSION) {
        throw std::runtime_error(
            "Wrong Version of serialization file version. version of file: " +
            version +
            "; current version: " + std::to_string(SERIALIZATION_VERSION));
      }

      const std::string complexRealRegex =
          R"(([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?(?![ \d\.]*(?:[eE][+-])?\d*[iI]))?)";
      const std::string complexImagRegex =
          R"(( ?[+-]? ?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)?[iI])?)";
      const std::string edgeRegex =
          " \\(((-?\\d+) (" + complexRealRegex + complexImagRegex + "))?\\)";
      const std::regex complexWeightRegex(complexRealRegex + complexImagRegex);

      std::string lineConstruct = "(\\d+) (\\d+)";
      for (std::size_t i = 0U; i < N; ++i) {
        lineConstruct += "(?:" + edgeRegex + ")";
      }
      lineConstruct += " *(?:#.*)?";
      const std::regex lineRegex(lineConstruct);
      std::smatch m;

      std::string line;
      if (std::getline(is, line)) {
        if (!std::regex_match(line, m, complexWeightRegex)) {
          throw std::runtime_error("Regex did not match second line: " + line);
        }
        rootweight.fromString(m.str(1), m.str(2));
      }

      while (std::getline(is, line)) {
        if (line.empty() || line.size() == 1) {
          continue;
        }

        if (!std::regex_match(line, m, lineRegex)) {
          throw std::runtime_error("Regex did not match line: " + line);
        }

        // match 1: node_idx
        // match 2: qubit_idx

        // repeats for every edge
        // match 3: edge content
        // match 4: edge_target_idx
        // match 5: real + imag (without i)
        // match 6: real
        // match 7: imag (without i)
        nodeIndex = std::stoi(m.str(1));
        v = static_cast<Qubit>(std::stoi(m.str(2)));

        for (auto edgeIdx = 3U, i = 0U; i < N; i++, edgeIdx += 5) {
          if (m.str(edgeIdx).empty()) {
            continue;
          }

          edgeIndices[i] = std::stoi(m.str(edgeIdx + 1));
          edgeWeights[i].fromString(m.str(edgeIdx + 3), m.str(edgeIdx + 4));
        }

        result = deserializeNode(nodeIndex, v, edgeIndices, edgeWeights, nodes);
      }
    }
    return {result.p, getCn().lookup(result.w * rootweight)};
  }

  template <class Edge = Edge<Node>>
  [[nodiscard]] Edge deserialize(const std::string& inputFilename,
                                 const bool readBinary) {
    auto ifs = std::ifstream(inputFilename, std::ios::binary);

    if (!ifs.good()) {
      throw std::invalid_argument("Cannot open serialized file: " +
                                  inputFilename);
    }

    return deserialize<Node>(ifs, readBinary);
  }

private:
  template <std::size_t N = std::tuple_size_v<decltype(Node::e)>>
  [[nodiscard]] CachedEdge<Node>
  deserializeNode(const std::int64_t index, const Qubit v,
                  std::array<std::int64_t, N>& edgeIdx,
                  const std::array<ComplexValue, N>& edgeWeight,
                  std::unordered_map<std::int64_t, Node*>& nodes) {
    if (index == -1) {
      return CachedEdge<Node>::zero();
    }

    std::array<CachedEdge<Node>, N> edges{};
    for (auto i = 0U; i < N; ++i) {
      if (edgeIdx[i] == -2) {
        edges[i] = CachedEdge<Node>::zero();
      } else {
        if (edgeIdx[i] == -1) {
          edges[i] = CachedEdge<Node>::one();
        } else {
          edges[i].p = nodes[edgeIdx[i]];
        }
        edges[i].w = edgeWeight[i];
      }
    }
    // reset
    edgeIdx.fill(-2);

    auto r = makeDDNode(v, edges);
    nodes[index] = r.p;
    return r;
  }

public:
  static constexpr auto NODE_MEMORY_MIB =
      static_cast<double>(sizeof(Node)) / static_cast<double>(1ULL << 20U);
  static constexpr auto EDGE_MEMORY_MIB =
      static_cast<double>(sizeof(Edge<Node>)) /
      static_cast<double>(1ULL << 20U);

  [[nodiscard]] double computeActiveMemoryMiB() const {
    const auto activeEntries = static_cast<double>(ut.getNumActiveEntries());
    return (activeEntries * NODE_MEMORY_MIB) +
           (activeEntries * EDGE_MEMORY_MIB);
  }
  [[nodiscard]] double computePeakMemoryMiB() const {
    const auto peakEntries = static_cast<double>(ut.getPeakNumActiveEntries());
    return (peakEntries * NODE_MEMORY_MIB) + (peakEntries * EDGE_MEMORY_MIB);
  }

  ///
  /// Kronecker/tensor product
  ///

  /**
   * @brief Computes the Kronecker product of two decision diagrams.
   *
   * @param x The first decision diagram.
   * @param y The second decision diagram.
   * @param yNumQubits The number of qubits in the second decision diagram.
   * @param incIdx Whether to increment the index of the nodes in the second
   * decision diagram.
   * @return The resulting decision diagram after computing the Kronecker
   * product.
   * @throws std::invalid_argument if the node type is `dNode` (density
   * matrices).
   */
  Edge<Node> kronecker(const Edge<Node>& x, const Edge<Node>& y,
                       const std::size_t yNumQubits, const bool incIdx = true) {
    if constexpr (std::is_same_v<Node, dNode>) {
      throw std::invalid_argument(
          "Kronecker is currently not supported for density matrices");
    }
    const auto e = kronecker2(x, y, yNumQubits, incIdx);
    return getCn().lookup(e);
  }

private:
  /**
   * @brief Internal function to compute the Kronecker product of two decision
   * diagrams.
   *
   * This function is used internally to compute the Kronecker product of two
   * decision diagrams (DDs) of type Node. It is not intended to be called
   * directly.
   *
   * @param x The first decision diagram.
   * @param y The second decision diagram.
   * @param yNumQubits The number of qubits in the second decision diagram.
   * @param incIdx Whether to increment the qubit index.
   * @return The resulting decision diagram after the Kronecker product.
   */
  CachedEdge<Node> kronecker2(const Edge<Node>& x, const Edge<Node>& y,
                              const std::size_t yNumQubits,
                              const bool incIdx = true) {
    if (x.w.exactlyZero() || y.w.exactlyZero()) {
      return CachedEdge<Node>::zero();
    }
    const auto xWeight = static_cast<ComplexValue>(x.w);
    if (xWeight.approximatelyZero()) {
      return CachedEdge<Node>::zero();
    }
    const auto yWeight = static_cast<ComplexValue>(y.w);
    if (yWeight.approximatelyZero()) {
      return CachedEdge<Node>::zero();
    }
    const auto rWeight = xWeight * yWeight;
    if (rWeight.approximatelyZero()) {
      return CachedEdge<Node>::zero();
    }

    if (x.isTerminal() && y.isTerminal()) {
      return {x.p, rWeight};
    }

    if constexpr (std::is_same_v<Node, mNode> || std::is_same_v<Node, dNode>) {
      if (x.isIdentity()) {
        return {y.p, rWeight};
      }
    } else {
      if (x.isTerminal()) {
        return {y.p, rWeight};
      }
      if (y.isTerminal()) {
        return {x.p, rWeight};
      }
    }

    // check if we already computed the product before and return the result
    if (const auto* r = ctKronecker.lookup(x.p, y.p); r != nullptr) {
      return {r->p, rWeight};
    }

    constexpr std::size_t n = std::tuple_size_v<decltype(x.p->e)>;
    std::array<CachedEdge<Node>, n> edge{};
    for (auto i = 0U; i < n; ++i) {
      edge[i] = kronecker2(x.p->e[i], y, yNumQubits, incIdx);
    }

    // Increase the qubit index
    Qubit idx = x.p->v;
    if (incIdx) {
      // use the given number of qubits if y is an identity
      if constexpr (std::is_same_v<Node, mNode> ||
                    std::is_same_v<Node, dNode>) {
        if (y.isIdentity()) {
          idx += static_cast<Qubit>(yNumQubits);
        } else {
          idx += static_cast<Qubit>(y.p->v + 1U);
        }
      } else {
        idx += static_cast<Qubit>(y.p->v + 1U);
      }
    }
    auto e = makeDDNode(idx, edge, true);
    ctKronecker.insert(x.p, y.p, {e.p, e.w});
    return {e.p, rWeight};
  }

protected:
  ComputeTable<CachedEdge<Node>, CachedEdge<Node>, CachedEdge<Node>> ctAdd;
  ComputeTable<CachedEdge<Node>, CachedEdge<Node>, CachedEdge<Node>>
      ctAddMagnitudes;
  ComputeTable<Node*, Node*, CachedEdge<Node>> ctKronecker;
};

} // namespace dd