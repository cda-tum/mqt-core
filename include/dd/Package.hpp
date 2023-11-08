#pragma once

#include "dd/Complex.hpp"
#include "dd/ComplexNumbers.hpp"
#include "dd/ComplexValue.hpp"
#include "dd/ComputeTable.hpp"
#include "dd/DDDefinitions.hpp"
#include "dd/DensityNoiseTable.hpp"
#include "dd/Edge.hpp"
#include "dd/GateMatrixDefinitions.hpp"
#include "dd/Package_fwd.hpp"
#include "dd/StochasticNoiseOperationTable.hpp"
#include "dd/UnaryComputeTable.hpp"
#include "dd/UniqueTable.hpp"
#include "operations/Control.hpp"

#include <algorithm>
#include <array>
#include <bitset>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <queue>
#include <random>
#include <regex>
#include <set>
#include <stack>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace dd {

template <class Config> class Package {
  static_assert(std::is_base_of_v<DDPackageConfig, Config>,
                "Config must be derived from DDPackageConfig");

  ///
  /// Construction, destruction, information and reset
  ///
public:
  static constexpr std::size_t MAX_POSSIBLE_QUBITS =
      static_cast<std::size_t>(std::numeric_limits<Qubit>::max()) + 1U;
  static constexpr std::size_t DEFAULT_QUBITS = 32U;
  explicit Package(std::size_t nq = DEFAULT_QUBITS) : nqubits(nq) {
    resize(nq);
  };
  ~Package() = default;
  Package(const Package& package) = delete;

  Package& operator=(const Package& package) = delete;

  // resize the package instance
  void resize(std::size_t nq) {
    if (nq > MAX_POSSIBLE_QUBITS) {
      throw std::invalid_argument("Requested too many qubits from package. "
                                  "Qubit datatype only allows up to " +
                                  std::to_string(MAX_POSSIBLE_QUBITS) +
                                  " qubits, while " + std::to_string(nq) +
                                  " were requested. Please recompile the "
                                  "package with a wider Qubit type!");
    }
    nqubits = nq;
    vUniqueTable.resize(nqubits);
    mUniqueTable.resize(nqubits);
    dUniqueTable.resize(nqubits);
    stochasticNoiseOperationCache.resize(nqubits);
    idTable.resize(nqubits);
  }

  // reset package state
  void reset() {
    clearUniqueTables();
    resetMemoryManagers();
    clearComputeTables();
  }

  /// Get the number of qubits
  [[nodiscard]] auto qubits() const { return nqubits; }

private:
  std::size_t nqubits;

public:
  /// The memory manager for vector nodes
  MemoryManager<vNode> vMemoryManager{Config::UT_VEC_INITIAL_ALLOCATION_SIZE};
  /// The memory manager for matrix nodes
  MemoryManager<mNode> mMemoryManager{Config::UT_MAT_INITIAL_ALLOCATION_SIZE};
  /// The memory manager for density matrix nodes
  MemoryManager<dNode> dMemoryManager{Config::UT_DM_INITIAL_ALLOCATION_SIZE};
  /**
   * @brief The memory manager for complex numbers
   * @note The real and imaginary part of complex numbers are treated
   * separately. Hence, it suffices for the manager to only manage real numbers.
   */
  MemoryManager<RealNumber> cMemoryManager{};
  /**
   * @brief The cache manager for complex numbers
   * @note Similar to the memory manager, the cache only maintains real entries,
   * but typically gives them out in pairs to form complex numbers.
   */
  MemoryManager<RealNumber> cCacheManager{};

  /**
   * @brief Get the memory manager for a given type
   * @tparam T The type to get the manager for
   * @return A reference to the manager
   */
  template <class T> [[nodiscard]] auto& getMemoryManager() {
    if constexpr (std::is_same_v<T, vNode>) {
      return vMemoryManager;
    } else if constexpr (std::is_same_v<T, mNode>) {
      return mMemoryManager;
    } else if constexpr (std::is_same_v<T, dNode>) {
      return dMemoryManager;
    } else if constexpr (std::is_same_v<T, RealNumber>) {
      return cMemoryManager;
    }
  }

  /**
   * @brief Reset all memory managers
   * @arg resizeToTotal If set to true, each manager allocates one chunk of
   * memory as large as all chunks combined before the reset.
   * @see MemoryManager::reset
   */
  void resetMemoryManagers(const bool resizeToTotal = false) {
    vMemoryManager.reset(resizeToTotal);
    mMemoryManager.reset(resizeToTotal);
    dMemoryManager.reset(resizeToTotal);
    cMemoryManager.reset(resizeToTotal);
    cCacheManager.reset(resizeToTotal);
  }

  /// The unique table used for vector nodes
  UniqueTable<vNode, Config::UT_VEC_NBUCKET> vUniqueTable{0U, vMemoryManager};
  /// The unique table used for matrix nodes
  UniqueTable<mNode, Config::UT_MAT_NBUCKET> mUniqueTable{0U, mMemoryManager};
  /// The unique table used for density matrix nodes
  UniqueTable<dNode, Config::UT_DM_NBUCKET> dUniqueTable{0U, dMemoryManager};
  /**
   * @brief The unique table used for complex numbers
   * @note The table actually only stores real numbers in the interval [0, 1],
   * but is used to manages all complex numbers throughout the package.
   * @see RealNumberUniqueTable
   */
  RealNumberUniqueTable cUniqueTable{cMemoryManager};
  ComplexNumbers cn{cUniqueTable, cCacheManager};

  /**
   * @brief Get the unique table for a given type
   * @tparam T The type to get the unique table for
   * @return A reference to the unique table
   */
  template <class T> [[nodiscard]] auto& getUniqueTable() {
    if constexpr (std::is_same_v<T, vNode>) {
      return vUniqueTable;
    } else if constexpr (std::is_same_v<T, mNode>) {
      return mUniqueTable;
    } else if constexpr (std::is_same_v<T, dNode>) {
      return dUniqueTable;
    } else if constexpr (std::is_same_v<T, RealNumber>) {
      return cUniqueTable;
    }
  }

  /**
   * @brief Clear all unique tables
   * @see UniqueTable::clear
   * @see RealNumberUniqueTable::clear
   */
  void clearUniqueTables() {
    vUniqueTable.clear();
    mUniqueTable.clear();
    dUniqueTable.clear();
    cUniqueTable.clear();
  }

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
  template <class Node> void incRef(const Edge<Node>& e) noexcept {
    cn.incRef(e.w);
    const auto& p = e.p;
    const auto inc = getUniqueTable<Node>().incRef(p);
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
  template <class Node> void decRef(const Edge<Node>& e) noexcept {
    cn.decRef(e.w);
    const auto& p = e.p;
    const auto dec = getUniqueTable<Node>().decRef(p);
    if (dec && p->ref == 0U) {
      for (const auto& child : p->e) {
        decRef(child);
      }
    }
  }

  bool garbageCollect(bool force = false) {
    // return immediately if no table needs collection
    if (!force && !vUniqueTable.possiblyNeedsCollection() &&
        !mUniqueTable.possiblyNeedsCollection() &&
        !dUniqueTable.possiblyNeedsCollection() &&
        !cUniqueTable.possiblyNeedsCollection()) {
      return false;
    }

    auto cCollect = cUniqueTable.garbageCollect(force);
    if (cCollect > 0) {
      // Collecting garbage in the complex numbers table requires collecting the
      // node tables as well
      force = true;
    }
    auto vCollect = vUniqueTable.garbageCollect(force);
    auto mCollect = mUniqueTable.garbageCollect(force);
    auto dCollect = dUniqueTable.garbageCollect(force);

    // invalidate all compute tables involving vectors if any vector node has
    // been collected
    if (vCollect > 0) {
      vectorAdd.clear();
      vectorInnerProduct.clear();
      vectorKronecker.clear();
      matrixVectorMultiplication.clear();
    }
    // invalidate all compute tables involving matrices if any matrix node has
    // been collected
    if (mCollect > 0 || dCollect > 0) {
      matrixAdd.clear();
      conjugateMatrixTranspose.clear();
      matrixKronecker.clear();
      matrixVectorMultiplication.clear();
      matrixMatrixMultiplication.clear();
      clearIdentityTable();
      stochasticNoiseOperationCache.clear();
      densityAdd.clear();
      densityDensityMultiplication.clear();
      densityNoise.clear();
    }
    // invalidate all compute tables where any component of the entry contains
    // numbers from the complex table if any complex numbers were collected
    if (cCollect > 0) {
      matrixVectorMultiplication.clear();
      matrixMatrixMultiplication.clear();
      conjugateMatrixTranspose.clear();
      vectorInnerProduct.clear();
      vectorKronecker.clear();
      matrixKronecker.clear();
      stochasticNoiseOperationCache.clear();
      densityAdd.clear();
      densityDensityMultiplication.clear();
      densityNoise.clear();
    }
    return vCollect > 0 || mCollect > 0 || cCollect > 0;
  }

  ///
  /// Vector nodes, edges and quantum states
  ///
  vEdge normalize(const vEdge& e, bool cached) {
    auto zero = std::array{e.p->e[0].w.approximatelyZero(),
                           e.p->e[1].w.approximatelyZero()};

    // make sure to release cached numbers approximately zero, but not exactly
    // zero
    if (cached) {
      for (auto i = 0U; i < RADIX; i++) {
        if (zero[i]) {
          cn.returnToCache(e.p->e[i].w);
          e.p->e[i] = vEdge::zero();
        }
      }
    }

    if (zero[0]) {
      // all equal to zero
      if (zero[1]) {
        if (!cached && !e.isTerminal()) {
          // If it is not a cached computation, the node has to be put back into
          // the chain
          vMemoryManager.returnEntry(e.p);
        }
        return vEdge::zero();
      }

      auto r = e;
      auto& w = r.p->e[1].w;
      if (cached) {
        r.w = w;
      } else {
        r.w = cn.lookup(w);
      }
      w = Complex::one();
      return r;
    }

    if (zero[1]) {
      auto r = e;
      auto& w = r.p->e[0].w;
      if (cached) {
        r.w = w;
      } else {
        r.w = cn.lookup(w);
      }
      w = Complex::one();
      return r;
    }

    const auto mag0 = ComplexNumbers::mag2(e.p->e[0].w);
    const auto mag1 = ComplexNumbers::mag2(e.p->e[1].w);
    const auto norm2 = mag0 + mag1;
    const auto mag2Max = (mag0 + RealNumber::eps >= mag1) ? mag0 : mag1;
    const auto argMax = (mag0 + RealNumber::eps >= mag1) ? 0 : 1;
    const auto norm = std::sqrt(norm2);
    const auto magMax = std::sqrt(mag2Max);
    const auto commonFactor = norm / magMax;

    auto r = e;
    auto& max = r.p->e[static_cast<std::size_t>(argMax)];
    if (cached) {
      if (max.w.exactlyOne()) {
        r.w = cn.lookup(commonFactor, 0.);
      } else {
        r.w = max.w;
        r.w.r->value *= commonFactor;
        r.w.i->value *= commonFactor;
      }
    } else {
      r.w = cn.lookup(RealNumber::val(max.w.r) * commonFactor,
                      RealNumber::val(max.w.i) * commonFactor);
      if (r.w.approximatelyZero()) {
        return vEdge::zero();
      }
    }

    max.w = cn.lookup(magMax / norm);
    if (max.w.exactlyZero()) {
      max = vEdge::zero();
    }

    const auto argMin = (argMax + 1) % 2;
    auto& min = r.p->e[static_cast<std::size_t>(argMin)];
    if (cached) {
      ComplexNumbers::div(min.w, min.w, r.w);
      min.w = cn.lookup(min.w, true);
    } else {
      min.w = cn.lookup(min.w / r.w);
    }
    if (min.w.exactlyZero()) {
      min = vEdge::zero();
    }

    return r;
  }

  dEdge makeZeroDensityOperator(const std::size_t n) {
    auto f = dEdge::one();
    for (std::size_t p = 0; p < n; p++) {
      f = makeDDNode(
          static_cast<Qubit>(p),
          std::array{f, dEdge::zero(), dEdge::zero(), dEdge::zero()});
    }
    return f;
  }

  // generate |0...0> with n qubits
  vEdge makeZeroState(const std::size_t n, const std::size_t start = 0) {
    if (n + start > nqubits) {
      throw std::runtime_error{
          "Requested state with " + std::to_string(n + start) +
          " qubits, but current package configuration only supports up to " +
          std::to_string(nqubits) +
          " qubits. Please allocate a larger package instance."};
    }
    auto f = vEdge::one();
    for (std::size_t p = start; p < n + start; p++) {
      f = makeDDNode(static_cast<Qubit>(p), std::array{f, vEdge::zero()});
    }
    return f;
  }
  // generate computational basis state |i> with n qubits
  vEdge makeBasisState(const std::size_t n, const std::vector<bool>& state,
                       const std::size_t start = 0) {
    if (n + start > nqubits) {
      throw std::runtime_error{
          "Requested state with " + std::to_string(n + start) +
          " qubits, but current package configuration only supports up to " +
          std::to_string(nqubits) +
          " qubits. Please allocate a larger package instance."};
    }
    auto f = vEdge::one();
    for (std::size_t p = start; p < n + start; ++p) {
      if (!state[p]) {
        f = makeDDNode(static_cast<Qubit>(p), std::array{f, vEdge::zero()});
      } else {
        f = makeDDNode(static_cast<Qubit>(p), std::array{vEdge::zero(), f});
      }
    }
    return f;
  }
  // generate general basis state with n qubits
  vEdge makeBasisState(const std::size_t n,
                       const std::vector<BasisStates>& state,
                       const std::size_t start = 0) {
    if (n + start > nqubits) {
      throw std::runtime_error{
          "Requested state with " + std::to_string(n + start) +
          " qubits, but current package configuration only supports up to " +
          std::to_string(nqubits) +
          " qubits. Please allocate a larger package instance."};
    }
    if (state.size() < n) {
      throw std::runtime_error(
          "Insufficient qubit states provided. Requested " + std::to_string(n) +
          ", but received " + std::to_string(state.size()));
    }

    auto f = vEdge::one();
    for (std::size_t p = start; p < n + start; ++p) {
      switch (state[p]) {
      case BasisStates::zero:
        f = makeDDNode(static_cast<Qubit>(p), std::array{f, vEdge::zero()});
        break;
      case BasisStates::one:
        f = makeDDNode(static_cast<Qubit>(p), std::array{vEdge::zero(), f});
        break;
      case BasisStates::plus:
        f = makeDDNode(
            static_cast<Qubit>(p),
            std::array<vEdge, RADIX>{{{f.p, cn.lookup(dd::SQRT2_2)},
                                      {f.p, cn.lookup(dd::SQRT2_2)}}});
        break;
      case BasisStates::minus:
        f = makeDDNode(
            static_cast<Qubit>(p),
            std::array<vEdge, RADIX>{{{f.p, cn.lookup(dd::SQRT2_2)},
                                      {f.p, cn.lookup(-dd::SQRT2_2)}}});
        break;
      case BasisStates::right:
        f = makeDDNode(
            static_cast<Qubit>(p),
            std::array<vEdge, RADIX>{{{f.p, cn.lookup(dd::SQRT2_2)},
                                      {f.p, cn.lookup(0, dd::SQRT2_2)}}});
        break;
      case BasisStates::left:
        f = makeDDNode(
            static_cast<Qubit>(p),
            std::array<vEdge, RADIX>{{{f.p, cn.lookup(dd::SQRT2_2)},
                                      {f.p, cn.lookup(0, -dd::SQRT2_2)}}});
        break;
      }
    }
    return f;
  }

  // generate the decision diagram from an arbitrary state vector
  vEdge makeStateFromVector(const CVec& stateVector) {
    if (stateVector.empty()) {
      return vEdge::one();
    }
    const auto& length = stateVector.size();
    if ((length & (length - 1)) != 0) {
      throw std::invalid_argument(
          "State vector must have a length of a power of two.");
    }

    if (length == 1) {
      return vEdge::terminal(cn.lookup(stateVector[0]));
    }

    [[maybe_unused]] const auto before = cn.cacheCount();

    const auto level = static_cast<Qubit>(std::log2(length) - 1);
    auto state =
        makeStateFromVector(stateVector.begin(), stateVector.end(), level);

    // the recursive function makes use of the cache, so we have to clean it up
    state.w = cn.lookup(state.w, true);

    [[maybe_unused]] const auto after = cn.cacheCount();
    assert(after == before);
    return state;
  }

  /**
      Converts a given matrix to a decision diagram
      @param matrix A complex matrix to convert to a DD.
      @return An mEdge that represents the DD.
      @throws std::invalid_argument If the given matrix is not square or its
  length is not a power of two.
  **/
  mEdge makeDDFromMatrix(const CMat& matrix) {
    if (matrix.empty()) {
      return mEdge::one();
    }

    const auto& length = matrix.size();
    if ((length & (length - 1)) != 0) {
      throw std::invalid_argument(
          "Matrix must have a length of a power of two.");
    }

    const auto& width = matrix[0].size();
    if (length != width) {
      throw std::invalid_argument("Matrix must be square.");
    }

    if (length == 1) {
      return mEdge::terminal(cn.lookup(matrix[0][0]));
    }

    [[maybe_unused]] const auto before = cn.cacheCount();

    const auto level = static_cast<Qubit>(std::log2(length) - 1);

    auto matrixDD = makeDDFromMatrix(matrix, level, 0, length, 0, width);

    matrixDD.w = cn.lookup(matrixDD.w, true);

    [[maybe_unused]] const auto after = cn.cacheCount();
    assert(after == before);
    return matrixDD;
  }

  ///
  /// Matrix nodes, edges and quantum gates
  ///
  template <class Node> Edge<Node> normalize(const Edge<Node>& e, bool cached) {
    if constexpr (std::is_same_v<Node, mNode> || std::is_same_v<Node, dNode>) {
      auto argmax = -1;

      auto zero = std::array{
          e.p->e[0].w.approximatelyZero(), e.p->e[1].w.approximatelyZero(),
          e.p->e[2].w.approximatelyZero(), e.p->e[3].w.approximatelyZero()};

      // make sure to release cached numbers approximately zero, but not exactly
      // zero
      if (cached) {
        for (auto i = 0U; i < NEDGE; i++) {
          auto& successor = e.p->e[i];
          if (zero[i]) {
            cn.returnToCache(successor.w);
            successor = Edge<Node>::zero();
          }
        }
      }

      fp max = 0;
      auto maxc = Complex::one();
      // determine max amplitude
      for (auto i = 0U; i < NEDGE; ++i) {
        if (zero[i]) {
          continue;
        }
        const auto& w = e.p->e[i].w;
        if (argmax == -1) {
          argmax = static_cast<decltype(argmax)>(i);
          max = ComplexNumbers::mag2(w);
          maxc = w;
        } else {
          auto mag = ComplexNumbers::mag2(w);
          if (mag - max > RealNumber::eps) {
            argmax = static_cast<decltype(argmax)>(i);
            max = mag;
            maxc = w;
          }
        }
      }

      // all equal to zero
      if (argmax == -1) {
        if (!e.isTerminal()) {
          getMemoryManager<Node>().returnEntry(e.p);
        }
        return Edge<Node>::zero();
      }

      auto r = e;
      // divide each entry by max
      for (auto i = 0U; i < NEDGE; ++i) {
        if (static_cast<decltype(argmax)>(i) == argmax) {
          r.p->e[i].w = Complex::one();
          if (r.w.exactlyOne()) {
            r.w = maxc;
            continue;
          }

          if (cached) {
            ComplexNumbers::mul(r.w, r.w, maxc);
          } else {
            r.w = cn.lookup(r.w * maxc);
          }
        } else {
          auto& successor = r.p->e[i];
          if (zero[i]) {
            assert(successor.w.exactlyZero() &&
                   "Should have been set to zero at the start");
            continue;
          }
          // TODO: it might be worth revisiting whether this check actually
          // improves performance or rather causes more instability.
          if (successor.w.approximatelyOne()) {
            if (cached) {
              cn.returnToCache(successor.w);
            }
            successor.w = Complex::one();
          }
          const auto c = successor.w / maxc;
          if (cached) {
            cn.returnToCache(successor.w);
          }
          successor.w = cn.lookup(c);
        }
      }
      return r;
    }
  }

  // build matrix representation for a single gate on an n-qubit circuit
  mEdge makeGateDD(const GateMatrix& mat, const std::size_t n,
                   const qc::Qubit target, const std::size_t start = 0) {
    return makeGateDD(mat, n, qc::Controls{}, target, start);
  }
  mEdge makeGateDD(const GateMatrix& mat, const std::size_t n,
                   const qc::Control& control, const qc::Qubit target,
                   const std::size_t start = 0) {
    return makeGateDD(mat, n, qc::Controls{control}, target, start);
  }
  mEdge makeGateDD(const GateMatrix& mat, const std::size_t n,
                   const qc::Controls& controls, const qc::Qubit target,
                   const std::size_t start = 0) {
    if (n + start > nqubits) {
      throw std::runtime_error{
          "Requested gate with " + std::to_string(n + start) +
          " qubits, but current package configuration only supports up to " +
          std::to_string(nqubits) +
          " qubits. Please allocate a larger package instance."};
    }
    std::array<mEdge, NEDGE> em{};
    auto it = controls.begin();
    for (auto i = 0U; i < NEDGE; ++i) {
      em[i] = mEdge::terminal(cn.lookup(mat[i]));
    }

    // process lines below target
    auto z = static_cast<Qubit>(start);
    for (; z < static_cast<Qubit>(target); ++z) {
      for (auto i1 = 0U; i1 < RADIX; ++i1) {
        for (auto i2 = 0U; i2 < RADIX; ++i2) {
          auto i = i1 * RADIX + i2;
          if (it != controls.end() && it->qubit == z) {
            auto edges = std::array{mEdge::zero(), mEdge::zero(), mEdge::zero(),
                                    mEdge::zero()};
            if (it->type == qc::Control::Type::Neg) { // neg. control
              edges[0] = em[i];
              if (i1 == i2) {
                if (z == 0U) {
                  edges[3] = mEdge::one();
                } else {
                  edges[3] = makeIdent(start, z - 1U);
                }
              }
            } else { // pos. control
              edges[3] = em[i];
              if (i1 == i2) {
                if (z == 0U) {
                  edges[0] = mEdge::one();
                } else {
                  edges[0] = makeIdent(start, z - 1U);
                }
              }
            }
            em[i] = makeDDNode(z, edges);
          } else { // not connected
            em[i] = makeDDNode(
                z, std::array{em[i], mEdge::zero(), mEdge::zero(), em[i]});
          }
        }
      }
      if (it != controls.end() && it->qubit == z) {
        ++it;
      }
    }

    // target line
    auto e = makeDDNode(z, em);

    // process lines above target
    for (; z < static_cast<Qubit>(n - 1 + start); z++) {
      auto q = static_cast<Qubit>(z + 1);
      if (it != controls.end() && it->qubit == static_cast<qc::Qubit>(q)) {
        if (it->type == qc::Control::Type::Neg) { // neg. control
          e = makeDDNode(q, std::array{e, mEdge::zero(), mEdge::zero(),
                                       makeIdent(start, q - 1U)});
        } else { // pos. control
          e = makeDDNode(q, std::array{makeIdent(start, q - 1U), mEdge::zero(),
                                       mEdge::zero(), e});
        }
        ++it;
      } else { // not connected
        e = makeDDNode(q, std::array{e, mEdge::zero(), mEdge::zero(), e});
      }
    }
    return e;
  }

  /**
  Creates the DD for a two-qubit gate
  @param mat Matrix representation of the gate
  @param n Number of qubits in the circuit
  @param target0 First target qubit
  @param target1 Second target qubit
  @param start Start index for the DD
  @return DD representing the gate
  @throws std::runtime_error if the number of qubits is larger than the package
  configuration
  **/
  mEdge makeTwoQubitGateDD(const TwoQubitGateMatrix& mat, const std::size_t n,
                           const qc::Qubit target0, const qc::Qubit target1,
                           const std::size_t start = 0) {
    // sanity check
    if (n + start > nqubits) {
      throw std::runtime_error{
          "Requested gate with " + std::to_string(n + start) +
          " qubits, but current package configuration only supports up to " +
          std::to_string(nqubits) +
          " qubits. Please allocate a larger package instance."};
    }

    // create terminal edge matrix
    std::array<std::array<mEdge, NEDGE>, NEDGE> em{};
    for (auto i1 = 0U; i1 < NEDGE; i1++) {
      const auto& matRow = mat.at(i1);
      auto& emRow = em.at(i1);
      for (auto i2 = 0U; i2 < NEDGE; i2++) {
        emRow.at(i2) = mEdge::terminal(cn.lookup(matRow.at(i2)));
      }
    }

    // process lines below smaller target (by creating identity structures)
    auto z = static_cast<Qubit>(start);
    const auto smallerTarget = std::min(target0, target1);
    for (; z < smallerTarget; ++z) {
      for (auto& row : em) {
        for (auto& entry : row) {
          entry = makeDDNode(
              z, std::array{entry, mEdge::zero(), mEdge::zero(), entry});
        }
      }
    }

    // process the smaller target by taking the 16 submatrices and appropriately
    // combining them into four DDs.
    std::array<mEdge, NEDGE> em0{};
    for (std::size_t row = 0; row < RADIX; ++row) {
      for (std::size_t col = 0; col < RADIX; ++col) {
        std::array<mEdge, NEDGE> local{};
        if (target0 > target1) {
          for (std::size_t i = 0; i < RADIX; ++i) {
            for (std::size_t j = 0; j < RADIX; ++j) {
              local.at(i * RADIX + j) =
                  em.at(row * RADIX + i).at(col * RADIX + j);
            }
          }
        } else {
          for (std::size_t i = 0; i < RADIX; ++i) {
            for (std::size_t j = 0; j < RADIX; ++j) {
              local.at(i * RADIX + j) =
                  em.at(i * RADIX + row).at(j * RADIX + col);
            }
          }
        }
        em0.at(row * RADIX + col) = makeDDNode(z, local);
      }
    }

    // process lines between the two targets (by creating identity structures)
    for (++z; z < std::max(target0, target1); ++z) {
      for (auto& entry : em0) {
        entry = makeDDNode(
            z, std::array{entry, mEdge::zero(), mEdge::zero(), entry});
      }
    }

    // process the larger target by combining the four DDs from the smaller
    // target
    auto e = makeDDNode(z, em0);

    // process lines above the larger target (by creating identity structures)
    const auto end = static_cast<Qubit>(n + start);
    for (++z; z < end; ++z) {
      e = makeDDNode(z, std::array{e, mEdge::zero(), mEdge::zero(), e});
    }

    return e;
  }

  mEdge makeSWAPDD(const std::size_t n, const qc::Controls& controls,
                   const qc::Qubit target0, const qc::Qubit target1,
                   const std::size_t start = 0) {
    auto c = controls;
    c.insert(qc::Control{target0});
    mEdge e = makeGateDD(X_MAT, n, c, target1, start);
    c.erase(qc::Control{target0});
    c.insert(qc::Control{target1});
    e = multiply(e, multiply(makeGateDD(X_MAT, n, c, target0, start), e));
    return e;
  }

  mEdge makePeresDD(const std::size_t n, const qc::Controls& controls,
                    const qc::Qubit target0, const qc::Qubit target1,
                    const std::size_t start = 0) {
    auto c = controls;
    c.insert(qc::Control{target1});
    mEdge e = makeGateDD(X_MAT, n, c, target0, start);
    e = multiply(makeGateDD(X_MAT, n, controls, target1, start), e);
    return e;
  }

  mEdge makePeresdagDD(const std::size_t n, const qc::Controls& controls,
                       const qc::Qubit target0, const qc::Qubit target1,
                       const std::size_t start = 0) {
    mEdge e = makeGateDD(X_MAT, n, controls, target1, start);
    auto c = controls;
    c.insert(qc::Control{target1});
    e = multiply(makeGateDD(X_MAT, n, c, target0, start), e);
    return e;
  }

  mEdge makeiSWAPDD(const std::size_t n, const qc::Controls& controls,
                    const qc::Qubit target0, const qc::Qubit target1,
                    const std::size_t start = 0) {
    mEdge e = makeGateDD(S_MAT, n, controls, target1, start);        // S q[1]
    e = multiply(e, makeGateDD(S_MAT, n, controls, target0, start)); // S q[0]
    e = multiply(e, makeGateDD(H_MAT, n, controls, target0, start)); // H q[0]
    auto c = controls;
    c.insert(qc::Control{target0});
    e = multiply(e, makeGateDD(X_MAT, n, c, target1, start)); // CX q[0], q[1]
    c.erase(qc::Control{target0});
    c.insert(qc::Control{target1});
    e = multiply(e, makeGateDD(X_MAT, n, c, target0, start)); // CX q[1], q[0]
    e = multiply(e, makeGateDD(H_MAT, n, controls, target1, start)); // H q[1]
    return e;
  }

  mEdge makeiSWAPinvDD(const std::size_t n, const qc::Controls& controls,
                       const qc::Qubit target0, const qc::Qubit target1,
                       const std::size_t start = 0) {
    mEdge e = makeGateDD(H_MAT, n, controls, target1, start); // H q[1]
    auto c = controls;
    c.insert(qc::Control{target1});
    e = multiply(e, makeGateDD(X_MAT, n, c, target0, start)); // CX q[1], q[0]
    c.erase(qc::Control{target1});
    c.insert(qc::Control{target0});
    e = multiply(e, makeGateDD(X_MAT, n, c, target1, start)); // CX q[0], q[1]
    e = multiply(e, makeGateDD(H_MAT, n, controls, target0, start)); // H q[0]
    e = multiply(e,
                 makeGateDD(SDG_MAT, n, controls, target0, start)); // Sdag q[0]
    e = multiply(e,
                 makeGateDD(SDG_MAT, n, controls, target1, start)); // Sdag q[1]
    return e;
  }

  mEdge makeDCXDD(const std::size_t n, const qc::Controls& controls,
                  const qc::Qubit target0, const qc::Qubit target1,
                  const std::size_t start = 0) {
    auto c = controls;
    c.insert(qc::Control{target0});
    mEdge e = makeGateDD(X_MAT, n, c, target1, start);
    c.erase(qc::Control{target0});
    c.insert(qc::Control{target1});
    e = multiply(e, makeGateDD(X_MAT, n, c, target0, start));
    return e;
  }

  mEdge makeRZZDD(const std::size_t n, const qc::Controls& controls,
                  const qc::Qubit target0, const qc::Qubit target1,
                  const fp theta, const std::size_t start = 0) {
    auto c = controls;
    c.insert(qc::Control{target0});
    auto e = makeGateDD(X_MAT, n, c, target1, start);
    c.erase(qc::Control{target0});
    e = multiply(e, makeGateDD(rzMat(theta), n, c, target1, start));
    c.insert(qc::Control{target0});
    e = multiply(e, makeGateDD(X_MAT, n, c, target1, start));
    return e;
  }

  mEdge makeRYYDD(const std::size_t n, const qc::Controls& controls,
                  const qc::Qubit target0, const qc::Qubit target1,
                  const fp theta, const std::size_t start = 0) {
    // no controls are necessary on the RX gates since they cancel if the
    // controls are 0.
    auto e = makeGateDD(rxMat(PI_2), n, qc::Controls{}, target0, start);
    e = multiply(e, makeGateDD(rxMat(PI_2), n, qc::Controls{}, target1, start));
    e = multiply(e, makeRZZDD(n, controls, target0, target1, theta, start));
    e = multiply(e,
                 makeGateDD(rxMat(-PI_2), n, qc::Controls{}, target1, start));
    e = multiply(e,
                 makeGateDD(rxMat(-PI_2), n, qc::Controls{}, target0, start));
    return e;
  }

  mEdge makeRXXDD(const std::size_t n, const qc::Controls& controls,
                  const qc::Qubit target0, const qc::Qubit target1,
                  const fp theta, const std::size_t start = 0) {
    // no controls are necessary on the H gates since they cancel if the
    // controls are 0.
    auto e = makeGateDD(H_MAT, n, qc::Controls{}, target0, start);
    e = multiply(e, makeGateDD(H_MAT, n, qc::Controls{}, target1, start));
    e = multiply(e, makeRZZDD(n, controls, target0, target1, theta, start));
    e = multiply(e, makeGateDD(H_MAT, n, qc::Controls{}, target1, start));
    e = multiply(e, makeGateDD(H_MAT, n, qc::Controls{}, target0, start));
    return e;
  }

  mEdge makeRZXDD(const std::size_t n, const qc::Controls& controls,
                  const qc::Qubit target0, const qc::Qubit target1,
                  const fp theta, const std::size_t start = 0) {
    // no controls are necessary on the H gates since they cancel if the
    // controls are 0.
    auto e = makeGateDD(H_MAT, n, qc::Controls{}, target1, start);
    e = multiply(e, makeRZZDD(n, controls, target0, target1, theta, start));
    e = multiply(e, makeGateDD(H_MAT, n, qc::Controls{}, target1, start));
    return e;
  }

  mEdge makeECRDD(const std::size_t n, const qc::Controls& controls,
                  const qc::Qubit target0, const qc::Qubit target1,
                  const std::size_t start = 0) {
    auto e = makeRZXDD(n, controls, target0, target1, -PI_4, start);
    e = multiply(e, makeGateDD(X_MAT, n, controls, target0, start));
    e = multiply(e, makeRZXDD(n, controls, target0, target1, PI_4, start));
    return e;
  }

  mEdge makeXXMinusYYDD(const std::size_t n, const qc::Controls& controls,
                        const qc::Qubit target0, const qc::Qubit target1,
                        const fp theta, const fp beta = 0.,
                        const std::size_t start = 0) {
    auto e = makeGateDD(rzMat(-beta), n, qc::Controls{}, target1, start);
    e = multiply(e,
                 makeGateDD(rzMat(-PI_2), n, qc::Controls{}, target0, start));
    e = multiply(e, makeGateDD(SX_MAT, n, qc::Controls{}, target0, start));
    e = multiply(e, makeGateDD(rzMat(PI_2), n, qc::Controls{}, target0, start));
    e = multiply(e, makeGateDD(S_MAT, n, qc::Controls{}, target1, start));
    e = multiply(e, makeGateDD(X_MAT, n, qc::Control{target0}, target1, start));
    // only the following two gates need to be controlled by the controls since
    // the other gates cancel if the controls are 0.
    e = multiply(e,
                 makeGateDD(ryMat(-theta / 2.), n, controls, target0, start));
    e = multiply(e, makeGateDD(ryMat(theta / 2.), n, controls, target1, start));

    e = multiply(e, makeGateDD(X_MAT, n, qc::Control{target0}, target1, start));
    e = multiply(e, makeGateDD(SDG_MAT, n, qc::Controls{}, target1, start));
    e = multiply(e,
                 makeGateDD(rzMat(-PI_2), n, qc::Controls{}, target0, start));
    e = multiply(e, makeGateDD(SXDG_MAT, n, qc::Controls{}, target0, start));
    e = multiply(e, makeGateDD(rzMat(PI_2), n, qc::Controls{}, target0, start));
    e = multiply(e, makeGateDD(rzMat(beta), n, qc::Controls{}, target1, start));
    return e;
  }

  mEdge makeXXPlusYYDD(const std::size_t n, const qc::Controls& controls,
                       const qc::Qubit target0, const qc::Qubit target1,
                       const fp theta, const fp beta = 0.,
                       const std::size_t start = 0) {
    auto e = makeGateDD(rzMat(beta), n, qc::Controls{}, target1, start);
    e = multiply(e,
                 makeGateDD(rzMat(-PI_2), n, qc::Controls{}, target0, start));
    e = multiply(e, makeGateDD(SX_MAT, n, qc::Controls{}, target0, start));
    e = multiply(e, makeGateDD(rzMat(PI_2), n, qc::Controls{}, target0, start));
    e = multiply(e, makeGateDD(S_MAT, n, qc::Controls{}, target1, start));
    e = multiply(e, makeGateDD(X_MAT, n, qc::Control{target0}, target1, start));
    // only the following two gates need to be controlled by the controls since
    // the other gates cancel if the controls are 0.
    e = multiply(e, makeGateDD(ryMat(theta / 2.), n, controls, target0, start));
    e = multiply(e, makeGateDD(ryMat(theta / 2.), n, controls, target1, start));

    e = multiply(e, makeGateDD(X_MAT, n, qc::Control{target0}, target1, start));
    e = multiply(e, makeGateDD(SDG_MAT, n, qc::Controls{}, target1, start));
    e = multiply(e,
                 makeGateDD(rzMat(-PI_2), n, qc::Controls{}, target0, start));
    e = multiply(e, makeGateDD(SXDG_MAT, n, qc::Controls{}, target0, start));
    e = multiply(e, makeGateDD(rzMat(PI_2), n, qc::Controls{}, target0, start));
    e = multiply(e,
                 makeGateDD(rzMat(-beta), n, qc::Controls{}, target1, start));
    return e;
  }

private:
  // check whether node represents a symmetric matrix or the identity
  void checkSpecialMatrices(mNode* p) {
    if (mNode::isTerminal(p)) {
      return;
    }

    // check if matrix resembles identity
    p->setIdentity(false);
    const auto& e0 = p->e[0];
    const auto& e1 = p->e[1];
    const auto& e2 = p->e[2];
    const auto& e3 = p->e[3];
    if (!mNode::isIdentity(e0.p) || !e1.w.exactlyZero() ||
        !e2.w.exactlyZero() || !e0.w.exactlyOne() || !e3.w.exactlyOne() ||
        !mNode::isIdentity(e3.p)) {
      return;
    }
    p->setIdentity(true);
  }

  vEdge makeStateFromVector(const CVec::const_iterator& begin,
                            const CVec::const_iterator& end,
                            const Qubit level) {
    if (level == 0U) {
      assert(std::distance(begin, end) == 2);
      const auto& zeroWeight = cn.getCached(*begin);
      const auto& oneWeight = cn.getCached(*std::next(begin));
      const auto zeroSuccessor = vEdge{vNode::getTerminal(), zeroWeight};
      const auto oneSuccessor = vEdge{vNode::getTerminal(), oneWeight};
      return makeDDNode<vNode>(0, {zeroSuccessor, oneSuccessor}, true);
    }

    const auto half = std::distance(begin, end) / 2;
    const auto zeroSuccessor =
        makeStateFromVector(begin, begin + half, level - 1);
    const auto oneSuccessor = makeStateFromVector(begin + half, end, level - 1);
    return makeDDNode<vNode>(level, {zeroSuccessor, oneSuccessor}, true);
  }

  /**
  Constructs a decision diagram (DD) from a complex matrix using a recursive
  algorithm.
  @param matrix The complex matrix from which to create the DD.
  @param level The current level of recursion. Starts at the highest level of
  the matrix (log base 2 of the matrix size - 1).
  @param rowStart The starting row of the quadrant being processed.
  @param rowEnd The ending row of the quadrant being processed.
  @param colStart The starting column of the quadrant being processed.
  @param colEnd The ending column of the quadrant being processed.
  @return An mEdge representing the root node of the created DD.
  @details This function recursively breaks down the matrix into quadrants until
  each quadrant has only one element. At each level of recursion, four new edges
  are created, one for each quadrant of the matrix. The four resulting decision
  diagram edges are used to create a new decision diagram node at the current
  level, and this node is returned as the result of the current recursive call.
  At the base case of recursion, the matrix has only one element, which is
  converted into a terminal node of the decision diagram.
  @note This function assumes that the matrix size is a power of two.
  **/
  mEdge makeDDFromMatrix(const CMat& matrix, const Qubit level,
                         const std::size_t rowStart, const std::size_t rowEnd,
                         const std::size_t colStart, const std::size_t colEnd) {
    // base case
    if (level == 0U) {
      assert(rowEnd - rowStart == 2);
      assert(colEnd - colStart == 2);
      const auto w0 = cn.getCached(matrix[rowStart][colStart]);
      const auto e0 = mEdge{mNode::getTerminal(), w0};
      const auto w1 = cn.getCached(matrix[rowStart][colStart + 1]);
      const auto e1 = mEdge{mNode::getTerminal(), w1};
      const auto w2 = cn.getCached(matrix[rowStart + 1][colStart]);
      const auto e2 = mEdge{mNode::getTerminal(), w2};
      const auto w3 = cn.getCached(matrix[rowStart + 1][colStart + 1]);
      const auto e3 = mEdge{mNode::getTerminal(), w3};
      return makeDDNode<mNode>(0U, {e0, e1, e2, e3}, true);
    }

    // recursively call the function on all quadrants
    const auto rowMid = (rowStart + rowEnd) / 2;
    const auto colMid = (colStart + colEnd) / 2;
    const auto l = static_cast<Qubit>(level - 1U);

    return makeDDNode<mNode>(
        level,
        {makeDDFromMatrix(matrix, l, rowStart, rowMid, colStart, colMid),
         makeDDFromMatrix(matrix, l, rowStart, rowMid, colMid, colEnd),
         makeDDFromMatrix(matrix, l, rowMid, rowEnd, colStart, colMid),
         makeDDFromMatrix(matrix, l, rowMid, rowEnd, colMid, colEnd)},
        true);
  }

public:
  // create a normalized DD node and return an edge pointing to it. The node is
  // not recreated if it already exists.
  template <class Node>
  Edge<Node> makeDDNode(
      const Qubit var,
      const std::array<Edge<Node>, std::tuple_size_v<decltype(Node::e)>>& edges,
      const bool cached = false,
      [[maybe_unused]] const bool generateDensityMatrix = false) {
    auto& memoryManager = getMemoryManager<Node>();
    Edge<Node> e{memoryManager.get(), Complex::one()};
    e.p->v = var;
    e.p->e = edges;

    if constexpr (std::is_same_v<Node, mNode> || std::is_same_v<Node, dNode>) {
      e.p->flags = 0;
      if constexpr (std::is_same_v<Node, dNode>) {
        e.p->setDensityMatrixNodeFlag(generateDensityMatrix);
      }
    }

    assert(e.p->ref == 0);
    for ([[maybe_unused]] const auto& edge : edges) {
      // an error here indicates that cached nodes are assigned multiple times.
      // Check if garbage collect correctly resets the cache tables!
      assert(edge.isTerminal() || edge.p->v == var - 1);
    }

    // normalize it
    e = normalize(e, cached);
    assert(e.isTerminal() || e.p->v == var);

    // look it up in the unique tables
    auto& uniqueTable = getUniqueTable<Node>();
    auto* l = uniqueTable.lookup(e.p);

    // set specific node properties for matrices
    if constexpr (std::is_same_v<Node, mNode>) {
      if (l == e.p) {
        checkSpecialMatrices(l);
      }
    }
    return {l, e.w};
  }

  template <class Node>
  Edge<Node> deleteEdge(const Edge<Node>& e, const Qubit v,
                        const std::size_t edgeIdx) {
    std::unordered_map<Node*, Edge<Node>> nodes{};
    return deleteEdge(e, v, edgeIdx, nodes);
  }

private:
  template <class Node>
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
    r.w = cn.lookup(r.w * e.w);
    return r;
  }

  ///
  /// Compute table definitions
  ///
public:
  void clearComputeTables() {
    vectorAdd.clear();
    matrixAdd.clear();
    conjugateMatrixTranspose.clear();
    matrixMatrixMultiplication.clear();
    matrixVectorMultiplication.clear();
    vectorInnerProduct.clear();
    vectorKronecker.clear();
    matrixKronecker.clear();

    clearIdentityTable();

    stochasticNoiseOperationCache.clear();
    densityAdd.clear();
    densityDensityMultiplication.clear();
    densityNoise.clear();
  }

  ///
  /// Measurements from state decision diagrams
  ///
  std::string measureAll(vEdge& rootEdge, const bool collapse,
                         std::mt19937_64& mt, const fp epsilon = 0.001) {
    if (std::abs(ComplexNumbers::mag2(rootEdge.w) - 1.0) > epsilon) {
      if (rootEdge.w.approximatelyZero()) {
        throw std::runtime_error(
            "Numerical instabilities led to a 0-vector! Abort simulation!");
      }
      std::cerr << "WARNING in MAll: numerical instability occurred during "
                   "simulation: |alpha|^2 + |beta|^2 = "
                << ComplexNumbers::mag2(rootEdge.w) << ", but should be 1!\n";
    }

    if (rootEdge.isTerminal()) {
      return "";
    }

    vEdge cur = rootEdge;
    const auto numberOfQubits = static_cast<std::size_t>(rootEdge.p->v) + 1U;

    std::string result(numberOfQubits, '0');

    std::uniform_real_distribution<fp> dist(0.0, 1.0L);

    for (auto i = numberOfQubits; i > 0; --i) {
      fp p0 = ComplexNumbers::mag2(cur.p->e.at(0).w);
      const fp p1 = ComplexNumbers::mag2(cur.p->e.at(1).w);
      const fp tmp = p0 + p1;

      if (std::abs(tmp - 1.0) > epsilon) {
        throw std::runtime_error("Added probabilities differ from 1 by " +
                                 std::to_string(std::abs(tmp - 1.0)));
      }
      p0 /= tmp;

      const fp threshold = dist(mt);
      if (threshold < p0) {
        cur = cur.p->e.at(0);
      } else {
        result[cur.p->v] = '1';
        cur = cur.p->e.at(1);
      }
    }

    if (collapse) {
      vEdge e = vEdge::one();
      std::array<vEdge, 2> edges{};
      for (std::size_t p = 0U; p < numberOfQubits; ++p) {
        if (result[p] == '0') {
          edges[0] = e;
          edges[1] = vEdge::zero();
        } else {
          edges[0] = vEdge::zero();
          edges[1] = e;
        }
        e = makeDDNode(static_cast<Qubit>(p), edges, false);
      }
      incRef(e);
      decRef(rootEdge);
      rootEdge = e;
    }

    return std::string{result.rbegin(), result.rend()};
  }

private:
  fp assignProbabilities(const vEdge& edge,
                         std::unordered_map<const vNode*, fp>& probs) {
    auto it = probs.find(edge.p);
    if (it != probs.end()) {
      return ComplexNumbers::mag2(edge.w) * it->second;
    }
    double sum{1};
    if (!edge.isTerminal()) {
      sum = assignProbabilities(edge.p->e[0], probs) +
            assignProbabilities(edge.p->e[1], probs);
    }

    probs.insert({edge.p, sum});

    return ComplexNumbers::mag2(edge.w) * sum;
  }

public:
  std::pair<dd::fp, dd::fp>
  determineMeasurementProbabilities(const vEdge& rootEdge, const Qubit index,
                                    const bool assumeProbabilityNormalization) {
    std::map<const vNode*, fp> probsMone;
    std::set<const vNode*> visited;
    std::queue<const vNode*> q;

    probsMone[rootEdge.p] = ComplexNumbers::mag2(rootEdge.w);
    visited.insert(rootEdge.p);
    q.push(rootEdge.p);

    while (q.front()->v != index) {
      const auto* ptr = q.front();
      q.pop();
      const fp prob = probsMone[ptr];

      const auto& s0 = ptr->e[0];
      if (const auto s0w = static_cast<ComplexValue>(s0.w);
          !s0w.approximatelyZero()) {
        const fp tmp1 = prob * s0w.mag2();
        if (visited.find(s0.p) != visited.end()) {
          probsMone[s0.p] = probsMone[s0.p] + tmp1;
        } else {
          probsMone[s0.p] = tmp1;
          visited.insert(s0.p);
          q.push(s0.p);
        }
      }

      const auto& s1 = ptr->e[1];
      if (const auto s1w = static_cast<ComplexValue>(s1.w);
          !s1w.approximatelyZero()) {
        const fp tmp1 = prob * s1w.mag2();
        if (visited.find(s1.p) != visited.end()) {
          probsMone[s1.p] = probsMone[s1.p] + tmp1;
        } else {
          probsMone[s1.p] = tmp1;
          visited.insert(s1.p);
          q.push(s1.p);
        }
      }
    }

    fp pzero{0};
    fp pone{0};

    if (assumeProbabilityNormalization) {
      while (!q.empty()) {
        const auto* ptr = q.front();
        q.pop();
        const auto& s0 = ptr->e[0];
        if (const auto s0w = static_cast<ComplexValue>(s0.w);
            !s0w.approximatelyZero()) {
          pzero += probsMone[ptr] * s0w.mag2();
        }
        const auto& s1 = ptr->e[1];
        if (const auto s1w = static_cast<ComplexValue>(s1.w);
            !s1w.approximatelyZero()) {
          pone += probsMone[ptr] * s1w.mag2();
        }
      }
    } else {
      std::unordered_map<const vNode*, fp> probs;
      assignProbabilities(rootEdge, probs);

      while (!q.empty()) {
        const auto* ptr = q.front();
        q.pop();

        const auto& s0 = ptr->e[0];
        if (const auto s0w = static_cast<ComplexValue>(s0.w);
            !s0w.approximatelyZero()) {
          pzero += probsMone[ptr] * probs[s0.p] * s0w.mag2();
        }
        const auto& s1 = ptr->e[1];
        if (const auto s1w = static_cast<ComplexValue>(s1.w);
            !s1w.approximatelyZero()) {
          pone += probsMone[ptr] * probs[s1.p] * s1w.mag2();
        }
      }
    }
    return {pzero, pone};
  }

  /**
   * @brief Measures the qubit with the given index in the given state vector
   * decision diagram. Collapses the state according to the measurement result.
   * @param rootEdge the root edge of the state vector decision diagram
   * @param index the index of the qubit to be measured
   * @param assumeProbabilityNormalization whether or not to assume that the
   * state vector decision diagram has normalized edge weights.
   * @param mt the random number generator
   * @param epsilon the numerical precision used for checking the normalization
   * of the state vector decision diagram
   * @return the measurement result ('0' or '1')
   * @throws std::runtime_error if a numerical instability is detected during
   * the measurement.
   */
  char measureOneCollapsing(vEdge& rootEdge, const Qubit index,
                            const bool assumeProbabilityNormalization,
                            std::mt19937_64& mt, const fp epsilon = 0.001) {
    const auto& [pzero, pone] = determineMeasurementProbabilities(
        rootEdge, index, assumeProbabilityNormalization);
    const fp sum = pzero + pone;
    if (std::abs(sum - 1) > epsilon) {
      throw std::runtime_error(
          "Numerical instability occurred during measurement: |alpha|^2 + "
          "|beta|^2 = " +
          std::to_string(pzero) + " + " + std::to_string(pone) + " = " +
          std::to_string(pzero + pone) + ", but should be 1!");
    }
    std::uniform_real_distribution<fp> dist(0., 1.);
    if (const auto threshold = dist(mt); threshold < pzero / sum) {
      performCollapsingMeasurement(rootEdge, index, pzero, true);
      return '0';
    }
    performCollapsingMeasurement(rootEdge, index, pone, false);
    return '1';
  }

  /**
   * @brief Performs a specific measurement on the given state vector decision
   * diagram. Collapses the state according to the measurement result.
   * @param rootEdge the root edge of the state vector decision diagram
   * @param index the index of the qubit to be measured
   * @param probability the probability of the measurement result (required for
   * normalization)
   * @param measureZero whether or not to measure '0' (otherwise '1' is
   * measured)
   */
  void performCollapsingMeasurement(vEdge& rootEdge, const Qubit index,
                                    const fp probability,
                                    const bool measureZero) {
    GateMatrix measurementMatrix = measureZero ? MEAS_ZERO_MAT : MEAS_ONE_MAT;

    const auto measurementGate =
        makeGateDD(measurementMatrix, rootEdge.p->v + 1U, index);

    vEdge e = multiply(measurementGate, rootEdge);

    assert(probability > 0.);
    e.w = cn.lookup(e.w / std::sqrt(probability));
    incRef(e);
    decRef(rootEdge);
    rootEdge = e;
  }

  ///
  /// Addition
  ///
  ComputeTable<vCachedEdge, vCachedEdge, vCachedEdge,
               Config::CT_VEC_ADD_NBUCKET>
      vectorAdd{};
  ComputeTable<mCachedEdge, mCachedEdge, mCachedEdge,
               Config::CT_MAT_ADD_NBUCKET>
      matrixAdd{};
  ComputeTable<dCachedEdge, dCachedEdge, dCachedEdge, Config::CT_DM_ADD_NBUCKET>
      densityAdd{};

  template <class Node> [[nodiscard]] auto& getAddComputeTable() {
    if constexpr (std::is_same_v<Node, vNode>) {
      return vectorAdd;
    } else if constexpr (std::is_same_v<Node, mNode>) {
      return matrixAdd;
    } else if constexpr (std::is_same_v<Node, dNode>) {
      return densityAdd;
    }
  }

  template <class Node>
  Edge<Node> add(const Edge<Node>& x, const Edge<Node>& y) {
    [[maybe_unused]] const auto before = cn.cacheCount();

    Qubit var{};
    if (!x.isTerminal()) {
      var = x.p->v;
    }
    if (!y.isTerminal() && (y.p->v) > var) {
      var = y.p->v;
    }

    auto result = add2(x, y, var);
    result.w = cn.lookup(result.w, true);

    [[maybe_unused]] const auto after = cn.cacheCount();
    assert(after == before);

    return result;
  }

  template <class Node>
  Edge<Node> add2(const Edge<Node>& x, const Edge<Node>& y, const Qubit var) {
    if (x.w.exactlyZero()) {
      if (y.w.exactlyZero()) {
        return Edge<Node>::zero();
      }
      return {y.p, cn.getCached(y.w)};
    }
    if (y.w.exactlyZero()) {
      return {x.p, cn.getCached(x.w)};
    }
    if (x.p == y.p) {
      auto r = y;
      r.w = cn.addCached(x.w, y.w);
      if (r.w.approximatelyZero()) {
        cn.returnToCache(r.w);
        return Edge<Node>::zero();
      }
      return r;
    }

    auto& computeTable = getAddComputeTable<Node>();
    if (const auto* r = computeTable.lookup({x.p, x.w}, {y.p, y.w});
        r != nullptr) {
      if (r->w.approximatelyZero()) {
        return Edge<Node>::zero();
      }
      return {r->p, cn.getCached(r->w)};
    }

    constexpr std::size_t n = std::tuple_size_v<decltype(x.p->e)>;
    std::array<Edge<Node>, n> edge{};
    for (std::size_t i = 0U; i < n; i++) {
      Edge<Node> e1{};
      if (!x.isTerminal()) {
        e1 = x.p->e[i];

        if (!e1.w.exactlyZero()) {
          e1.w = cn.mulCached(e1.w, x.w);
        }
      } else {
        e1 = x;
        if (y.p->e[i].isTerminal()) {
          e1 = Edge<Node>::zero();
        }
      }
      Edge<Node> e2{};
      if (!y.isTerminal()) {
        e2 = y.p->e[i];

        if (!e2.w.exactlyZero()) {
          e2.w = cn.mulCached(e2.w, y.w);
        }
      } else {
        e2 = y;
        if (x.p->e[i].isTerminal()) {
          e2 = Edge<Node>::zero();
        }
      }

      if constexpr (std::is_same_v<Node, dNode>) {
        dEdge::applyDmChangesToEdges(e1, e2);
        edge[i] = add2(e1, e2, var - 1);
        dEdge::revertDmChangesToEdges(e1, e2);
      } else {
        edge[i] = add2(e1, e2, var - 1);
      }

      if (!x.isTerminal() && x.p->v == var) {
        cn.returnToCache(e1.w);
      }

      if (!y.isTerminal() && y.p->v == var) {
        cn.returnToCache(e2.w);
      }
    }

    auto e = makeDDNode(var, edge, true);

    computeTable.insert({x.p, x.w}, {y.p, y.w}, {e.p, e.w});
    return e;
  }

  ///
  /// Matrix (conjugate) transpose
  ///
  UnaryComputeTable<mEdge, mEdge, Config::CT_MAT_CONJ_TRANS_NBUCKET>
      conjugateMatrixTranspose{};

  mEdge conjugateTranspose(const mEdge& a) {
    if (a.isTerminal()) { // terminal case
      return {a.p, ComplexNumbers::conj(a.w)};
    }

    // check if in compute table
    if (const auto* r = conjugateMatrixTranspose.lookup(a); r != nullptr) {
      return *r;
    }

    std::array<mEdge, NEDGE> e{};
    // conjugate transpose submatrices and rearrange as required
    for (auto i = 0U; i < RADIX; ++i) {
      for (auto j = 0U; j < RADIX; ++j) {
        e[RADIX * i + j] = conjugateTranspose(a.p->e[RADIX * j + i]);
      }
    }
    // create new top node
    auto res = makeDDNode(a.p->v, e);

    // adjust top weight including conjugate
    res.w = cn.lookup(res.w * ComplexNumbers::conj(a.w));

    // put it in the compute table
    conjugateMatrixTranspose.insert(a, res);
    return res;
  }

  ///
  /// Multiplication
  ///
  ComputeTable<mEdge, vEdge, vCachedEdge, Config::CT_MAT_VEC_MULT_NBUCKET>
      matrixVectorMultiplication{};
  ComputeTable<mEdge, mEdge, mCachedEdge, Config::CT_MAT_MAT_MULT_NBUCKET>
      matrixMatrixMultiplication{};
  ComputeTable<dEdge, dEdge, dCachedEdge, Config::CT_DM_DM_MULT_NBUCKET>
      densityDensityMultiplication{};

  template <class RightOperandNode>
  [[nodiscard]] auto& getMultiplicationComputeTable() {
    if constexpr (std::is_same_v<RightOperandNode, vNode>) {
      return matrixVectorMultiplication;
    } else if constexpr (std::is_same_v<RightOperandNode, mNode>) {
      return matrixMatrixMultiplication;
    } else if constexpr (std::is_same_v<RightOperandNode, dNode>) {
      return densityDensityMultiplication;
    }
  }

  dEdge applyOperationToDensity(dEdge& e, const mEdge& operation,
                                const bool generateDensityMatrix = false) {
    [[maybe_unused]] const auto before = cn.cacheCount();
    auto tmp0 = conjugateTranspose(operation);
    auto tmp1 = multiply(e, densityFromMatrixEdge(tmp0), 0, false);
    auto tmp2 = multiply(densityFromMatrixEdge(operation), tmp1, 0,
                         generateDensityMatrix);
    incRef(tmp2);
    dEdge::alignDensityEdge(e);
    decRef(e);
    e = tmp2;

    if (generateDensityMatrix) {
      dEdge::setDensityMatrixTrue(e);
    }

    return e;
  }

  template <class LeftOperandNode, class RightOperandNode>
  Edge<RightOperandNode>
  multiply(const Edge<LeftOperandNode>& x, const Edge<RightOperandNode>& y,
           const Qubit start = 0,
           [[maybe_unused]] const bool generateDensityMatrix = false) {
    using LEdge = Edge<LeftOperandNode>;
    using REdge = Edge<RightOperandNode>;
    static_assert(std::disjunction_v<std::is_same<LEdge, mEdge>,
                                     std::is_same<LEdge, dEdge>>,
                  "Left operand must be a matrix or density matrix");
    static_assert(std::disjunction_v<std::is_same<REdge, vEdge>,
                                     std::is_same<REdge, mEdge>,
                                     std::is_same<REdge, dEdge>>,
                  "Right operand must be a vector, matrix or density matrix");

    [[maybe_unused]] const auto before = cn.cacheCount();

    Qubit var{};
    REdge e;
    if constexpr (std::is_same_v<LEdge, dEdge>) {
      auto xCopy = x;
      auto yCopy = y;
      dEdge::applyDmChangesToEdges(xCopy, yCopy);

      if (!xCopy.isTerminal()) {
        var = xCopy.p->v;
      }
      if (!y.isTerminal() && yCopy.p->v > var) {
        var = yCopy.p->v;
      }

      e = multiply2(xCopy, yCopy, var, start, generateDensityMatrix);
      dEdge::revertDmChangesToEdges(xCopy, yCopy);
    } else {
      if (!x.isTerminal()) {
        var = x.p->v;
      }
      if (!y.isTerminal() && y.p->v > var) {
        var = y.p->v;
      }
      e = multiply2(x, y, var, start);
    }

    e.w = cn.lookup(e.w, true);

    [[maybe_unused]] const auto after = cn.cacheCount();
    assert(before == after);

    return e;
  }

private:
  template <class LeftOperandNode, class RightOperandNode>
  Edge<RightOperandNode>
  multiply2(const Edge<LeftOperandNode>& x, const Edge<RightOperandNode>& y,
            const Qubit var, const Qubit start = 0,
            [[maybe_unused]] const bool generateDensityMatrix = false) {
    using LEdge = Edge<LeftOperandNode>;
    using REdge = Edge<RightOperandNode>;
    using ResultEdge = Edge<RightOperandNode>;

    if (x.w.exactlyZero() || y.w.exactlyZero()) {
      return ResultEdge::zero();
    }

    if (x.isIdentity()) {
      return {y.p, cn.mulCached(x.w, y.w)};
    }

    if constexpr (std::is_same_v<RightOperandNode, mNode> ||
                  std::is_same_v<RightOperandNode, dNode>) {
      if (y.isIdentity()) {
        return {x.p, cn.mulCached(x.w, y.w)};
      }
    }
    assert(x.p != nullptr);
    assert(y.p != nullptr);

    auto xCopy = LEdge{x.p, Complex::one()};
    auto yCopy = REdge{y.p, Complex::one()};

    auto& computeTable = getMultiplicationComputeTable<RightOperandNode>();
    if (const auto* r =
            computeTable.lookup(xCopy, yCopy, generateDensityMatrix);
        r != nullptr) {
      if (r->w.approximatelyZero()) {
        return ResultEdge::zero();
      }
      auto e = ResultEdge{r->p, cn.getCached(r->w)};
      ComplexNumbers::mul(e.w, e.w, x.w);
      ComplexNumbers::mul(e.w, e.w, y.w);
      if (e.w.approximatelyZero()) {
        cn.returnToCache(e.w);
        return ResultEdge::zero();
      }
      return e;
    }

    constexpr std::size_t n = std::tuple_size_v<decltype(y.p->e)>;

    constexpr std::size_t rows = RADIX;
    constexpr std::size_t cols = n == NEDGE ? RADIX : 1U;

    std::array<ResultEdge, n> edge{};
    for (auto i = 0U; i < rows; i++) {
      for (auto j = 0U; j < cols; j++) {
        auto idx = cols * i + j;
        edge[idx] = ResultEdge::zero();
        for (auto k = 0U; k < rows; k++) {
          const auto xIdx = rows * i + k;
          LEdge e1 = x.p->e[xIdx];

          const auto yIdx = j + cols * k;
          REdge e2 = y.p->e[yIdx];

          const auto v = static_cast<Qubit>(var - 1);
          if constexpr (std::is_same_v<LeftOperandNode, dNode>) {
            dEdge m;
            dEdge::applyDmChangesToEdges(e1, e2);
            if (!generateDensityMatrix || idx == 1) {
              // When generateDensityMatrix is false or I have the first edge I
              // don't optimize anything and set generateDensityMatrix to false
              // for all child edges
              m = multiply2(e1, e2, v, start, false);
            } else if (idx == 2) {
              // When I have the second edge and generateDensityMatrix == false,
              // then edge[2] == edge[1]
              if (k == 0) {
                if (edge[1].w.approximatelyZero()) {
                  edge[2] = ResultEdge::zero();
                } else {
                  edge[2] = {edge[1].p, cn.getCached(edge[1].w)};
                }
              }
              continue;
            } else {
              m = multiply2(e1, e2, v, start, generateDensityMatrix);
            }

            if (k == 0 || edge[idx].w.exactlyZero()) {
              edge[idx] = m;
            } else if (!m.w.exactlyZero()) {
              dEdge::applyDmChangesToEdges(edge[idx], m);
              const auto w = edge[idx].w;
              edge[idx] = add2(edge[idx], m, v);
              dEdge::revertDmChangesToEdges(edge[idx], e2);
              cn.returnToCache(w);
              cn.returnToCache(m.w);
            }
            // Undo modifications on density matrices
            dEdge::revertDmChangesToEdges(e1, e2);
          } else {
            auto m = multiply2(e1, e2, v, start);

            if (k == 0 || edge[idx].w.exactlyZero()) {
              edge[idx] = m;
            } else if (!m.w.exactlyZero()) {
              const auto w = edge[idx].w;
              edge[idx] = add2(edge[idx], m, v);
              cn.returnToCache(w);
              cn.returnToCache(m.w);
            }
          }
        }
      }
    }

    auto e = makeDDNode(var, edge, true, generateDensityMatrix);
    computeTable.insert(xCopy, yCopy, {e.p, e.w});

    if (!e.w.exactlyZero()) {
      if (e.w.exactlyOne()) {
        e.w = cn.mulCached(x.w, y.w);
      } else {
        ComplexNumbers::mul(e.w, e.w, x.w);
        ComplexNumbers::mul(e.w, e.w, y.w);
      }
      if (e.w.approximatelyZero()) {
        cn.returnToCache(e.w);
        return ResultEdge::zero();
      }
    }
    return e;
  }

  ///
  /// Inner product, fidelity, expectation value
  ///
public:
  ComputeTable<vEdge, vEdge, vCachedEdge, Config::CT_VEC_INNER_PROD_NBUCKET>
      vectorInnerProduct{};

  /**
      Calculates the inner product of two vector decision diagrams x and y.
      @param x a vector DD representing a quantum state
      @param y a vector DD representing a quantum state
      @return a complex number representing the scalar product of the DDs
  **/
  ComplexValue innerProduct(const vEdge& x, const vEdge& y) {
    if (x.isTerminal() || y.isTerminal() || x.w.approximatelyZero() ||
        y.w.approximatelyZero()) { // the 0 case
      return {0, 0};
    }

    [[maybe_unused]] const auto before = cn.cacheCount();

    auto w = x.p->v;
    if (y.p->v > w) {
      w = y.p->v;
    }
    // Overall normalization factor needs to be conjugated
    // before input into recursive private function
    auto xCopy = vEdge{x.p, ComplexNumbers::conj(x.w)};
    const ComplexValue ip = innerProduct(xCopy, y, w + 1U);

    [[maybe_unused]] const auto after = cn.cacheCount();
    assert(after == before);

    return ip;
  }

  fp fidelity(const vEdge& x, const vEdge& y) {
    const auto fid = innerProduct(x, y);
    return fid.r * fid.r + fid.i * fid.i;
  }

  fp fidelityOfMeasurementOutcomes(const vEdge& e, const SparsePVec& probs) {
    if (e.w.approximatelyZero()) {
      return 0.;
    }
    return fidelityOfMeasurementOutcomesRecursive(e, probs, 0);
  }

  fp fidelityOfMeasurementOutcomesRecursive(const vEdge& e,
                                            const SparsePVec& probs,
                                            const std::size_t i) {
    const auto top = ComplexNumbers::mag(e.w);
    if (e.isTerminal()) {
      if (auto it = probs.find(i); it != probs.end()) {
        return top * std::sqrt(it->second);
      }
      return 0.;
    }

    std::size_t leftIdx = i;
    fp leftContribution = 0.;
    if (!e.p->e[0].w.approximatelyZero()) {
      leftContribution =
          fidelityOfMeasurementOutcomesRecursive(e.p->e[0], probs, leftIdx);
    }

    std::size_t rightIdx = i | (1ULL << e.p->v);
    auto rightContribution = 0.;
    if (!e.p->e[1].w.approximatelyZero()) {
      rightContribution =
          fidelityOfMeasurementOutcomesRecursive(e.p->e[1], probs, rightIdx);
    }

    return top * (leftContribution + rightContribution);
  }

private:
  /**
      Private function to recursively calculate the inner product of two vector
  decision diagrams x and y with var levels.
      @param x a vector DD representing a quantum state
      @param y a vector DD representing a quantum state
      @param var the number of levels contained in each vector DD
      @return a complex number  representing the scalar product of the DDs
      @note This function is called recursively such that the number of levels
  decreases each time to traverse the DDs.
  **/
  ComplexValue innerProduct(const vEdge& x, const vEdge& y, Qubit var) {
    const auto xWeight = static_cast<ComplexValue>(x.w);
    if (xWeight.approximatelyZero()) {
      return 0;
    }
    const auto yWeight = static_cast<ComplexValue>(y.w);
    if (yWeight.approximatelyZero()) {
      return 0;
    }

    const auto rWeight = xWeight * yWeight;
    if (var == 0) { // Multiplies terminal weights
      return rWeight;
    }

    // Set to one to generate more lookup hits
    auto xCopy = vEdge{x.p, Complex::one()};
    auto yCopy = vEdge{y.p, Complex::one()};
    if (const auto* r = vectorInnerProduct.lookup(xCopy, yCopy); r != nullptr) {
      return r->w * rWeight;
    }

    auto w = static_cast<Qubit>(var - 1U);
    ComplexValue sum = 0;
    // Iterates through edge weights recursively until terminal
    for (auto i = 0U; i < RADIX; i++) {
      vEdge e1{};
      if (!x.isTerminal() && x.p->v == w) {
        e1 = x.p->e[i];
        e1.w = ComplexNumbers::conj(e1.w);
      } else {
        e1 = xCopy;
      }
      vEdge e2{};
      if (!y.isTerminal() && y.p->v == w) {
        e2 = y.p->e[i];
      } else {
        e2 = yCopy;
      }
      sum += innerProduct(e1, e2, w);
    }

    vectorInnerProduct.insert(xCopy, yCopy, vCachedEdge ::terminal(sum));
    return sum * rWeight;
  }

public:
  /**
      Calculates the expectation value of an operator x with respect to a
  quantum state y given their corresponding decision diagrams.
      @param x a matrix DD representing the operator
      @param y a vector DD representing the quantum state
      @return a floating point value representing the expectation value of the
  operator with respect to the quantum state
      @throw an exception message is thrown if the edges are not on the same
  level or if the expectation value is non-real.
      @note This function calls the multiply() function to apply the operator to
  the quantum state, then calls innerProduct() to calculate the overlap between
  the original state and the applied state i.e. <Psi| Psi'> = <Psi| (Op|Psi>).
            It also calls the garbageCollect() function to free up any unused
  memory.
  **/
  fp expectationValue(const mEdge& x, const vEdge& y) {
    assert(!x.isTerminal() && !y.isTerminal());
    if (x.p->v != y.p->v) {
      throw std::invalid_argument(
          "Observable and state must act on the same number of qubits to "
          "compute the expectation value.");
    }

    auto yPrime = multiply(x, y);
    const ComplexValue expValue = innerProduct(y, yPrime);

    assert(RealNumber::approximatelyZero(expValue.i));
    return expValue.r;
  }

  ///
  /// Kronecker/tensor product
  ///

  ComputeTable<vEdge, vEdge, vCachedEdge, Config::CT_VEC_KRON_NBUCKET>
      vectorKronecker{};
  ComputeTable<mEdge, mEdge, mCachedEdge, Config::CT_MAT_KRON_NBUCKET>
      matrixKronecker{};

  template <class Node> [[nodiscard]] auto& getKroneckerComputeTable() {
    if constexpr (std::is_same_v<Node, vNode>) {
      return vectorKronecker;
    } else {
      return matrixKronecker;
    }
  }

  template <class Node>
  Edge<Node> kronecker(const Edge<Node>& x, const Edge<Node>& y,
                       const bool incIdx = true) {
    if constexpr (std::is_same_v<Node, dNode>) {
      throw std::invalid_argument(
          "Kronecker is currently not supported for density matrices");
    }

    auto e = kronecker2(x, y, incIdx);
    return {e.p, cn.lookup(e.w, true)};
  }

  // extent the DD pointed to by `e` with `h` identities on top and `l`
  // identities at the bottom
  mEdge extend(const mEdge& e, const Qubit h, const Qubit l = 0) {
    auto f = (l > 0) ? kronecker(e, makeIdent(l)) : e;
    auto g = (h > 0) ? kronecker(makeIdent(h), f) : f;
    return g;
  }

private:
  template <class Node>
  Edge<Node> kronecker2(const Edge<Node>& x, const Edge<Node>& y,
                        const bool incIdx = true) {
    if (x.w.exactlyZero() || y.w.exactlyZero()) {
      return Edge<Node>::zero();
    }
    const auto xWeight = static_cast<ComplexValue>(x.w);
    if (xWeight.approximatelyZero()) {
      return Edge<Node>::zero();
    }
    const auto yWeight = static_cast<ComplexValue>(y.w);
    if (yWeight.approximatelyZero()) {
      return Edge<Node>::zero();
    }
    const auto rWeight = xWeight * yWeight;
    if (rWeight.approximatelyZero()) {
      return Edge<Node>::zero();
    }

    if (x.isTerminal()) {
      return {y.p, cn.getCached(rWeight)};
    }

    auto xCopy = Edge<Node>{x.p, Complex::one()};
    auto yCopy = Edge<Node>{y.p, Complex::one()};
    auto& computeTable = getKroneckerComputeTable<Node>();
    if (const auto* r = computeTable.lookup(xCopy, yCopy); r != nullptr) {
      return {r->p, cn.getCached(rWeight)};
    }

    constexpr std::size_t n = std::tuple_size_v<decltype(x.p->e)>;
    // special case handling for matrices
    if constexpr (n == NEDGE) {
      if (x.p->isIdentity()) {
        auto idx = incIdx ? static_cast<Qubit>(y.p->v + 1) : y.p->v;
        auto e = makeDDNode(idx, std::array{yCopy, Edge<Node>::zero(),
                                            Edge<Node>::zero(), yCopy});
        for (auto i = 0; i < x.p->v; ++i) {
          idx = incIdx ? (e.p->v + 1) : e.p->v;
          e = makeDDNode(
              idx, std::array{e, Edge<Node>::zero(), Edge<Node>::zero(), e});
        }
        computeTable.insert(xCopy, yCopy, {e.p, e.w});
        return {e.p, cn.getCached(rWeight)};
      }
    }

    std::array<Edge<Node>, n> edge{};
    for (auto i = 0U; i < n; ++i) {
      edge[i] = kronecker2(x.p->e[i], y, incIdx);
    }

    auto idx = incIdx ? (y.p->v + x.p->v + 1) : x.p->v;
    auto e = makeDDNode(static_cast<Qubit>(idx), edge, true);
    computeTable.insert(xCopy, yCopy, {e.p, e.w});
    return {e.p, cn.getCached(rWeight)};
  }

  ///
  /// (Partial) trace
  ///
public:
  mEdge partialTrace(const mEdge& a, const std::vector<bool>& eliminate) {
    [[maybe_unused]] const auto before = cn.cacheCount();
    const auto result = trace(a, eliminate);
    [[maybe_unused]] const auto after = cn.cacheCount();
    assert(before == after);
    return result;
  }
  ComplexValue trace(const mEdge& a) {
    const auto eliminate = std::vector<bool>(nqubits, true);
    [[maybe_unused]] const auto before = cn.cacheCount();
    const auto res = partialTrace(a, eliminate);
    [[maybe_unused]] const auto after = cn.cacheCount();
    assert(before == after);
    return {RealNumber::val(res.w.r), RealNumber::val(res.w.i)};
  }
  bool isCloseToIdentity(const mEdge& m, const dd::fp tol = 1e-10) {
    std::unordered_set<decltype(m.p)> visited{};
    visited.reserve(mUniqueTable.getNumActiveEntries());
    return isCloseToIdentityRecursive(m, visited, tol);
  }

private:
  /// TODO: introduce a compute table for the trace?
  mEdge trace(const mEdge& a, const std::vector<bool>& eliminate,
              std::size_t alreadyEliminated = 0) {
    if (a.w.approximatelyZero()) {
      return mEdge::zero();
    }

    if (a.isTerminal() || std::none_of(eliminate.begin(), eliminate.end(),
                                       [](bool v) { return v; })) {
      return a;
    }

    const auto v = a.p->v;
    if (eliminate[v]) {
      const auto elims = alreadyEliminated + 1;
      auto r = mEdge::zero();

      const auto t0 = trace(a.p->e[0], eliminate, elims);
      r = add2(r, t0, v - 1);
      auto r1 = r;

      const auto t1 = trace(a.p->e[3], eliminate, elims);
      r = add2(r, t1, v - 1);
      auto r2 = r;

      if (r.w.exactlyOne()) {
        r.w = a.w;
      } else {
        // better safe than sorry. this may result in complex
        // values with magnitude > 1 in the complex table
        r.w = cn.lookup(r.w * a.w);
      }

      cn.returnToCache(r1.w);
      cn.returnToCache(r2.w);

      return r;
    }

    std::array<mEdge, NEDGE> edge{};
    std::transform(
        a.p->e.cbegin(), a.p->e.cend(), edge.begin(),
        [this, &eliminate, &alreadyEliminated](const mEdge& e) -> mEdge {
          return trace(e, eliminate, alreadyEliminated);
        });
    const auto adjustedV =
        static_cast<Qubit>(static_cast<std::size_t>(a.p->v) -
                           (static_cast<std::size_t>(std::count(
                                eliminate.begin(), eliminate.end(), true)) -
                            alreadyEliminated));
    auto r = makeDDNode(adjustedV, edge);
    if (r.w.exactlyOne()) {
      r.w = a.w;
    } else {
      r.w = cn.lookup(r.w * a.w);
    }
    return r;
  }

  bool isCloseToIdentityRecursive(const mEdge& m,
                                  std::unordered_set<decltype(m.p)>& visited,
                                  const dd::fp tol) {
    // immediately return if this node has already been visited
    if (visited.find(m.p) != visited.end()) {
      return true;
    }

    // immediately return of this node is identical to the identity
    if (m.isTerminal() || m.p->isIdentity()) {
      return true;
    }

    // check whether any of the middle successors is non-zero, i.e., m = [ x 0 0
    // y ]
    const auto mag1 = dd::ComplexNumbers::mag2(m.p->e[1U].w);
    const auto mag2 = dd::ComplexNumbers::mag2(m.p->e[2U].w);
    if (mag1 > tol || mag2 > tol) {
      visited.insert(m.p);
      return false;
    }

    // check whether  m = [ ~1 0 0 y ]
    const auto mag0 = dd::ComplexNumbers::mag2(m.p->e[0U].w);
    if (std::abs(mag0 - 1.0) > tol) {
      visited.insert(m.p);
      return false;
    }
    const auto arg0 = dd::ComplexNumbers::arg(m.p->e[0U].w);
    if (std::abs(arg0) > tol) {
      visited.insert(m.p);
      return false;
    }

    // check whether m = [ x 0 0 ~1 ] or m = [ x 0 0 ~0 ] (the last case is true
    // for an ancillary qubit)
    const auto mag3 = dd::ComplexNumbers::mag2(m.p->e[3U].w);
    if (mag3 > tol) {
      if (std::abs(mag3 - 1.0) > tol) {
        visited.insert(m.p);
        return false;
      }
      const auto arg3 = dd::ComplexNumbers::arg(m.p->e[3U].w);
      if (std::abs(arg3) > tol) {
        visited.insert(m.p);
        return false;
      }
    }

    // m either has the form [ ~1 0 0 ~1 ] or [ ~1 0 0 ~0 ]
    const auto ident0 = isCloseToIdentityRecursive(m.p->e[0U], visited, tol);
    if (!ident0) {
      visited.insert(m.p);
      return false;
    }

    // m either has the form [ I 0 0 ~1 ] or [ I 0 0 ~0 ]
    const auto ident3 = isCloseToIdentityRecursive(m.p->e[3U], visited, tol);
    visited.insert(m.p);
    return ident3;
  }

public:
  ///
  /// Identity matrices
  ///
  // create n-qubit identity DD. makeIdent(n) === makeIdent(0, n-1)
  mEdge makeIdent(const std::size_t n) {
    if (n == 0U) {
      return mEdge::one();
    }
    return makeIdent(0, n - 1);
  }
  mEdge makeIdent(const std::size_t leastSignificantQubit,
                  const std::size_t mostSignificantQubit) {
    if (mostSignificantQubit < leastSignificantQubit) {
      return mEdge::one();
    }

    const auto& entry = idTable.at(mostSignificantQubit);
    if (leastSignificantQubit == 0 && entry.p != nullptr) {
      return entry;
    }

    if (mostSignificantQubit >= 1) {
      const auto& prevEntry = idTable.at(mostSignificantQubit - 1);
      if (!prevEntry.isTerminal()) {
        idTable.at(mostSignificantQubit) = makeDDNode(
            static_cast<Qubit>(mostSignificantQubit),
            std::array{prevEntry, mEdge::zero(), mEdge::zero(), prevEntry});
        return idTable[mostSignificantQubit];
      }
    }

    auto e = makeDDNode(
        static_cast<Qubit>(leastSignificantQubit),
        std::array{mEdge::one(), mEdge::zero(), mEdge::zero(), mEdge::one()});
    for (auto k = leastSignificantQubit + 1; k <= mostSignificantQubit; ++k) {
      e = makeDDNode(static_cast<Qubit>(k),
                     std::array{e, mEdge::zero(), mEdge::zero(), e});
    }
    if (leastSignificantQubit == 0) {
      idTable.at(mostSignificantQubit) = e;
    }
    return e;
  }

  // identity table access and reset
  [[nodiscard]] const auto& getIdentityTable() const { return idTable; }

  void clearIdentityTable() {
    for (auto& entry : idTable) {
      entry.p = nullptr;
    }
  }

  mEdge createInitialMatrix(const std::size_t n,
                            const std::vector<bool>& ancillary) {
    auto e = makeIdent(n);
    incRef(e);
    return reduceAncillae(e, ancillary);
  }

private:
  std::vector<mEdge> idTable{};

  ///
  /// Noise Operations
  ///
public:
  StochasticNoiseOperationTable<mEdge, Config::STOCHASTIC_CACHE_OPS>
      stochasticNoiseOperationCache{nqubits};
  DensityNoiseTable<dEdge, dEdge, Config::CT_DM_NOISE_NBUCKET> densityNoise{};

  ///
  /// Ancillary and garbage reduction
  ///
  mEdge reduceAncillae(mEdge& e, const std::vector<bool>& ancillary,
                       const bool regular = true) {
    // return if no more garbage left
    if (std::none_of(ancillary.begin(), ancillary.end(),
                     [](bool v) { return v; }) ||
        e.isTerminal()) {
      return e;
    }
    Qubit lowerbound = 0;
    for (auto i = 0U; i < ancillary.size(); ++i) {
      if (ancillary[i]) {
        lowerbound = static_cast<Qubit>(i);
        break;
      }
    }
    if (e.p->v < lowerbound) {
      return e;
    }
    auto f = reduceAncillaeRecursion(e, ancillary, lowerbound, regular);
    incRef(f);
    decRef(e);
    return f;
  }

  // Garbage reduction works for reversible circuits --- to be thoroughly tested
  // for quantum circuits
  vEdge reduceGarbage(vEdge& e, const std::vector<bool>& garbage) {
    // return if no more garbage left
    if (std::none_of(garbage.begin(), garbage.end(),
                     [](bool v) { return v; }) ||
        e.isTerminal()) {
      return e;
    }
    Qubit lowerbound = 0;
    for (std::size_t i = 0U; i < garbage.size(); ++i) {
      if (garbage[i]) {
        lowerbound = static_cast<Qubit>(i);
        break;
      }
    }
    if (e.p->v < lowerbound) {
      return e;
    }
    auto f = reduceGarbageRecursion(e, garbage, lowerbound);
    incRef(f);
    decRef(e);
    return f;
  }
  mEdge reduceGarbage(mEdge& e, const std::vector<bool>& garbage,
                      const bool regular = true) {
    // return if no more garbage left
    if (std::none_of(garbage.begin(), garbage.end(),
                     [](bool v) { return v; }) ||
        e.isTerminal()) {
      return e;
    }
    Qubit lowerbound = 0;
    for (auto i = 0U; i < garbage.size(); ++i) {
      if (garbage[i]) {
        lowerbound = static_cast<Qubit>(i);
        break;
      }
    }
    if (e.p->v < lowerbound) {
      return e;
    }
    auto f = reduceGarbageRecursion(e, garbage, lowerbound, regular);
    incRef(f);
    decRef(e);
    return f;
  }

private:
  mEdge reduceAncillaeRecursion(mEdge& e, const std::vector<bool>& ancillary,
                                const Qubit lowerbound,
                                const bool regular = true) {
    if (e.p->v < lowerbound) {
      return e;
    }

    auto f = e;

    std::array<mEdge, NEDGE> edges{};
    std::bitset<NEDGE> handled{};
    for (auto i = 0U; i < NEDGE; ++i) {
      if (!handled.test(i)) {
        if (e.p->e[i].isTerminal()) {
          edges[i] = e.p->e[i];
        } else {
          edges[i] = reduceAncillaeRecursion(f.p->e[i], ancillary, lowerbound,
                                             regular);
          for (auto j = i + 1; j < NEDGE; ++j) {
            if (e.p->e[i].p == e.p->e[j].p) {
              edges[j] = edges[i];
              handled.set(j);
            }
          }
        }
        handled.set(i);
      }
    }
    f = makeDDNode(f.p->v, edges);

    // something to reduce for this qubit
    if (ancillary[f.p->v]) {
      if (regular) {
        if (!f.p->e[1].w.exactlyZero() || !f.p->e[3].w.exactlyZero()) {
          f = makeDDNode(f.p->v, std::array{f.p->e[0], mEdge::zero(), f.p->e[2],
                                            mEdge::zero()});
        }
      } else {
        if (!f.p->e[2].w.exactlyZero() || !f.p->e[3].w.exactlyZero()) {
          f = makeDDNode(f.p->v, std::array{f.p->e[0], f.p->e[1], mEdge::zero(),
                                            mEdge::zero()});
        }
      }
    }
    f.w = cn.lookup(f.w * e.w);
    return f;
  }

  vEdge reduceGarbageRecursion(vEdge& e, const std::vector<bool>& garbage,
                               const Qubit lowerbound) {
    if (e.p->v < lowerbound) {
      return e;
    }

    auto f = e;

    std::array<vEdge, RADIX> edges{};
    std::bitset<RADIX> handled{};
    for (auto i = 0U; i < RADIX; ++i) {
      if (!handled.test(i)) {
        if (e.p->e[i].isTerminal()) {
          edges[i] = e.p->e[i];
        } else {
          edges[i] = reduceGarbageRecursion(f.p->e[i], garbage, lowerbound);
          for (auto j = i + 1; j < RADIX; ++j) {
            if (e.p->e[i].p == e.p->e[j].p) {
              edges[j] = edges[i];
              handled.set(j);
            }
          }
        }
        handled.set(i);
      }
    }
    f = makeDDNode(f.p->v, edges);

    // something to reduce for this qubit
    if (garbage[f.p->v]) {
      if (!f.p->e[1].w.exactlyZero()) {
        vEdge g{};
        if (f.p->e[0].w.exactlyZero() && !f.p->e[1].w.exactlyZero()) {
          g = f.p->e[1];
        } else if (!f.p->e[1].w.exactlyZero()) {
          g = add(f.p->e[0], f.p->e[1]);
        } else {
          g = f.p->e[0];
        }
        f = makeDDNode(e.p->v, std::array{g, vEdge::zero()});
      }
    }
    f.w = cn.lookup(f.w * e.w);

    // Quick-fix for normalization bug
    if (ComplexNumbers::mag2(f.w) > 1.0) {
      f.w = Complex::one();
    }

    return f;
  }
  mEdge reduceGarbageRecursion(mEdge& e, const std::vector<bool>& garbage,
                               const Qubit lowerbound,
                               const bool regular = true) {
    if (e.p->v < lowerbound) {
      return e;
    }

    auto f = e;

    std::array<mEdge, NEDGE> edges{};
    std::bitset<NEDGE> handled{};
    for (auto i = 0U; i < NEDGE; ++i) {
      if (!handled.test(i)) {
        if (e.p->e[i].isTerminal()) {
          edges[i] = e.p->e[i];
        } else {
          edges[i] =
              reduceGarbageRecursion(f.p->e[i], garbage, lowerbound, regular);
          for (auto j = i + 1; j < NEDGE; ++j) {
            if (e.p->e[i].p == e.p->e[j].p) {
              edges[j] = edges[i];
              handled.set(j);
            }
          }
        }
        handled.set(i);
      }
    }
    f = makeDDNode(f.p->v, edges);

    // something to reduce for this qubit
    if (garbage[f.p->v]) {
      if (regular) {
        if (!f.p->e[2].w.exactlyZero() || !f.p->e[3].w.exactlyZero()) {
          mEdge g{};
          if (f.p->e[0].w.exactlyZero() && !f.p->e[2].w.exactlyZero()) {
            g = f.p->e[2];
          } else if (!f.p->e[2].w.exactlyZero()) {
            g = add(f.p->e[0], f.p->e[2]);
          } else {
            g = f.p->e[0];
          }
          mEdge h{};
          if (f.p->e[1].w.exactlyZero() && !f.p->e[3].w.exactlyZero()) {
            h = f.p->e[3];
          } else if (!f.p->e[3].w.exactlyZero()) {
            h = add(f.p->e[1], f.p->e[3]);
          } else {
            h = f.p->e[1];
          }
          f = makeDDNode(e.p->v,
                         std::array{g, h, mEdge::zero(), mEdge::zero()});
        }
      } else {
        if (!f.p->e[1].w.exactlyZero() || !f.p->e[3].w.exactlyZero()) {
          mEdge g{};
          if (f.p->e[0].w.exactlyZero() && !f.p->e[1].w.exactlyZero()) {
            g = f.p->e[1];
          } else if (!f.p->e[1].w.exactlyZero()) {
            g = add(f.p->e[0], f.p->e[1]);
          } else {
            g = f.p->e[0];
          }
          mEdge h{};
          if (f.p->e[2].w.exactlyZero() && !f.p->e[3].w.exactlyZero()) {
            h = f.p->e[3];
          } else if (!f.p->e[3].w.exactlyZero()) {
            h = add(f.p->e[2], f.p->e[3]);
          } else {
            h = f.p->e[2];
          }
          f = makeDDNode(e.p->v,
                         std::array{g, mEdge::zero(), h, mEdge::zero()});
        }
      }
    }
    f.w = cn.lookup(f.w * e.w);

    // Quick-fix for normalization bug
    if (ComplexNumbers::mag2(f.w) > 1.0) {
      f.w = Complex::one();
    }

    return f;
  }

  ///
  /// Vector and matrix extraction from DDs
  ///
public:
  // transfers a decision diagram from another package to this package
  template <class Node> Edge<Node> transfer(Edge<Node>& original) {
    if (original.isTerminal()) {
      return {original.p, cn.lookup(original.w)};
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
          edges[i].w = cn.lookup(currentEdge->p->e[i].w);
        }
        root = makeDDNode(currentEdge->p->v, edges);
        mappedNode[currentEdge->p] = root.p;
        currentEdge = nullptr;
      }
    } while (!stack.empty());
    root.w = cn.lookup(original.w * root.w);
    return root;
  }

  ///
  /// Deserialization
  /// Note: do not rely on the binary format being portable across different
  /// architectures/platforms
  ///

  template <class Node, class Edge = Edge<Node>,
            std::size_t N = std::tuple_size_v<decltype(Node::e)>>
  Edge deserialize(std::istream& is, const bool readBinary = false) {
    auto result = Edge::zero();
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
    return {result.p, cn.lookup(result.w * rootweight)};
  }

  template <class Node, class Edge = Edge<Node>>
  Edge deserialize(const std::string& inputFilename, const bool readBinary) {
    auto ifs = std::ifstream(inputFilename, std::ios::binary);

    if (!ifs.good()) {
      throw std::invalid_argument("Cannot open serialized file: " +
                                  inputFilename);
    }

    return deserialize<Node>(ifs, readBinary);
  }

private:
  template <class Node, class Edge = Edge<Node>,
            std::size_t N = std::tuple_size_v<decltype(Node::e)>>
  Edge deserializeNode(const std::int64_t index, const Qubit v,
                       std::array<std::int64_t, N>& edgeIdx,
                       const std::array<ComplexValue, N>& edgeWeight,
                       std::unordered_map<std::int64_t, Node*>& nodes) {
    if (index == -1) {
      return Edge::zero();
    }

    std::array<Edge, N> edges{};
    for (auto i = 0U; i < N; ++i) {
      if (edgeIdx[i] == -2) {
        edges[i] = Edge::zero();
      } else {
        if (edgeIdx[i] == -1) {
          edges[i] = Edge::one();
        } else {
          edges[i].p = nodes[edgeIdx[i]];
        }
        edges[i].w = cn.lookup(edgeWeight[i]);
      }
    }

    auto newedge = makeDDNode(v, edges);
    nodes[index] = newedge.p;

    // reset
    edgeIdx.fill(-2);

    return newedge;
  }
};

} // namespace dd
