#pragma once

#include "Definitions.hpp"
#include "Permutation.hpp"
#include "dd/CachedEdge.hpp"
#include "dd/Complex.hpp"
#include "dd/ComplexNumbers.hpp"
#include "dd/ComputeTable.hpp"
#include "dd/DDDefinitions.hpp"
#include "dd/DDpackageConfig.hpp"
#include "dd/DensityNoiseTable.hpp"
#include "dd/Edge.hpp"
#include "dd/GateMatrixDefinitions.hpp"
#include "dd/MemoryManager.hpp"
#include "dd/Node.hpp"
#include "dd/Package_fwd.hpp"
#include "dd/RealNumber.hpp"
#include "dd/RealNumberUniqueTable.hpp"
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
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <optional>
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
#include <utility>
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
  ComplexNumbers cn{cUniqueTable};

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

    auto f = vCachedEdge::one();
    for (std::size_t p = start; p < n + start; ++p) {
      switch (state[p]) {
      case BasisStates::zero:
        f = makeDDNode(static_cast<Qubit>(p),
                       std::array{f, vCachedEdge::zero()});
        break;
      case BasisStates::one:
        f = makeDDNode(static_cast<Qubit>(p),
                       std::array{vCachedEdge::zero(), f});
        break;
      case BasisStates::plus:
        f = makeDDNode(static_cast<Qubit>(p),
                       std::array<vCachedEdge, RADIX>{
                           {{f.p, dd::SQRT2_2}, {f.p, dd::SQRT2_2}}});
        break;
      case BasisStates::minus:
        f = makeDDNode(static_cast<Qubit>(p),
                       std::array<vCachedEdge, RADIX>{
                           {{f.p, dd::SQRT2_2}, {f.p, -dd::SQRT2_2}}});
        break;
      case BasisStates::right:
        f = makeDDNode(static_cast<Qubit>(p),
                       std::array<vCachedEdge, RADIX>{
                           {{f.p, dd::SQRT2_2}, {f.p, {0, dd::SQRT2_2}}}});
        break;
      case BasisStates::left:
        f = makeDDNode(static_cast<Qubit>(p),
                       std::array<vCachedEdge, RADIX>{
                           {{f.p, dd::SQRT2_2}, {f.p, {0, -dd::SQRT2_2}}}});
        break;
      }
    }
    return {f.p, cn.lookup(f.w)};
  }

  // generate general GHZ state with n qubits
  vEdge makeGHZState(const std::size_t n) {
    if (n > nqubits) {
      throw std::runtime_error{
          "Requested state with " + std::to_string(n) +
          " qubits, but current package configuration only supports up to " +
          std::to_string(nqubits) +
          " qubits. Please allocate a larger package instance."};
    }

    if (n == 0U) {
      return vEdge::one();
    }

    auto leftSubtree = vEdge::one();
    auto rightSubtree = vEdge::one();

    for (std::size_t p = 0; p < n - 1; ++p) {
      leftSubtree = makeDDNode(static_cast<Qubit>(p),
                               std::array{leftSubtree, vEdge::zero()});
      rightSubtree = makeDDNode(static_cast<Qubit>(p),
                                std::array{vEdge::zero(), rightSubtree});
    }

    return makeDDNode(
        static_cast<Qubit>(n - 1),
        std::array<vEdge, RADIX>{
            {{leftSubtree.p, {&constants::sqrt2over2, &constants::zero}},
             {rightSubtree.p, {&constants::sqrt2over2, &constants::zero}}}});
  }

  // generate general W state with n qubits
  vEdge makeWState(const std::size_t n) {
    if (n > nqubits) {
      throw std::runtime_error{
          "Requested state with " + std::to_string(n) +
          " qubits, but current package configuration only supports up to " +
          std::to_string(nqubits) +
          " qubits. Please allocate a larger package instance."};
    }

    if (n == 0U) {
      return vEdge::one();
    }

    auto leftSubtree = vEdge::zero();
    if ((1. / sqrt(static_cast<double>(n))) < RealNumber::eps) {
      throw std::runtime_error(
          "Requested qubit size for generating W-state would lead to an "
          "underflow due to 1 / sqrt(n) being smaller than the currently set "
          "tolerance " +
          std::to_string(RealNumber::eps) +
          ". If you still wanna run the computation, please lower "
          "the tolerance accordingly.");
    }

    auto rightSubtree = vEdge::terminal(cn.lookup(1. / std::sqrt(n)));
    for (size_t p = 0; p < n; ++p) {
      leftSubtree = makeDDNode(static_cast<Qubit>(p),
                               std::array{leftSubtree, rightSubtree});
      if (p != n - 1U) {
        rightSubtree = makeDDNode(static_cast<Qubit>(p),
                                  std::array{rightSubtree, vEdge::zero()});
      }
    }
    return leftSubtree;
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

    const auto level = static_cast<Qubit>(std::log2(length) - 1);
    const auto state =
        makeStateFromVector(stateVector.begin(), stateVector.end(), level);
    return {state.p, cn.lookup(state.w)};
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

    const auto level = static_cast<Qubit>(std::log2(length) - 1);
    const auto matrixDD = makeDDFromMatrix(matrix, level, 0, length, 0, width);
    return {matrixDD.p, cn.lookup(matrixDD.w)};
  }

  ///
  /// Matrix nodes, edges and quantum gates
  ///
  // build matrix representation for a single gate on an n-qubit circuit
  mEdge makeGateDD(const GateMatrix& mat, const qc::Qubit target) {
    return makeGateDD(mat, qc::Controls{}, target);
  }
  mEdge makeGateDD(const GateMatrix& mat, const qc::Control& control,
                   const qc::Qubit target) {
    return makeGateDD(mat, qc::Controls{control}, target);
  }
  mEdge makeGateDD(const GateMatrix& mat, const qc::Controls& controls,
                   const qc::Qubit target) {
    if (std::any_of(controls.begin(), controls.end(),
                    [this](const auto& c) {
                      return c.qubit > static_cast<Qubit>(nqubits - 1U);
                    }) ||
        target > static_cast<Qubit>(nqubits - 1U)) {
      throw std::runtime_error{
          "Requested gate acting on qubit(s) with index larger than " +
          std::to_string(nqubits - 1U) +
          " while the package configuration only supports up to " +
          std::to_string(nqubits) +
          " qubits. Please allocate a larger package instance."};
    }
    std::array<mCachedEdge, NEDGE> em{};
    for (auto i = 0U; i < NEDGE; ++i) {
      em[i] = mCachedEdge::terminal(mat[i]);
    }

    if (controls.empty()) {
      // Single qubit operation
      const auto e = makeDDNode(static_cast<Qubit>(target), em);
      return {e.p, cn.lookup(e.w)};
    }

    auto it = controls.begin();
    auto edges = std::array{mCachedEdge::zero(), mCachedEdge::zero(),
                            mCachedEdge::zero(), mCachedEdge::zero()};

    // process lines below target
    for (; it != controls.end() && it->qubit < target; ++it) {
      for (auto i1 = 0U; i1 < RADIX; ++i1) {
        for (auto i2 = 0U; i2 < RADIX; ++i2) {
          auto i = i1 * RADIX + i2;
          if (it->type == qc::Control::Type::Neg) { // neg. control
            edges[0] = em[i];
            edges[3] = (i1 == i2) ? mCachedEdge::one() : mCachedEdge::zero();
          } else { // pos. control
            edges[0] = (i1 == i2) ? mCachedEdge::one() : mCachedEdge::zero();
            edges[3] = em[i];
          }
          em[i] = makeDDNode(static_cast<Qubit>(it->qubit), edges);
        }
      }
    }

    // target line
    auto e = makeDDNode(static_cast<Qubit>(target), em);

    // process lines above target
    for (; it != controls.end(); ++it) {
      if (it->type == qc::Control::Type::Neg) { // neg. control
        edges[0] = e;
        edges[3] = mCachedEdge::one();
        e = makeDDNode(static_cast<Qubit>(it->qubit), edges);
      } else { // pos. control
        edges[0] = mCachedEdge::one();
        edges[3] = e;
        e = makeDDNode(static_cast<Qubit>(it->qubit), edges);
      }
    }
    return {e.p, cn.lookup(e.w)};
  }

  /**
  Creates the DD for a two-qubit gate
  @param mat Matrix representation of the gate
  @param target0 First target qubit
  @param target1 Second target qubit
  @return DD representing the gate
  @throws std::runtime_error if the number of qubits is larger than the package
  configuration
  **/
  mEdge makeTwoQubitGateDD(const TwoQubitGateMatrix& mat,
                           const qc::Qubit target0, const qc::Qubit target1) {
    return makeTwoQubitGateDD(mat, qc::Controls{}, target0, target1);
  }

  /**
  Creates the DD for a two-qubit gate
  @param mat Matrix representation of the gate
  @param controls Control qubits of the two-qubit gate
  @param target0 First target qubit
  @param target1 Second target qubit
  @return DD representing the gate
  @throws std::runtime_error if the number of qubits is larger than the package
  configuration
  **/
  mEdge makeTwoQubitGateDD(const TwoQubitGateMatrix& mat,
                           const qc::Controls& controls,
                           const qc::Qubit target0, const qc::Qubit target1) {
    // sanity check
    if (std::any_of(controls.begin(), controls.end(),
                    [this](const auto& c) {
                      return c.qubit > static_cast<Qubit>(nqubits - 1U);
                    }) ||
        target0 > static_cast<Qubit>(nqubits - 1U) ||
        target1 > static_cast<Qubit>(nqubits - 1U)) {
      throw std::runtime_error{
          "Requested gate acting on qubit(s) with index larger than " +
          std::to_string(nqubits - 1U) +
          " while the package configuration only supports up to " +
          std::to_string(nqubits) +
          " qubits. Please allocate a larger package instance."};
    }

    // create terminal edge matrix
    std::array<std::array<mCachedEdge, NEDGE>, NEDGE> em{};
    for (auto i1 = 0U; i1 < NEDGE; i1++) {
      const auto& matRow = mat.at(i1);
      auto& emRow = em.at(i1);
      for (auto i2 = 0U; i2 < NEDGE; i2++) {
        emRow.at(i2) = mCachedEdge::terminal(matRow.at(i2));
      }
    }

    // process lines below smaller target
    auto it = controls.begin();
    const auto smallerTarget = std::min(target0, target1);

    auto edges = std::array{mCachedEdge::zero(), mCachedEdge::zero(),
                            mCachedEdge::zero(), mCachedEdge::zero()};

    for (; it != controls.end() && it->qubit < smallerTarget; ++it) {
      for (auto row = 0U; row < NEDGE; ++row) {
        for (auto col = 0U; col < NEDGE; ++col) {
          if (it->type == qc::Control::Type::Neg) { // negative control
            edges[0] = em[row][col];
            edges[3] = (row == col) ? mCachedEdge::one() : mCachedEdge::zero();
          } else { // positive control
            edges[0] = (row == col) ? mCachedEdge::one() : mCachedEdge::zero();
            edges[3] = em[row][col];
          }
          em[row][col] = makeDDNode(static_cast<Qubit>(it->qubit), edges);
        }
      }
    }

    // process the smaller target by taking the 16 submatrices and appropriately
    // combining them into four DDs.
    std::array<mCachedEdge, NEDGE> em0{};
    for (std::size_t row = 0; row < RADIX; ++row) {
      for (std::size_t col = 0; col < RADIX; ++col) {
        std::array<mCachedEdge, NEDGE> local{};
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
        em0.at(row * RADIX + col) =
            makeDDNode(static_cast<Qubit>(smallerTarget), local);
      }
    }

    // process lines between the two targets
    const auto largerTarget = std::max(target0, target1);
    for (; it != controls.end() && it->qubit < largerTarget; ++it) {
      for (auto i = 0U; i < NEDGE; ++i) {
        if (it->type == qc::Control::Type::Neg) { // negative control
          edges[0] = em0[i];
          edges[3] =
              (i == 0 || i == 3) ? mCachedEdge::one() : mCachedEdge::zero();
        } else { // positive control
          edges[0] =
              (i == 0 || i == 3) ? mCachedEdge::one() : mCachedEdge::zero();
          edges[3] = em0[i];
        }
        em0[i] = makeDDNode(static_cast<Qubit>(it->qubit), edges);
      }
    }

    // process the larger target by combining the four DDs from the smaller
    // target
    auto e = makeDDNode(static_cast<Qubit>(largerTarget), em0);

    // process lines above the larger target
    for (; it != controls.end(); ++it) {
      if (it->type == qc::Control::Type::Neg) { // negative control
        edges[0] = e;
        edges[3] = mCachedEdge::one();
      } else { // positive control
        edges[0] = mCachedEdge::one();
        edges[3] = e;
      }
      e = makeDDNode(static_cast<Qubit>(it->qubit), edges);
    }

    return {e.p, cn.lookup(e.w)};
  }

private:
  vCachedEdge makeStateFromVector(const CVec::const_iterator& begin,
                                  const CVec::const_iterator& end,
                                  const Qubit level) {
    if (level == 0U) {
      assert(std::distance(begin, end) == 2);
      const auto zeroSuccessor = vCachedEdge::terminal(*begin);
      const auto oneSuccessor = vCachedEdge::terminal(*(begin + 1));
      return makeDDNode<vNode, CachedEdge>(0, {zeroSuccessor, oneSuccessor});
    }

    const auto half = std::distance(begin, end) / 2;
    const auto zeroSuccessor =
        makeStateFromVector(begin, begin + half, level - 1);
    const auto oneSuccessor = makeStateFromVector(begin + half, end, level - 1);
    return makeDDNode<vNode, CachedEdge>(level, {zeroSuccessor, oneSuccessor});
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
  @return An mCachedEdge representing the root node of the created DD.
  @details This function recursively breaks down the matrix into quadrants until
  each quadrant has only one element. At each level of recursion, four new edges
  are created, one for each quadrant of the matrix. The four resulting decision
  diagram edges are used to create a new decision diagram node at the current
  level, and this node is returned as the result of the current recursive call.
  At the base case of recursion, the matrix has only one element, which is
  converted into a terminal node of the decision diagram.
  @note This function assumes that the matrix size is a power of two.
  **/
  mCachedEdge makeDDFromMatrix(const CMat& matrix, const Qubit level,
                               const std::size_t rowStart,
                               const std::size_t rowEnd,
                               const std::size_t colStart,
                               const std::size_t colEnd) {
    // base case
    if (level == 0U) {
      assert(rowEnd - rowStart == 2);
      assert(colEnd - colStart == 2);
      return makeDDNode<mNode, CachedEdge>(
          0U, {mCachedEdge::terminal(matrix[rowStart][colStart]),
               mCachedEdge::terminal(matrix[rowStart][colStart + 1]),
               mCachedEdge::terminal(matrix[rowStart + 1][colStart]),
               mCachedEdge::terminal(matrix[rowStart + 1][colStart + 1])});
    }

    // recursively call the function on all quadrants
    const auto rowMid = (rowStart + rowEnd) / 2;
    const auto colMid = (colStart + colEnd) / 2;
    const auto l = static_cast<Qubit>(level - 1U);

    return makeDDNode<mNode, CachedEdge>(
        level, {makeDDFromMatrix(matrix, l, rowStart, rowMid, colStart, colMid),
                makeDDFromMatrix(matrix, l, rowStart, rowMid, colMid, colEnd),
                makeDDFromMatrix(matrix, l, rowMid, rowEnd, colStart, colMid),
                makeDDFromMatrix(matrix, l, rowMid, rowEnd, colMid, colEnd)});
  }

public:
  // create a normalized DD node and return an edge pointing to it. The node is
  // not recreated if it already exists.
  template <class Node, template <class> class EdgeType>
  EdgeType<Node>
  makeDDNode(const Qubit var,
             const std::array<EdgeType<Node>,
                              std::tuple_size_v<decltype(Node::e)>>& edges,
             [[maybe_unused]] const bool generateDensityMatrix = false) {
    auto& memoryManager = getMemoryManager<Node>();
    auto p = memoryManager.get();
    assert(p->ref == 0U);

    p->v = var;
    if constexpr (std::is_same_v<Node, mNode> || std::is_same_v<Node, dNode>) {
      p->flags = 0;
      if constexpr (std::is_same_v<Node, dNode>) {
        p->setDensityMatrixNodeFlag(generateDensityMatrix);
      }
    }

    auto e = EdgeType<Node>::normalize(p, edges, memoryManager, cn);
    if constexpr (std::is_same_v<Node, mNode> || std::is_same_v<Node, dNode>) {
      if (!e.isTerminal()) {
        const auto& es = e.p->e;
        // Check if node resembles the identity. If so, skip it.
        if ((es[0].p == es[3].p) &&
            (es[0].w.exactlyOne() && es[1].w.exactlyZero() &&
             es[2].w.exactlyZero() && es[3].w.exactlyOne())) {
          auto* ptr = es[0].p;
          memoryManager.returnEntry(e.p);
          return EdgeType<Node>{ptr, e.w};
        }
      }
    }

    // look it up in the unique tables
    auto& uniqueTable = getUniqueTable<Node>();
    auto* l = uniqueTable.lookup(e.p);

    return EdgeType<Node>{l, e.w};
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
    vectorAddMagnitudes.clear();
    matrixAddMagnitudes.clear();
    conjugateVector.clear();
    conjugateMatrixTranspose.clear();
    matrixMatrixMultiplication.clear();
    matrixVectorMultiplication.clear();
    vectorInnerProduct.clear();
    vectorKronecker.clear();
    matrixKronecker.clear();

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
        e = makeDDNode(static_cast<Qubit>(p), edges);
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

  char measureOneCollapsing(dEdge& e, const Qubit index, std::mt19937_64& mt) {
    char measuredResult = '0';
    dEdge::alignDensityEdge(e);
    const auto nrQubits = e.p->v + 1U;
    dEdge::setDensityMatrixTrue(e);

    auto const measZeroDd = makeGateDD(MEAS_ZERO_MAT, index);

    auto tmp0 = conjugateTranspose(measZeroDd);
    auto tmp1 = multiply(e, densityFromMatrixEdge(tmp0), false);
    auto tmp2 = multiply(densityFromMatrixEdge(measZeroDd), tmp1, true);
    auto densityMatrixTrace = trace(tmp2, nrQubits);

    std::uniform_real_distribution<fp> dist(0., 1.);
    if (const auto threshold = dist(mt); threshold > densityMatrixTrace.r) {
      auto const measOneDd = makeGateDD(MEAS_ONE_MAT, index);
      tmp0 = conjugateTranspose(measOneDd);
      tmp1 = multiply(e, densityFromMatrixEdge(tmp0), false);
      tmp2 = multiply(densityFromMatrixEdge(measOneDd), tmp1, true);
      measuredResult = '1';
      densityMatrixTrace = trace(tmp2, nrQubits);
    }

    incRef(tmp2);
    dEdge::alignDensityEdge(e);
    decRef(e);
    e = tmp2;
    dEdge::setDensityMatrixTrue(e);

    // Normalize density matrix
    auto result = e.w / densityMatrixTrace;
    cn.decRef(e.w);
    e.w = cn.lookup(result);
    cn.incRef(e.w);
    return measuredResult;
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

    const auto measurementGate = makeGateDD(measurementMatrix, index);

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

  ComputeTable<vCachedEdge, vCachedEdge, vCachedEdge,
               Config::CT_VEC_ADD_MAG_NBUCKET>
      vectorAddMagnitudes{};
  ComputeTable<mCachedEdge, mCachedEdge, mCachedEdge,
               Config::CT_MAT_ADD_MAG_NBUCKET>
      matrixAddMagnitudes{};

  template <class Node> [[nodiscard]] auto& getAddMagnitudesComputeTable() {
    if constexpr (std::is_same_v<Node, vNode>) {
      return vectorAddMagnitudes;
    } else if constexpr (std::is_same_v<Node, mNode>) {
      return matrixAddMagnitudes;
    }
  }

  template <class Node>
  Edge<Node> add(const Edge<Node>& x, const Edge<Node>& y) {
    Qubit var{};
    if (!x.isTerminal()) {
      var = x.p->v;
    }
    if (!y.isTerminal() && (y.p->v) > var) {
      var = y.p->v;
    }

    const auto result = add2(CachedEdge{x.p, x.w}, {y.p, y.w}, var);
    return cn.lookup(result);
  }

  template <class Node>
  CachedEdge<Node> add2(const CachedEdge<Node>& x, const CachedEdge<Node>& y,
                        const Qubit var) {
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

    auto& computeTable = getAddComputeTable<Node>();
    if (const auto* r = computeTable.lookup(x, y); r != nullptr) {
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
        edge[i] = add2(e1, e2, var - 1);
        dNode::revertDmChangesToNode(e2.p);
        dNode::revertDmChangesToNode(e1.p);
      } else {
        edge[i] = add2(e1, e2, var - 1);
      }
    }
    auto r = makeDDNode(var, edge);
    computeTable.insert(x, y, r);
    return r;
  }

  /**
   For two vectors (or matrices) x and y it returns a result r such that for
   each index i: r[i] = sqrt(|x[i]|^2 + |y[i]|^2)
   @param x DD representation of the first operand
   @param y DD representation of the first operand
   @param var number of qubits in the DD
   @return DD representing the result
   **/
  template <class Node>
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

    auto& computeTable = getAddMagnitudesComputeTable<Node>();
    if (const auto* r = computeTable.lookup(x, y); r != nullptr) {
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
    computeTable.insert(x, y, r);
    return r;
  }

  ///
  /// Vector conjugation
  ///
  UnaryComputeTable<vNode*, vCachedEdge, Config::CT_VEC_CONJ_NBUCKET>
      conjugateVector{};

  vEdge conjugate(const vEdge& a) {
    auto r = conjugateRec(a);
    return {r.p, cn.lookup(r.w)};
  }

  vCachedEdge conjugateRec(const vEdge& a) {
    if (a.isZeroTerminal()) {
      return vCachedEdge::zero();
    }

    if (a.isTerminal()) {
      return {a.p, ComplexNumbers::conj(a.w)};
    }

    if (const auto* r = conjugateVector.lookup(a.p); r != nullptr) {
      return {r->p, r->w * ComplexNumbers::conj(a.w)};
    }

    std::array<vCachedEdge, 2> e{};
    e[0] = conjugateRec(a.p->e[0]);
    e[1] = conjugateRec(a.p->e[1]);
    auto res = makeDDNode(a.p->v, e);
    conjugateVector.insert(a.p, res);
    res.w = res.w * ComplexNumbers::conj(a.w);
    return res;
  }

  ///
  /// Matrix (conjugate) transpose
  ///
  UnaryComputeTable<mNode*, mCachedEdge, Config::CT_MAT_CONJ_TRANS_NBUCKET>
      conjugateMatrixTranspose{};

  mEdge conjugateTranspose(const mEdge& a) {
    auto r = conjugateTransposeRec(a);
    return {r.p, cn.lookup(r.w)};
  }

  mCachedEdge conjugateTransposeRec(const mEdge& a) {
    if (a.isTerminal()) { // terminal case
      return {a.p, ComplexNumbers::conj(a.w)};
    }

    // check if in compute table
    if (const auto* r = conjugateMatrixTranspose.lookup(a.p); r != nullptr) {
      return {r->p, r->w * ComplexNumbers::conj(a.w)};
    }

    std::array<mCachedEdge, NEDGE> e{};
    // conjugate transpose submatrices and rearrange as required
    for (auto i = 0U; i < RADIX; ++i) {
      for (auto j = 0U; j < RADIX; ++j) {
        e[RADIX * i + j] = conjugateTransposeRec(a.p->e[RADIX * j + i]);
      }
    }
    // create new top node
    auto res = makeDDNode(a.p->v, e);

    // put it in the compute table
    conjugateMatrixTranspose.insert(a.p, res);

    // adjust top weight including conjugate
    return {res.p, res.w * ComplexNumbers::conj(a.w)};
  }

  ///
  /// Multiplication
  ///
  ComputeTable<mNode*, vNode*, vCachedEdge, Config::CT_MAT_VEC_MULT_NBUCKET>
      matrixVectorMultiplication{};
  ComputeTable<mNode*, mNode*, mCachedEdge, Config::CT_MAT_MAT_MULT_NBUCKET>
      matrixMatrixMultiplication{};
  ComputeTable<dNode*, dNode*, dCachedEdge, Config::CT_DM_DM_MULT_NBUCKET>
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

  dEdge applyOperationToDensity(dEdge& e, const mEdge& operation) {
    auto tmp0 = conjugateTranspose(operation);
    auto tmp1 = multiply(e, densityFromMatrixEdge(tmp0), false);
    auto tmp2 = multiply(densityFromMatrixEdge(operation), tmp1, true);
    incRef(tmp2);
    dEdge::alignDensityEdge(e);
    decRef(e);
    e = tmp2;
    dEdge::setDensityMatrixTrue(e);
    return e;
  }

  template <class LeftOperandNode, class RightOperandNode>
  Edge<RightOperandNode>
  multiply(const Edge<LeftOperandNode>& x, const Edge<RightOperandNode>& y,
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
    Qubit var{};
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

      const auto e = multiply2(xCopy, yCopy, var, generateDensityMatrix);
      dEdge::revertDmChangesToEdges(xCopy, yCopy);
      return cn.lookup(e);
    } else {
      if (!x.isTerminal()) {
        var = x.p->v;
      }
      if (!y.isTerminal() && y.p->v > var) {
        var = y.p->v;
      }
      const auto e = multiply2(x, y, var);
      return cn.lookup(e);
    }
  }

private:
  template <class LeftOperandNode, class RightOperandNode>
  CachedEdge<RightOperandNode>
  multiply2(const Edge<LeftOperandNode>& x, const Edge<RightOperandNode>& y,
            const Qubit var,
            [[maybe_unused]] const bool generateDensityMatrix = false) {
    using LEdge = Edge<LeftOperandNode>;
    using REdge = Edge<RightOperandNode>;
    using ResultEdge = CachedEdge<RightOperandNode>;

    if (x.w.exactlyZero() || y.w.exactlyZero()) {
      return ResultEdge::zero();
    }

    const auto xWeight = static_cast<ComplexValue>(x.w);
    const auto yWeight = static_cast<ComplexValue>(y.w);
    const auto rWeight = xWeight * yWeight;
    if (x.isIdentity()) {
      if constexpr (!std::is_same_v<RightOperandNode, dNode>) {
        return {y.p, rWeight};
      } else {
        if (y.isIdentity() ||
            (dNode::isDensityMatrixTempFlagSet(y.p->flags) &&
             generateDensityMatrix) ||
            (!dNode::isDensityMatrixTempFlagSet(y.p->flags) &&
             !generateDensityMatrix)) {
          return {y.p, rWeight};
        }
      }
    }

    if constexpr (std::is_same_v<RightOperandNode, mNode> ||
                  std::is_same_v<RightOperandNode, dNode>) {
      if (y.isIdentity()) {
        if constexpr (!std::is_same_v<LeftOperandNode, dNode>) {
          return {x.p, rWeight};
        } else {
          if (x.isIdentity() ||
              (dNode::isDensityMatrixTempFlagSet(x.p->flags) &&
               generateDensityMatrix) ||
              (!dNode::isDensityMatrixTempFlagSet(x.p->flags) &&
               !generateDensityMatrix)) {
            return {x.p, rWeight};
          }
        }
      }
    }

    auto& computeTable = getMultiplicationComputeTable<RightOperandNode>();
    if (const auto* r = computeTable.lookup(x.p, y.p, generateDensityMatrix);
        r != nullptr) {
      return {r->p, r->w * rWeight};
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
          LEdge e1{};
          if (x.p != nullptr && x.p->v == var) {
            e1 = x.p->e[xIdx];
          } else {
            if (xIdx == 0 || xIdx == 3) {
              e1 = LEdge{x.p, Complex::one()};
            } else {
              e1 = LEdge::zero();
            }
          }

          const auto yIdx = j + cols * k;
          REdge e2{};
          if (y.p != nullptr && y.p->v == var) {
            e2 = y.p->e[yIdx];
          } else {
            if (yIdx == 0 || yIdx == 3) {
              e2 = REdge{y.p, Complex::one()};
            } else {
              e2 = REdge::zero();
            }
          }

          const auto v = static_cast<Qubit>(var - 1);
          if constexpr (std::is_same_v<LeftOperandNode, dNode>) {
            dCachedEdge m;
            dEdge::applyDmChangesToEdges(e1, e2);
            if (!generateDensityMatrix || idx == 1) {
              // When generateDensityMatrix is false or I have the first edge I
              // don't optimize anything and set generateDensityMatrix to false
              // for all child edges
              m = multiply2(e1, e2, v, false);
            } else if (idx == 2) {
              // When I have the second edge and generateDensityMatrix == false,
              // then edge[2] == edge[1]
              if (k == 0) {
                if (edge[1].w.approximatelyZero()) {
                  edge[2] = ResultEdge::zero();
                } else {
                  edge[2] = edge[1];
                }
              }
              continue;
            } else {
              m = multiply2(e1, e2, v, generateDensityMatrix);
            }

            if (k == 0 || edge[idx].w.exactlyZero()) {
              edge[idx] = m;
            } else if (!m.w.exactlyZero()) {
              dNode::applyDmChangesToNode(edge[idx].p);
              dNode::applyDmChangesToNode(m.p);
              edge[idx] = add2(edge[idx], m, v);
              dNode::revertDmChangesToNode(m.p);
              dNode::revertDmChangesToNode(edge[idx].p);
            }
            // Undo modifications on density matrices
            dEdge::revertDmChangesToEdges(e1, e2);
          } else {
            auto m = multiply2(e1, e2, v);

            if (k == 0 || edge[idx].w.exactlyZero()) {
              edge[idx] = m;
            } else if (!m.w.exactlyZero()) {
              edge[idx] = add2(edge[idx], m, v);
            }
          }
        }
      }
    }

    auto e = makeDDNode(var, edge, generateDensityMatrix);
    computeTable.insert(x.p, y.p, e);

    e.w = e.w * rWeight;
    return e;
  }

  ///
  /// Inner product, fidelity, expectation value
  ///
public:
  ComputeTable<vNode*, vNode*, vCachedEdge, Config::CT_VEC_INNER_PROD_NBUCKET>
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
      return 0;
    }

    const auto w = std::max(x.p->v, y.p->v);
    // Overall normalization factor needs to be conjugated
    // before input into recursive private function
    auto xCopy = vEdge{x.p, ComplexNumbers::conj(x.w)};
    return innerProduct(xCopy, y, w + 1U);
  }

  fp fidelity(const vEdge& x, const vEdge& y) {
    return innerProduct(x, y).mag2();
  }

  fp fidelityOfMeasurementOutcomes(const vEdge& e, const SparsePVec& probs,
                                   const qc::Permutation& permutation = {}) {
    if (e.w.approximatelyZero()) {
      return 0.;
    }
    return fidelityOfMeasurementOutcomesRecursive(e, probs, 0, permutation,
                                                  e.p->v + 1U);
  }

  fp fidelityOfMeasurementOutcomesRecursive(const vEdge& e,
                                            const SparsePVec& probs,
                                            const std::size_t i,
                                            const qc::Permutation& permutation,
                                            const std::size_t nQubits) {
    const auto top = ComplexNumbers::mag(e.w);
    if (e.isTerminal()) {
      auto idx = i;
      if (!permutation.empty()) {
        const auto binaryString = intToBinaryString(i, nQubits);
        std::string filteredString(permutation.size(), '0');
        for (const auto& [physical, logical] : permutation) {
          filteredString[logical] = binaryString[physical];
        }
        idx = std::stoull(filteredString, nullptr, 2);
      }
      if (auto it = probs.find(idx); it != probs.end()) {
        return top * std::sqrt(it->second);
      }
      return 0.;
    }

    std::size_t leftIdx = i;
    fp leftContribution = 0.;
    if (!e.p->e[0].w.approximatelyZero()) {
      leftContribution = fidelityOfMeasurementOutcomesRecursive(
          e.p->e[0], probs, leftIdx, permutation, nQubits);
    }

    std::size_t rightIdx = i | (1ULL << e.p->v);
    auto rightContribution = 0.;
    if (!e.p->e[1].w.approximatelyZero()) {
      rightContribution = fidelityOfMeasurementOutcomesRecursive(
          e.p->e[1], probs, rightIdx, permutation, nQubits);
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

    if (const auto* r = vectorInnerProduct.lookup(x.p, y.p); r != nullptr) {
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
        e1 = {x.p, Complex::one()};
      }
      vEdge e2{};
      if (!y.isTerminal() && y.p->v == w) {
        e2 = y.p->e[i];
      } else {
        e2 = {y.p, Complex::one()};
      }
      sum += innerProduct(e1, e2, w);
    }
    vectorInnerProduct.insert(x.p, y.p, vCachedEdge::terminal(sum));
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
    assert(!x.isZeroTerminal() && !y.isTerminal());
    if (!x.isTerminal() && x.p->v > y.p->v) {
      throw std::invalid_argument(
          "Observable must not act on more qubits than the state to compute the"
          "expectation value.");
    }

    auto yPrime = multiply(x, y);
    const ComplexValue expValue = innerProduct(y, yPrime);

    assert(RealNumber::approximatelyZero(expValue.i));
    return expValue.r;
  }

  ///
  /// Kronecker/tensor product
  ///

  ComputeTable<vNode*, vNode*, vCachedEdge, Config::CT_VEC_KRON_NBUCKET>
      vectorKronecker{};
  ComputeTable<mNode*, mNode*, mCachedEdge, Config::CT_MAT_KRON_NBUCKET>
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
                       const std::size_t yNumQubits, const bool incIdx = true) {
    if constexpr (std::is_same_v<Node, dNode>) {
      throw std::invalid_argument(
          "Kronecker is currently not supported for density matrices");
    }

    const auto e = kronecker2(x, y, yNumQubits, incIdx);
    return cn.lookup(e);
  }

private:
  template <class Node>
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
    }

    // check if we already computed the product before and return the result
    auto& computeTable = getKroneckerComputeTable<Node>();
    if (const auto* r = computeTable.lookup(x.p, y.p); r != nullptr) {
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
    computeTable.insert(x.p, y.p, {e.p, e.w});
    return {e.p, rWeight};
  }

  ///
  /// (Partial) trace
  ///
public:
  mEdge partialTrace(const mEdge& a, const std::vector<bool>& eliminate) {
    auto r = trace(a, eliminate, eliminate.size());
    return {r.p, cn.lookup(r.w)};
  }

  template <class Node>
  ComplexValue trace(const Edge<Node>& a, const std::size_t numQubits) {
    if (a.isIdentity()) {
      return a.w * std::pow(2, numQubits);
    }
    const auto eliminate = std::vector<bool>(numQubits, true);
    return trace(a, eliminate, numQubits).w;
  }

  /**
        Checks if a given matrix is close to the identity matrix, while ignoring
    any potential garbage qubits and ignoring the diagonal weights if
    `checkCloseToOne` is set to false.
        @param m An mEdge that represents the DD of the matrix.
        @param tol The accepted tolerance for the edge weights of the DD.
        @param garbage A vector of boolean values that defines which qubits are
    considered garbage qubits. If it's empty, then no qubit is considered to be
    a garbage qubit.
        @param checkCloseToOne If false, the function only checks if the matrix
    is close to a diagonal matrix.
    **/
  bool isCloseToIdentity(const mEdge& m, const dd::fp tol = 1e-10,
                         const std::vector<bool>& garbage = {},
                         const bool checkCloseToOne = true) {
    std::unordered_set<decltype(m.p)> visited{};
    visited.reserve(mUniqueTable.getNumActiveEntries());
    return isCloseToIdentityRecursive(m, visited, tol, garbage,
                                      checkCloseToOne);
  }

private:
  /// TODO: introduce a compute table for the trace?
  template <class Node>
  CachedEdge<Node> trace(const Edge<Node>& a,
                         const std::vector<bool>& eliminate, std::size_t level,
                         std::size_t alreadyEliminated = 0) {
    const auto aWeight = static_cast<ComplexValue>(a.w);
    if (aWeight.approximatelyZero()) {
      return CachedEdge<Node>::zero();
    }

    if (std::none_of(eliminate.begin(), eliminate.end(),
                     [](bool v) { return v; })) {
      return CachedEdge<Node>{a.p, aWeight};
    }

    if (a.isIdentity()) {
      const auto elims =
          std::count(eliminate.begin(),
                     eliminate.begin() + static_cast<int64_t>(level), true);
      return CachedEdge<Node>{a.p, aWeight * std::pow(2, elims)};
    }

    const auto v = a.p->v;
    if (eliminate[v]) {
      const auto elims = alreadyEliminated + 1;
      auto r = add2(trace(a.p->e[0], eliminate, level - 1, elims),
                    trace(a.p->e[3], eliminate, level - 1, elims), v - 1);

      r.w = r.w * aWeight;
      return r;
    }

    std::array<CachedEdge<Node>, NEDGE> edge{};
    std::transform(a.p->e.cbegin(), a.p->e.cend(), edge.begin(),
                   [this, &eliminate, &alreadyEliminated,
                    &level](const Edge<Node>& e) -> CachedEdge<Node> {
                     return trace(e, eliminate, level - 1, alreadyEliminated);
                   });
    const auto adjustedV =
        static_cast<Qubit>(static_cast<std::size_t>(a.p->v) -
                           (static_cast<std::size_t>(std::count(
                                eliminate.begin(), eliminate.end(), true)) -
                            alreadyEliminated));
    auto r = makeDDNode(adjustedV, edge);
    r.w = r.w * aWeight;
    return r;
  }

  bool isCloseToIdentityRecursive(const mEdge& m,
                                  std::unordered_set<decltype(m.p)>& visited,
                                  const dd::fp tol,
                                  const std::vector<bool>& garbage,
                                  const bool checkCloseToOne) {
    // immediately return if this node is identical to the identity or zero
    if (m.isTerminal()) {
      return true;
    }

    // immediately return if this node has already been visited
    if (visited.find(m.p) != visited.end()) {
      return true;
    }

    const auto n = m.p->v;

    if (garbage.size() > n && garbage[n]) {
      return isCloseToIdentityRecursive(m.p->e[0U], visited, tol, garbage,
                                        checkCloseToOne) &&
             isCloseToIdentityRecursive(m.p->e[1U], visited, tol, garbage,
                                        checkCloseToOne) &&
             isCloseToIdentityRecursive(m.p->e[2U], visited, tol, garbage,
                                        checkCloseToOne) &&
             isCloseToIdentityRecursive(m.p->e[3U], visited, tol, garbage,
                                        checkCloseToOne);
    }

    // check whether any of the middle successors is non-zero, i.e., m = [ x 0 0
    // y ]
    const auto mag1 = dd::ComplexNumbers::mag2(m.p->e[1U].w);
    const auto mag2 = dd::ComplexNumbers::mag2(m.p->e[2U].w);
    if (mag1 > tol || mag2 > tol) {
      return false;
    }

    if (checkCloseToOne) {
      // check whether  m = [ ~1 0 0 y ]
      const auto mag0 = dd::ComplexNumbers::mag2(m.p->e[0U].w);
      if (std::abs(mag0 - 1.0) > tol) {
        return false;
      }
      const auto arg0 = dd::ComplexNumbers::arg(m.p->e[0U].w);
      if (std::abs(arg0) > tol) {
        return false;
      }

      // check whether m = [ x 0 0 ~1 ] or m = [ x 0 0 ~0 ] (the last case is
      // true for an ancillary qubit)
      const auto mag3 = dd::ComplexNumbers::mag2(m.p->e[3U].w);
      if (mag3 > tol) {
        if (std::abs(mag3 - 1.0) > tol) {
          return false;
        }
        const auto arg3 = dd::ComplexNumbers::arg(m.p->e[3U].w);
        if (std::abs(arg3) > tol) {
          return false;
        }
      }
    }
    // m either has the form [ ~1 0 0 ~1 ] or [ ~1 0 0 ~0 ]
    const auto ident0 = isCloseToIdentityRecursive(m.p->e[0U], visited, tol,
                                                   garbage, checkCloseToOne);

    if (!ident0) {
      return false;
    }
    // m either has the form [ I 0 0 ~1 ] or [ I 0 0 ~0 ]
    const auto ident3 = isCloseToIdentityRecursive(m.p->e[3U], visited, tol,
                                                   garbage, checkCloseToOne);

    visited.insert(m.p);
    return ident3;
  }

public:
  ///
  /// Identity matrices
  ///
  // create identity DD represented by the one-terminal.
  mEdge makeIdent() { return mEdge::one(); }

  mEdge createInitialMatrix(const std::vector<bool>& ancillary) {
    auto e = makeIdent();
    return reduceAncillae(e, ancillary);
  }

  ///
  /// Noise Operations
  ///
  StochasticNoiseOperationTable<mEdge, Config::STOCHASTIC_CACHE_OPS>
      stochasticNoiseOperationCache{nqubits};
  DensityNoiseTable<dEdge, dEdge, Config::CT_DM_NOISE_NBUCKET> densityNoise{};

  ///
  /// Ancillary and garbage reduction
  ///
  mEdge reduceAncillae(mEdge& e, const std::vector<bool>& ancillary,
                       const bool regular = true) {
    // return if no more ancillaries left
    if (std::none_of(ancillary.begin(), ancillary.end(),
                     [](bool v) { return v; }) ||
        e.isZeroTerminal()) {
      return e;
    }

    // if we have only identities and no other nodes
    if (e.isIdentity()) {
      auto g = e;
      for (auto i = 0U; i < ancillary.size(); ++i) {
        if (ancillary[i]) {
          g = makeDDNode(
              static_cast<Qubit>(i),
              std::array{g, mEdge::zero(), mEdge::zero(), mEdge::zero()});
        }
      }
      incRef(g);
      return g;
    }

    Qubit lowerbound = 0;
    for (auto i = 0U; i < ancillary.size(); ++i) {
      if (ancillary[i]) {
        lowerbound = static_cast<Qubit>(i);
        break;
      }
    }

    auto g = CachedEdge<mNode>{e.p, 1.};
    if (e.p->v >= lowerbound) {
      g = reduceAncillaeRecursion(e.p, ancillary, lowerbound, regular);
    }

    for (std::size_t i = e.p->v + 1; i < ancillary.size(); ++i) {
      if (ancillary[i]) {
        g = makeDDNode(static_cast<Qubit>(i),
                       std::array{g, mCachedEdge::zero(), mCachedEdge::zero(),
                                  mCachedEdge::zero()});
      }
    }
    const auto res = mEdge{g.p, cn.lookup(g.w * e.w)};
    incRef(res);
    decRef(e);
    return res;
  }

  /**
  For each garbage qubit q, sums all the entries for q = 0 and q = 1 and sets
  the entry for q = 0 to the sum and the entry for q = 1 to zero. In order to be
  sure that the probabilities of the resulting state are the sum of the
  probabilities of the initial state, we don't simply compute a sum, but we
  compute `sqrt(|a|^2 + |b|^2)` for two entries `a` and `b`.
  @param e DD representation of the matrix/vector
  @param garbage vector that describes which qubits are garbage and which ones
  are not. If garbage[i] = true, then qubit q_i is considered garbage
  @param normalizeWeights By default set to `false`. If set to `true`,  the
  function changes all weights in the DD to their magnitude, also for
  non-garbage qubits. This is used for checking partial equivalence of circuits.
  For partial equivalence, only the measurement probabilities are considered, so
  we need to consider only the magnitudes of each entry.
  @return DD representing of the reduced matrix/vector
  **/
  vEdge reduceGarbage(vEdge& e, const std::vector<bool>& garbage,
                      const bool normalizeWeights = false) {
    // return if no more garbage left
    if (!normalizeWeights && (std::none_of(garbage.begin(), garbage.end(),
                                           [](bool v) { return v; }) ||
                              e.isTerminal())) {
      return e;
    }
    Qubit lowerbound = 0;
    for (std::size_t i = 0U; i < garbage.size(); ++i) {
      if (garbage[i]) {
        lowerbound = static_cast<Qubit>(i);
        break;
      }
    }
    if (!normalizeWeights && e.p->v < lowerbound) {
      return e;
    }
    const auto f =
        reduceGarbageRecursion(e.p, garbage, lowerbound, normalizeWeights);
    auto weight = e.w * f.w;
    if (normalizeWeights) {
      weight = weight.mag();
    }
    const auto res = vEdge{f.p, cn.lookup(weight)};
    incRef(res);
    decRef(e);
    return res;
  }
  mEdge reduceGarbage(mEdge& e, const std::vector<bool>& garbage,
                      const bool regular = true,
                      const bool normalizeWeights = false) {
    // return if no more garbage left
    if (!normalizeWeights && (std::none_of(garbage.begin(), garbage.end(),
                                           [](bool v) { return v; }) ||
                              e.isZeroTerminal())) {
      return e;
    }

    // if we have only identities and no other nodes
    if (e.isIdentity()) {
      auto g = e;
      for (auto i = 0U; i < garbage.size(); ++i) {
        if (garbage[i]) {
          if (regular) {
            g = makeDDNode(static_cast<Qubit>(i),
                           std::array{g, g, mEdge::zero(), mEdge::zero()});
          } else {
            g = makeDDNode(static_cast<Qubit>(i),
                           std::array{g, mEdge::zero(), g, mEdge::zero()});
          }
        }
      }
      incRef(g);
      return g;
    }

    Qubit lowerbound = 0;
    for (auto i = 0U; i < garbage.size(); ++i) {
      if (garbage[i]) {
        lowerbound = static_cast<Qubit>(i);
        break;
      }
    }

    auto g = CachedEdge<mNode>{e.p, 1.};
    if (e.p->v >= lowerbound || normalizeWeights) {
      g = reduceGarbageRecursion(e.p, garbage, lowerbound, regular,
                                 normalizeWeights);
    }

    for (std::size_t i = e.p->v + 1; i < garbage.size(); ++i) {
      if (garbage[i]) {
        if (regular) {
          g = makeDDNode(
              static_cast<Qubit>(i),
              std::array{g, g, mCachedEdge::zero(), mCachedEdge::zero()});
        } else {
          g = makeDDNode(
              static_cast<Qubit>(i),
              std::array{g, mCachedEdge::zero(), g, mCachedEdge::zero()});
        }
      }
    }

    auto weight = g.w * e.w;
    if (normalizeWeights) {
      weight = weight.mag();
    }
    const auto res = mEdge{g.p, cn.lookup(weight)};

    incRef(res);
    decRef(e);
    return res;
  }

private:
  mCachedEdge reduceAncillaeRecursion(mNode* p,
                                      const std::vector<bool>& ancillary,
                                      const Qubit lowerbound,
                                      const bool regular = true) {
    if (p->v < lowerbound) {
      return {p, 1.};
    }

    std::array<mCachedEdge, NEDGE> edges{};
    std::bitset<NEDGE> handled{};
    for (auto i = 0U; i < NEDGE; ++i) {
      if (ancillary[p->v]) {
        // no need to reduce ancillaries for entries that will be zeroed anyway
        if ((i == 3) || (i == 1 && regular) || (i == 2 && !regular)) {
          continue;
        }
      }
      if (handled.test(i)) {
        continue;
      }

      if (p->e[i].isZeroTerminal()) {
        edges[i] = {p->e[i].p, p->e[i].w};
        handled.set(i);
        continue;
      }

      if (p->e[i].isIdentity()) {
        auto g = mCachedEdge::one();
        for (auto j = lowerbound; j < p->v; ++j) {
          if (ancillary[j]) {
            g = makeDDNode(j, std::array{g, mCachedEdge::zero(),
                                         mCachedEdge::zero(),
                                         mCachedEdge::zero()});
          }
        }
        edges[i] = {g.p, p->e[i].w};
        handled.set(i);
        continue;
      }

      edges[i] =
          reduceAncillaeRecursion(p->e[i].p, ancillary, lowerbound, regular);
      for (Qubit j = p->e[i].p->v + 1U; j < p->v; ++j) {
        if (ancillary[j]) {
          edges[i] = makeDDNode(j, std::array{edges[i], mCachedEdge::zero(),
                                              mCachedEdge::zero(),
                                              mCachedEdge::zero()});
        }
      }

      for (auto j = i + 1U; j < NEDGE; ++j) {
        if (p->e[i].p == p->e[j].p) {
          edges[j] = edges[i];
          edges[j].w = edges[j].w * p->e[j].w;
          handled.set(j);
        }
      }
      edges[i].w = edges[i].w * p->e[i].w;
      handled.set(i);
    }
    if (!ancillary[p->v]) {
      return makeDDNode(p->v, edges);
    }

    // something to reduce for this qubit
    if (regular) {
      return makeDDNode(p->v, std::array{edges[0], mCachedEdge::zero(),
                                         edges[2], mCachedEdge::zero()});
    }
    return makeDDNode(p->v, std::array{edges[0], edges[1], mCachedEdge::zero(),
                                       mCachedEdge::zero()});
  }

  vCachedEdge reduceGarbageRecursion(vNode* p, const std::vector<bool>& garbage,
                                     const Qubit lowerbound,
                                     const bool normalizeWeights = false) {
    if (!normalizeWeights && p->v < lowerbound) {
      return {p, 1.};
    }

    std::array<vCachedEdge, RADIX> edges{};
    std::bitset<RADIX> handled{};
    for (auto i = 0U; i < RADIX; ++i) {
      if (!handled.test(i)) {
        if (p->e[i].isTerminal()) {
          const auto weight = normalizeWeights
                                  ? ComplexNumbers::mag(p->e[i].w)
                                  : static_cast<ComplexValue>(p->e[i].w);
          edges[i] = {p->e[i].p, weight};
        } else {
          edges[i] = reduceGarbageRecursion(p->e[i].p, garbage, lowerbound,
                                            normalizeWeights);
          for (auto j = i + 1; j < RADIX; ++j) {
            if (p->e[i].p == p->e[j].p) {
              edges[j] = edges[i];
              edges[j].w = edges[j].w * p->e[j].w;
              if (normalizeWeights) {
                edges[j].w = edges[j].w.mag();
              }
              handled.set(j);
            }
          }
          edges[i].w = edges[i].w * p->e[i].w;
          if (normalizeWeights) {
            edges[i].w = edges[i].w.mag();
          }
        }
        handled.set(i);
      }
    }
    if (!garbage[p->v]) {
      return makeDDNode(p->v, edges);
    }
    // something to reduce for this qubit
    return makeDDNode(p->v,
                      std::array{addMagnitudes(edges[0], edges[1], p->v - 1),
                                 vCachedEdge ::zero()});
  }
  mCachedEdge reduceGarbageRecursion(mNode* p, const std::vector<bool>& garbage,
                                     const Qubit lowerbound,
                                     const bool regular = true,
                                     const bool normalizeWeights = false) {
    if (!normalizeWeights && p->v < lowerbound) {
      return {p, 1.};
    }

    std::array<mCachedEdge, NEDGE> edges{};
    std::bitset<NEDGE> handled{};
    for (auto i = 0U; i < NEDGE; ++i) {
      if (handled.test(i)) {
        continue;
      }

      if (p->e[i].isZeroTerminal()) {
        edges[i] = mCachedEdge::zero();
        handled.set(i);
        continue;
      }

      if (p->e[i].isIdentity()) {
        edges[i] = mCachedEdge::one();
        for (auto j = lowerbound; j < p->v; ++j) {
          if (garbage[j]) {
            if (regular) {
              edges[i] = makeDDNode(j, std::array{edges[i], edges[i],
                                                  mCachedEdge::zero(),
                                                  mCachedEdge::zero()});
            } else {
              edges[i] =
                  makeDDNode(j, std::array{edges[i], mCachedEdge::zero(),
                                           edges[i], mCachedEdge::zero()});
            }
          }
        }
        if (normalizeWeights) {
          edges[i].w = edges[i].w * ComplexNumbers::mag(p->e[i].w);
        } else {
          edges[i].w = edges[i].w * p->e[i].w;
        }
        handled.set(i);
        continue;
      }

      edges[i] = reduceGarbageRecursion(p->e[i].p, garbage, lowerbound, regular,
                                        normalizeWeights);
      for (Qubit j = p->e[i].p->v + 1U; j < p->v; ++j) {
        if (garbage[j]) {
          if (regular) {
            edges[i] = makeDDNode(j, std::array{edges[i], edges[i],
                                                mCachedEdge::zero(),
                                                mCachedEdge::zero()});
          } else {
            edges[i] = makeDDNode(j, std::array{edges[i], mCachedEdge::zero(),
                                                edges[i], mCachedEdge::zero()});
          }
        }
      }

      for (auto j = i + 1; j < NEDGE; ++j) {
        if (p->e[i].p == p->e[j].p) {
          edges[j] = edges[i];
          edges[j].w = edges[j].w * p->e[j].w;
          if (normalizeWeights) {
            edges[j].w = edges[j].w.mag();
          }
          handled.set(j);
        }
      }
      edges[i].w = edges[i].w * p->e[i].w;
      if (normalizeWeights) {
        edges[i].w = edges[i].w.mag();
      }
      handled.set(i);
    }
    if (!garbage[p->v]) {
      return makeDDNode(p->v, edges);
    }

    if (regular) {
      return makeDDNode(p->v,
                        std::array{addMagnitudes(edges[0], edges[2], p->v - 1),
                                   addMagnitudes(edges[1], edges[3], p->v - 1),
                                   mCachedEdge::zero(), mCachedEdge::zero()});
    }
    return makeDDNode(p->v,
                      std::array{addMagnitudes(edges[0], edges[1], p->v - 1),
                                 mCachedEdge::zero(),
                                 addMagnitudes(edges[2], edges[3], p->v - 1),
                                 mCachedEdge::zero()});
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
  template <class Node, std::size_t N = std::tuple_size_v<decltype(Node::e)>>
  CachedEdge<Node>
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
};

} // namespace dd
