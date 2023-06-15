#pragma once

#include "Complex.hpp"
#include "ComplexCache.hpp"
#include "ComplexNumbers.hpp"
#include "ComplexTable.hpp"
#include "ComplexValue.hpp"
#include "ComputeTable.hpp"
#include "Control.hpp"
#include "Definitions.hpp"
#include "DensityNoiseTable.hpp"
#include "Edge.hpp"
#include "GateMatrixDefinitions.hpp"
#include "Node.hpp"
#include "StochasticNoiseOperationTable.hpp"
#include "ToffoliTable.hpp"
#include "UnaryComputeTable.hpp"
#include "UniqueTable.hpp"

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
struct DDPackageConfig {
  // Note the order of parameters here must be the *same* as in the template
  // definition.
  static constexpr std::size_t UT_VEC_NBUCKET = 32768U;
  static constexpr std::size_t UT_VEC_INITIAL_ALLOCATION_SIZE = 2048U;
  static constexpr std::size_t UT_MAT_NBUCKET = 32768U;
  static constexpr std::size_t UT_MAT_INITIAL_ALLOCATION_SIZE = 2048U;
  static constexpr std::size_t CT_VEC_ADD_NBUCKET = 16384U;
  static constexpr std::size_t CT_MAT_ADD_NBUCKET = 16384U;
  static constexpr std::size_t CT_MAT_TRANS_NBUCKET = 4096U;
  static constexpr std::size_t CT_MAT_CONJ_TRANS_NBUCKET = 4096U;
  static constexpr std::size_t CT_MAT_VEC_MULT_NBUCKET = 16384U;
  static constexpr std::size_t CT_MAT_MAT_MULT_NBUCKET = 16384U;
  static constexpr std::size_t CT_VEC_KRON_NBUCKET = 4096U;
  static constexpr std::size_t CT_MAT_KRON_NBUCKET = 4096U;
  static constexpr std::size_t CT_VEC_INNER_PROD_NBUCKET = 4096U;
  static constexpr std::size_t CT_DM_NOISE_NBUCKET = 1U;
  static constexpr std::size_t UT_DM_NBUCKET = 1U;
  static constexpr std::size_t UT_DM_INITIAL_ALLOCATION_SIZE = 1U;
  static constexpr std::size_t CT_DM_DM_MULT_NBUCKET = 1U;
  static constexpr std::size_t CT_DM_ADD_NBUCKET = 1U;

  // The number of different quantum operations. I.e., the number of operations
  // defined in the QFR OpType.hpp This parameter is required to initialize the
  // StochasticNoiseOperationTable.hpp
  static constexpr std::size_t STOCHASTIC_CACHE_OPS = 1;
};

template <class Config = DDPackageConfig> class Package {
  static_assert(std::is_base_of_v<DDPackageConfig, Config>,
                "Config must be derived from DDPackageConfig");
  ///
  /// Complex number handling
  ///
public:
  ComplexNumbers cn{};

  ///
  /// Construction, destruction, information and reset
  ///

  static constexpr std::size_t MAX_POSSIBLE_QUBITS =
      static_cast<std::make_unsigned_t<Qubit>>(
          std::numeric_limits<Qubit>::max()) +
      1U;
  static constexpr std::size_t DEFAULT_QUBITS = 128;
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
    clearComputeTables();
    cn.clear();
  }

  // getter for qubits
  [[nodiscard]] auto qubits() const { return nqubits; }

private:
  std::size_t nqubits;

  ///
  /// Vector nodes, edges and quantum states
  ///
public:
  vEdge normalize(const vEdge& e, bool cached) {
    auto zero = std::array{e.p->e[0].w.approximatelyZero(),
                           e.p->e[1].w.approximatelyZero()};

    // make sure to release cached numbers approximately zero, but not exactly
    // zero
    if (cached) {
      for (auto i = 0U; i < RADIX; i++) {
        if (zero[i] && e.p->e[i].w != Complex::zero) {
          cn.returnToCache(e.p->e[i].w);
          e.p->e[i] = vEdge::zero;
        }
      }
    }

    if (zero[0]) {
      // all equal to zero
      if (zero[1]) {
        if (!cached && !e.isTerminal()) {
          // If it is not a cached computation, the node has to be put back into
          // the chain
          vUniqueTable.returnNode(e.p);
        }
        return vEdge::zero;
      }

      auto r = e;
      auto& w = r.p->e[1].w;
      if (cached && !w.exactlyOne()) {
        r.w = w;
      } else {
        r.w = cn.lookup(w);
      }
      w = Complex::one;
      return r;
    }

    if (zero[1]) {
      auto r = e;
      auto& w = r.p->e[0].w;
      if (cached && !w.exactlyOne()) {
        r.w = w;
      } else {
        r.w = cn.lookup(w);
      }
      w = Complex::one;
      return r;
    }

    const auto mag0 = ComplexNumbers::mag2(e.p->e[0].w);
    const auto mag1 = ComplexNumbers::mag2(e.p->e[1].w);
    const auto norm2 = mag0 + mag1;
    const auto mag2Max =
        (mag0 + ComplexTable<>::tolerance() >= mag1) ? mag0 : mag1;
    const auto argMax = (mag0 + ComplexTable<>::tolerance() >= mag1) ? 0 : 1;
    const auto norm = std::sqrt(norm2);
    const auto magMax = std::sqrt(mag2Max);
    const auto commonFactor = norm / magMax;

    auto r = e;
    auto& max = r.p->e[static_cast<std::size_t>(argMax)];
    if (cached && !max.w.exactlyOne()) {
      r.w = max.w;
      r.w.r->value *= commonFactor;
      r.w.i->value *= commonFactor;
    } else {
      r.w = cn.lookup(CTEntry::val(max.w.r) * commonFactor,
                      CTEntry::val(max.w.i) * commonFactor);
      if (r.w.approximatelyZero()) {
        return vEdge::zero;
      }
    }

    max.w = cn.lookup(magMax / norm, 0.);
    if (max.w == Complex::zero) {
      max = vEdge::zero;
    }

    const auto argMin = (argMax + 1) % 2;
    auto& min = r.p->e[static_cast<std::size_t>(argMin)];
    if (cached) {
      cn.returnToCache(min.w);
      ComplexNumbers::div(min.w, min.w, r.w);
      min.w = cn.lookup(min.w);
    } else {
      auto c = cn.getTemporary();
      ComplexNumbers::div(c, min.w, r.w);
      min.w = cn.lookup(c);
    }
    if (min.w == Complex::zero) {
      min = vEdge::zero;
    }

    return r;
  }

  dEdge makeZeroDensityOperator(QubitCount n) {
    auto f = dEdge::one;
    for (std::size_t p = 0; p < n; p++) {
      f = makeDDNode(static_cast<Qubit>(p),
                     std::array{f, dEdge::zero, dEdge::zero, dEdge::zero});
    }
    return f;
  }

  // generate |0...0> with n qubits
  vEdge makeZeroState(QubitCount n, std::size_t start = 0) {
    if (n + start > nqubits) {
      throw std::runtime_error{
          "Requested state with " + std::to_string(n + start) +
          " qubits, but current package configuration only supports up to " +
          std::to_string(nqubits) +
          " qubits. Please allocate a larger package instance."};
    }
    auto f = vEdge::one;
    for (std::size_t p = start; p < n + start; p++) {
      f = makeDDNode(static_cast<Qubit>(p), std::array{f, vEdge::zero});
    }
    return f;
  }
  // generate computational basis state |i> with n qubits
  vEdge makeBasisState(QubitCount n, const std::vector<bool>& state,
                       std::size_t start = 0) {
    if (n + start > nqubits) {
      throw std::runtime_error{
          "Requested state with " + std::to_string(n + start) +
          " qubits, but current package configuration only supports up to " +
          std::to_string(nqubits) +
          " qubits. Please allocate a larger package instance."};
    }
    auto f = vEdge::one;
    for (std::size_t p = start; p < n + start; ++p) {
      if (!state[p]) {
        f = makeDDNode(static_cast<Qubit>(p), std::array{f, vEdge::zero});
      } else {
        f = makeDDNode(static_cast<Qubit>(p), std::array{vEdge::zero, f});
      }
    }
    return f;
  }
  // generate general basis state with n qubits
  vEdge makeBasisState(QubitCount n, const std::vector<BasisStates>& state,
                       std::size_t start = 0) {
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

    auto f = vEdge::one;
    for (std::size_t p = start; p < n + start; ++p) {
      switch (state[p]) {
      case BasisStates::zero:
        f = makeDDNode(static_cast<Qubit>(p), std::array{f, vEdge::zero});
        break;
      case BasisStates::one:
        f = makeDDNode(static_cast<Qubit>(p), std::array{vEdge::zero, f});
        break;
      case BasisStates::plus:
        f = makeDDNode(
            static_cast<Qubit>(p),
            std::array<vEdge, RADIX>{{{f.p, cn.lookup(dd::SQRT2_2, 0)},
                                      {f.p, cn.lookup(dd::SQRT2_2, 0)}}});
        break;
      case BasisStates::minus:
        f = makeDDNode(
            static_cast<Qubit>(p),
            std::array<vEdge, RADIX>{{{f.p, cn.lookup(dd::SQRT2_2, 0)},
                                      {f.p, cn.lookup(-dd::SQRT2_2, 0)}}});
        break;
      case BasisStates::right:
        f = makeDDNode(
            static_cast<Qubit>(p),
            std::array<vEdge, RADIX>{{{f.p, cn.lookup(dd::SQRT2_2, 0)},
                                      {f.p, cn.lookup(0, dd::SQRT2_2)}}});
        break;
      case BasisStates::left:
        f = makeDDNode(
            static_cast<Qubit>(p),
            std::array<vEdge, RADIX>{{{f.p, cn.lookup(dd::SQRT2_2, 0)},
                                      {f.p, cn.lookup(0, -dd::SQRT2_2)}}});
        break;
      }
    }
    return f;
  }

  // generate the decision diagram from an arbitrary state vector
  vEdge makeStateFromVector(const CVec& stateVector) {
    if (stateVector.empty()) {
      return vEdge::one;
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
    if (state.w != Complex::zero) {
      cn.returnToCache(state.w);
      state.w = cn.lookup(state.w);
    }

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
      return mEdge::one;
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

    if (matrixDD.w != Complex::zero) {
      cn.returnToCache(matrixDD.w);
      matrixDD.w = cn.lookup(matrixDD.w);
    }

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
          if (zero[i] && e.p->e[i].w != Complex::zero) {
            cn.returnToCache(e.p->e[i].w);
            e.p->e[i] = Edge<Node>::zero;
          }
        }
      }

      fp max = 0;
      auto maxc = Complex::one;
      // determine max amplitude
      for (auto i = 0U; i < NEDGE; ++i) {
        if (zero[i]) {
          continue;
        }
        if (argmax == -1) {
          argmax = static_cast<decltype(argmax)>(i);
          max = ComplexNumbers::mag2(e.p->e[i].w);
          maxc = e.p->e[i].w;
        } else {
          auto mag = ComplexNumbers::mag2(e.p->e[i].w);
          if (mag - max > ComplexTable<>::tolerance()) {
            argmax = static_cast<decltype(argmax)>(i);
            max = mag;
            maxc = e.p->e[i].w;
          }
        }
      }

      // all equal to zero
      if (argmax == -1) {
        if (!cached && !e.isTerminal()) {
          // If it is not a cached computation, the node has to be put back into
          // the chain
          getUniqueTable<Node>().returnNode(e.p);
        }
        return Edge<Node>::zero;
      }

      auto r = e;
      // divide each entry by max
      for (auto i = 0U; i < NEDGE; ++i) {
        if (static_cast<decltype(argmax)>(i) == argmax) {
          if (cached) {
            if (r.w.exactlyOne()) {
              r.w = maxc;
            } else {
              ComplexNumbers::mul(r.w, r.w, maxc);
            }
          } else {
            if (r.w.exactlyOne()) {
              r.w = maxc;
            } else {
              auto c = cn.getTemporary();
              ComplexNumbers::mul(c, r.w, maxc);
              r.w = cn.lookup(c);
            }
          }
          r.p->e[i].w = Complex::one;
        } else {
          if (zero[i]) {
            if (cached && r.p->e[i].w != Complex::zero) {
              cn.returnToCache(r.p->e[i].w);
            }
            r.p->e[i] = Edge<Node>::zero;
            continue;
          }
          if (cached && !zero[i] && !r.p->e[i].w.exactlyOne()) {
            cn.returnToCache(r.p->e[i].w);
          }
          if (r.p->e[i].w.approximatelyOne()) {
            r.p->e[i].w = Complex::one;
          }
          auto c = cn.getTemporary();
          ComplexNumbers::div(c, r.p->e[i].w, maxc);
          r.p->e[i].w = cn.lookup(c);
        }
      }
      return r;
    }
  }

  // build matrix representation for a single gate on an n-qubit circuit
  mEdge makeGateDD(const std::array<ComplexValue, NEDGE>& mat, QubitCount n,
                   Qubit target, std::size_t start = 0) {
    return makeGateDD(mat, n, Controls{}, target, start);
  }
  mEdge makeGateDD(const std::array<ComplexValue, NEDGE>& mat, QubitCount n,
                   const Control& control, Qubit target,
                   std::size_t start = 0) {
    return makeGateDD(mat, n, Controls{control}, target, start);
  }
  mEdge makeGateDD(const std::array<ComplexValue, NEDGE>& mat, QubitCount n,
                   const Controls& controls, Qubit target,
                   std::size_t start = 0) {
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
      // NOLINTNEXTLINE(clang-diagnostic-float-equal) it has to be really zero
      if (mat[i].r == 0 && mat[i].i == 0) {
        em[i] = mEdge::zero;
      } else {
        em[i] = mEdge::terminal(cn.lookup(mat[i]));
      }
    }

    // process lines below target
    auto z = static_cast<Qubit>(start);
    for (; z < target; z++) {
      for (auto i1 = 0U; i1 < RADIX; i1++) {
        for (auto i2 = 0U; i2 < RADIX; i2++) {
          auto i = i1 * RADIX + i2;
          if (it != controls.end() && it->qubit == z) {
            if (it->type == Control::Type::neg) { // neg. control
              em[i] = makeDDNode(
                  z,
                  std::array{em[i], mEdge::zero, mEdge::zero,
                             (i1 == i2) ? makeIdent(static_cast<Qubit>(start),
                                                    static_cast<Qubit>(z - 1))
                                        : mEdge::zero});
            } else { // pos. control
              em[i] = makeDDNode(
                  z,
                  std::array{(i1 == i2) ? makeIdent(static_cast<Qubit>(start),
                                                    static_cast<Qubit>(z - 1))
                                        : mEdge::zero,
                             mEdge::zero, mEdge::zero, em[i]});
            }
          } else { // not connected
            em[i] = makeDDNode(
                z, std::array{em[i], mEdge::zero, mEdge::zero, em[i]});
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
      if (it != controls.end() && it->qubit == q) {
        if (it->type == Control::Type::neg) { // neg. control
          e = makeDDNode(q, std::array{e, mEdge::zero, mEdge::zero,
                                       makeIdent(static_cast<Qubit>(start),
                                                 static_cast<Qubit>(q - 1))});
        } else { // pos. control
          e = makeDDNode(q, std::array{makeIdent(static_cast<Qubit>(start),
                                                 static_cast<Qubit>(q - 1)),
                                       mEdge::zero, mEdge::zero, e});
        }
        ++it;
      } else { // not connected
        e = makeDDNode(q, std::array{e, mEdge::zero, mEdge::zero, e});
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
  mEdge makeTwoQubitGateDD(
      const std::array<std::array<ComplexValue, NEDGE>, NEDGE>& mat,
      const QubitCount n, const Qubit target0, const Qubit target1,
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
        const auto& matEntry = matRow.at(i2);
        auto& emEntry = emRow.at(i2);
        // NOLINTNEXTLINE(clang-diagnostic-float-equal) it has to be really zero
        if (matEntry.r == 0 && matEntry.i == 0) {
          emEntry = mEdge::zero;
        } else {
          emEntry = mEdge::terminal(cn.lookup(matEntry));
        }
      }
    }

    // process lines below smaller target (by creating identity structures)
    auto z = static_cast<Qubit>(start);
    const auto smallerTarget = std::min(target0, target1);
    for (; z < smallerTarget; ++z) {
      for (auto& row : em) {
        for (auto& entry : row) {
          entry =
              makeDDNode(z, std::array{entry, mEdge::zero, mEdge::zero, entry});
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
        entry =
            makeDDNode(z, std::array{entry, mEdge::zero, mEdge::zero, entry});
      }
    }

    // process the larger target by combining the four DDs from the smaller
    // target
    auto e = makeDDNode(z, em0);

    // process lines above the larger target (by creating identity structures)
    for (++z; z < static_cast<Qubit>(n + start); ++z) {
      e = makeDDNode(z, std::array{e, mEdge::zero, mEdge::zero, e});
    }

    return e;
  }

  mEdge makeSWAPDD(const QubitCount n, const Qubit target0, const Qubit target1,
                   const std::size_t start = 0) {
    return makeTwoQubitGateDD(SWAPmat, n, target0, target1, start);
  }
  mEdge makeSWAPDD(const QubitCount n, const Controls& controls,
                   const Qubit target0, const Qubit target1,
                   const std::size_t start = 0) {
    auto c = controls;
    c.insert(Control{target0});
    mEdge e = makeGateDD(Xmat, n, c, target1, start);
    c.erase(Control{target0});
    c.insert(Control{target1});
    e = multiply(e, multiply(makeGateDD(Xmat, n, c, target0, start), e));
    return e;
  }

  mEdge makePeresDD(const QubitCount n, const Controls& controls,
                    const Qubit target0, const Qubit target1,
                    const std::size_t start = 0) {
    auto c = controls;
    c.insert(Control{target1});
    mEdge e = makeGateDD(Xmat, n, c, target0, start);
    e = multiply(makeGateDD(Xmat, n, controls, target1, start), e);
    return e;
  }

  mEdge makePeresdagDD(const QubitCount n, const Controls& controls,
                       const Qubit target0, const Qubit target1,
                       const std::size_t start = 0) {
    mEdge e = makeGateDD(Xmat, n, controls, target1, start);
    auto c = controls;
    c.insert(Control{target1});
    e = multiply(makeGateDD(Xmat, n, c, target0, start), e);
    return e;
  }

  mEdge makeiSWAPDD(const QubitCount n, const Qubit target0,
                    const Qubit target1, const std::size_t start = 0) {
    return makeTwoQubitGateDD(iSWAPmat, n, target0, target1, start);
  }
  mEdge makeiSWAPDD(const QubitCount n, const Controls& controls,
                    const Qubit target0, const Qubit target1,
                    const std::size_t start = 0) {
    mEdge e = makeGateDD(Smat, n, controls, target1, start);        // S q[1]
    e = multiply(e, makeGateDD(Smat, n, controls, target0, start)); // S q[0]
    e = multiply(e, makeGateDD(Hmat, n, controls, target0, start)); // H q[0]
    auto c = controls;
    c.insert(Control{target0});
    e = multiply(e, makeGateDD(Xmat, n, c, target1, start)); // CX q[0], q[1]
    c.erase(Control{target0});
    c.insert(Control{target1});
    e = multiply(e, makeGateDD(Xmat, n, c, target0, start)); // CX q[1], q[0]
    e = multiply(e, makeGateDD(Hmat, n, controls, target1, start)); // H q[1]
    return e;
  }

  mEdge makeiSWAPinvDD(const QubitCount n, const Qubit target0,
                       const Qubit target1, const std::size_t start = 0) {
    return makeTwoQubitGateDD(iSWAPinvmat, n, target0, target1, start);
  }
  mEdge makeiSWAPinvDD(const QubitCount n, const Controls& controls,
                       const Qubit target0, const Qubit target1,
                       const std::size_t start = 0) {
    mEdge e = makeGateDD(Hmat, n, controls, target1, start); // H q[1]
    auto c = controls;
    c.insert(Control{target1});
    e = multiply(e, makeGateDD(Xmat, n, c, target0, start)); // CX q[1], q[0]
    c.erase(Control{target1});
    c.insert(Control{target0});
    e = multiply(e, makeGateDD(Xmat, n, c, target1, start)); // CX q[0], q[1]
    e = multiply(e, makeGateDD(Hmat, n, controls, target0, start)); // H q[0]
    e = multiply(e,
                 makeGateDD(Sdagmat, n, controls, target0, start)); // Sdag q[0]
    e = multiply(e,
                 makeGateDD(Sdagmat, n, controls, target1, start)); // Sdag q[1]
    return e;
  }

  mEdge makeDCXDD(const QubitCount n, const Qubit target0, const Qubit target1,
                  const std::size_t start = 0) {
    return makeTwoQubitGateDD(DCXmat, n, target0, target1, start);
  }
  mEdge makeDCXDD(const QubitCount n, const Controls& controls,
                  const Qubit target0, const Qubit target1,
                  const std::size_t start = 0) {
    auto c = controls;
    c.insert(Control{target0});
    mEdge e = makeGateDD(Xmat, n, c, target1, start);
    c.erase(Control{target0});
    c.insert(Control{target1});
    e = multiply(e, makeGateDD(Xmat, n, c, target0, start));
    return e;
  }

  mEdge makeRZZDD(const QubitCount n, const Qubit target0, const Qubit target1,
                  const fp theta, const std::size_t start = 0) {
    return makeTwoQubitGateDD(RZZmat(theta), n, target0, target1, start);
  }
  mEdge makeRZZDD(const QubitCount n, const Controls& controls,
                  const Qubit target0, const Qubit target1, const fp theta,
                  const std::size_t start = 0) {
    auto c = controls;
    c.insert(Control{target0});
    auto e = makeGateDD(Xmat, n, c, target1, start);
    c.erase(Control{target0});
    e = multiply(e, makeGateDD(RZmat(theta), n, c, target1, start));
    c.insert(Control{target0});
    e = multiply(e, makeGateDD(Xmat, n, c, target1, start));
    return e;
  }

  mEdge makeRYYDD(const QubitCount n, const Qubit target0, const Qubit target1,
                  const fp theta, const std::size_t start = 0) {
    return makeTwoQubitGateDD(RYYmat(theta), n, target0, target1, start);
  }
  mEdge makeRYYDD(const QubitCount n, const Controls& controls,
                  const Qubit target0, const Qubit target1, const fp theta,
                  const std::size_t start = 0) {
    // no controls are necessary on the RX gates since they cancel if the
    // controls are 0.
    auto e = makeGateDD(RXmat(PI_2), n, Controls{}, target0, start);
    e = multiply(e, makeGateDD(RXmat(PI_2), n, Controls{}, target1, start));
    e = multiply(e, makeRZZDD(n, controls, target0, target1, theta, start));
    e = multiply(e, makeGateDD(RXmat(-PI_2), n, Controls{}, target1, start));
    e = multiply(e, makeGateDD(RXmat(-PI_2), n, Controls{}, target0, start));
    return e;
  }

  mEdge makeRXXDD(const QubitCount n, const Qubit target0, const Qubit target1,
                  const fp theta, const std::size_t start = 0) {
    return makeTwoQubitGateDD(RXXmat(theta), n, target0, target1, start);
  }
  mEdge makeRXXDD(const QubitCount n, const Controls& controls,
                  const Qubit target0, const Qubit target1, const fp theta,
                  const std::size_t start = 0) {
    // no controls are necessary on the H gates since they cancel if the
    // controls are 0.
    auto e = makeGateDD(Hmat, n, Controls{}, target0, start);
    e = multiply(e, makeGateDD(Hmat, n, Controls{}, target1, start));
    e = multiply(e, makeRZZDD(n, controls, target0, target1, theta, start));
    e = multiply(e, makeGateDD(Hmat, n, Controls{}, target1, start));
    e = multiply(e, makeGateDD(Hmat, n, Controls{}, target0, start));
    return e;
  }

  mEdge makeRZXDD(const QubitCount n, const Qubit target0, const Qubit target1,
                  const fp theta, const std::size_t start = 0) {
    return makeTwoQubitGateDD(RZXmat(theta), n, target0, target1, start);
  }
  mEdge makeRZXDD(const QubitCount n, const Controls& controls,
                  const Qubit target0, const Qubit target1, const fp theta,
                  const std::size_t start = 0) {
    // no controls are necessary on the H gates since they cancel if the
    // controls are 0.
    auto e = makeGateDD(Hmat, n, Controls{}, target1, start);
    e = multiply(e, makeRZZDD(n, controls, target0, target1, theta, start));
    e = multiply(e, makeGateDD(Hmat, n, Controls{}, target1, start));
    return e;
  }

  mEdge makeECRDD(const QubitCount n, const Qubit target0, const Qubit target1,
                  const std::size_t start = 0) {
    return makeTwoQubitGateDD(ECRmat, n, target0, target1, start);
  }
  mEdge makeECRDD(const QubitCount n, const Controls& controls,
                  const Qubit target0, const Qubit target1,
                  const std::size_t start = 0) {
    auto e = makeRZXDD(n, controls, target0, target1, -PI_4, start);
    e = multiply(e, makeGateDD(Xmat, n, controls, target0, start));
    e = multiply(e, makeRZXDD(n, controls, target0, target1, PI_4, start));
    return e;
  }

  mEdge makeXXMinusYYDD(const QubitCount n, const Qubit target0,
                        const Qubit target1, const fp theta, const fp beta = 0.,
                        const std::size_t start = 0) {
    return makeTwoQubitGateDD(XXMinusYYmat(theta, beta), n, target0, target1,
                              start);
  }
  mEdge makeXXMinusYYDD(const QubitCount n, const Controls& controls,
                        const Qubit target0, const Qubit target1,
                        const fp theta, const fp beta = 0.,
                        const std::size_t start = 0) {
    auto e = makeGateDD(RZmat(-beta), n, Controls{}, target1, start);
    e = multiply(e, makeGateDD(RZmat(-PI_2), n, Controls{}, target0, start));
    e = multiply(e, makeGateDD(SXmat, n, Controls{}, target0, start));
    e = multiply(e, makeGateDD(RZmat(PI_2), n, Controls{}, target0, start));
    e = multiply(e, makeGateDD(Smat, n, Controls{}, target1, start));
    e = multiply(e, makeGateDD(Xmat, n, Control{target0}, target1, start));
    // only the following two gates need to be controlled by the controls since
    // the other gates cancel if the controls are 0.
    e = multiply(e,
                 makeGateDD(RYmat(-theta / 2.), n, controls, target0, start));
    e = multiply(e, makeGateDD(RYmat(theta / 2.), n, controls, target1, start));

    e = multiply(e, makeGateDD(Xmat, n, Control{target0}, target1, start));
    e = multiply(e, makeGateDD(Sdagmat, n, Controls{}, target1, start));
    e = multiply(e, makeGateDD(RZmat(-PI_2), n, Controls{}, target0, start));
    e = multiply(e, makeGateDD(SXdagmat, n, Controls{}, target0, start));
    e = multiply(e, makeGateDD(RZmat(PI_2), n, Controls{}, target0, start));
    e = multiply(e, makeGateDD(RZmat(beta), n, Controls{}, target1, start));
    return e;
  }

  mEdge makeXXPlusYYDD(const QubitCount n, const Qubit target0,
                       const Qubit target1, const fp theta, const fp beta = 0.,
                       const std::size_t start = 0) {
    return makeTwoQubitGateDD(XXPlusYYmat(theta, beta), n, target0, target1,
                              start);
  }
  mEdge makeXXPlusYYDD(const QubitCount n, const Controls& controls,
                       const Qubit target0, const Qubit target1, const fp theta,
                       const fp beta = 0., const std::size_t start = 0) {
    auto e = makeGateDD(RZmat(beta), n, Controls{}, target1, start);
    e = multiply(e, makeGateDD(RZmat(-PI_2), n, Controls{}, target0, start));
    e = multiply(e, makeGateDD(SXmat, n, Controls{}, target0, start));
    e = multiply(e, makeGateDD(RZmat(PI_2), n, Controls{}, target0, start));
    e = multiply(e, makeGateDD(Smat, n, Controls{}, target1, start));
    e = multiply(e, makeGateDD(Xmat, n, Control{target0}, target1, start));
    // only the following two gates need to be controlled by the controls since
    // the other gates cancel if the controls are 0.
    e = multiply(e, makeGateDD(RYmat(theta / 2.), n, controls, target0, start));
    e = multiply(e, makeGateDD(RYmat(theta / 2.), n, controls, target1, start));

    e = multiply(e, makeGateDD(Xmat, n, Control{target0}, target1, start));
    e = multiply(e, makeGateDD(Sdagmat, n, Controls{}, target1, start));
    e = multiply(e, makeGateDD(RZmat(-PI_2), n, Controls{}, target0, start));
    e = multiply(e, makeGateDD(SXdagmat, n, Controls{}, target0, start));
    e = multiply(e, makeGateDD(RZmat(PI_2), n, Controls{}, target0, start));
    e = multiply(e, makeGateDD(RZmat(-beta), n, Controls{}, target1, start));
    return e;
  }

private:
  // check whether node represents a symmetric matrix or the identity
  void checkSpecialMatrices(mNode* p) {
    if (p->v == -1) {
      return;
    }

    p->setIdentity(false);
    p->setSymmetric(false);

    // check if matrix is symmetric
    if (!p->e[0].p->isSymmetric() || !p->e[3].p->isSymmetric()) {
      return;
    }
    if (transpose(p->e[1]) != p->e[2]) {
      return;
    }
    p->setSymmetric(true);

    // check if matrix resembles identity
    if (!(p->e[0].p->isIdentity()) || (p->e[1].w) != Complex::zero ||
        (p->e[2].w) != Complex::zero || (p->e[0].w) != Complex::one ||
        (p->e[3].w) != Complex::one || !(p->e[3].p->isIdentity())) {
      return;
    }
    p->setIdentity(true);
  }

  vEdge makeStateFromVector(const CVec::const_iterator& begin,
                            const CVec::const_iterator& end,
                            const Qubit level) {
    if (level == 0) {
      assert(std::distance(begin, end) == 2);
      const auto& zeroWeight = cn.getCached(begin->real(), begin->imag());
      const auto& oneWeight =
          cn.getCached(std::next(begin)->real(), std::next(begin)->imag());
      const auto zeroSuccessor = vEdge{vNode::terminal, zeroWeight};
      const auto oneSuccessor = vEdge{vNode::terminal, oneWeight};
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
  @throw std::invalid_argument If level is negative.
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
    if (level == -1) {
      assert(rowEnd - rowStart == 1);
      assert(colEnd - colStart == 1);
      return {mNode::terminal, cn.getCached(matrix[rowStart][colStart])};
    }

    // recursively call the function on all quadrants
    const auto rowMid = (rowStart + rowEnd) / 2;
    const auto colMid = (colStart + colEnd) / 2;

    const auto edge0 =
        makeDDFromMatrix(matrix, level - 1, rowStart, rowMid, colStart, colMid);
    const auto edge1 =
        makeDDFromMatrix(matrix, level - 1, rowStart, rowMid, colMid, colEnd);
    const auto edge2 =
        makeDDFromMatrix(matrix, level - 1, rowMid, rowEnd, colStart, colMid);
    const auto edge3 =
        makeDDFromMatrix(matrix, level - 1, rowMid, rowEnd, colMid, colEnd);

    return makeDDNode<mNode>(level, {edge0, edge1, edge2, edge3}, true);
  }

  ///
  /// Unique tables, Reference counting and garbage collection
  ///
public:
  // unique tables
  template <class Node> [[nodiscard]] auto& getUniqueTable() {
    if constexpr (std::is_same_v<Node, vNode>) {
      return vUniqueTable;
    } else if constexpr (std::is_same_v<Node, mNode>) {
      return mUniqueTable;
    } else if constexpr (std::is_same_v<Node, dNode>) {
      return dUniqueTable;
    }
  }

  template <class Node> void incRef(const Edge<Node>& e) {
    getUniqueTable<Node>().incRef(e);
  }
  template <class Node> void decRef(const Edge<Node>& e) {
    getUniqueTable<Node>().decRef(e);
  }

  UniqueTable<vNode, Config::UT_VEC_NBUCKET,
              Config::UT_VEC_INITIAL_ALLOCATION_SIZE>
      vUniqueTable{nqubits};
  UniqueTable<mNode, Config::UT_MAT_NBUCKET,
              Config::UT_MAT_INITIAL_ALLOCATION_SIZE>
      mUniqueTable{nqubits};
  UniqueTable<dNode, Config::UT_DM_NBUCKET,
              Config::UT_DM_INITIAL_ALLOCATION_SIZE>
      dUniqueTable{nqubits};

  bool garbageCollect(bool force = false) {
    // return immediately if no table needs collection
    if (!force && !vUniqueTable.possiblyNeedsCollection() &&
        !mUniqueTable.possiblyNeedsCollection() &&
        !dUniqueTable.possiblyNeedsCollection() &&
        !cn.complexTable.possiblyNeedsCollection()) {
      return false;
    }

    auto cCollect = cn.garbageCollect(force);
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
      matrixTranspose.clear();
      conjugateMatrixTranspose.clear();
      matrixKronecker.clear();
      matrixVectorMultiplication.clear();
      matrixMatrixMultiplication.clear();
      toffoliTable.clear();
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
      matrixTranspose.clear();
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

  void clearUniqueTables() {
    vUniqueTable.clear();
    mUniqueTable.clear();
    dUniqueTable.clear();
  }

  // create a normalized DD node and return an edge pointing to it. The node is
  // not recreated if it already exists.
  template <class Node>
  Edge<Node> makeDDNode(
      Qubit var,
      const std::array<Edge<Node>, std::tuple_size_v<decltype(Node::e)>>& edges,
      bool cached = false,
      [[maybe_unused]] const bool generateDensityMatrix = false) {
    auto& uniqueTable = getUniqueTable<Node>();
    Edge<Node> e{uniqueTable.getNode(), Complex::one};
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
      assert(edge.p->v == var - 1 || edge.isTerminal());
    }

    // normalize it
    e = normalize(e, cached);
    assert(e.p->v == var || e.isTerminal());

    // look it up in the unique tables
    auto l = uniqueTable.lookup(e, false);
    assert(l.p->v == var || l.isTerminal());

    // set specific node properties for matrices
    if constexpr (std::is_same_v<Node, mNode>) {
      if (l.p == e.p) {
        checkSpecialMatrices(l.p);
      }
    }
    return l;
  }

  template <class Node>
  Edge<Node> deleteEdge(const Edge<Node>& e, dd::Qubit v, std::size_t edgeIdx) {
    std::unordered_map<Node*, Edge<Node>> nodes{};
    return deleteEdge(e, v, edgeIdx, nodes);
  }

private:
  template <class Node>
  Edge<Node> deleteEdge(const Edge<Node>& e, dd::Qubit v, std::size_t edgeIdx,
                        std::unordered_map<Node*, Edge<Node>>& nodes) {
    if (e.p == nullptr || e.isTerminal()) {
      return e;
    }

    const auto& nodeit = nodes.find(e.p);
    Edge<Node> newedge{};
    if (nodeit != nodes.end()) {
      newedge = nodeit->second;
    } else {
      constexpr std::size_t n = std::tuple_size_v<decltype(e.p->e)>;
      std::array<Edge<Node>, n> edges{};
      if (e.p->v == v) {
        for (std::size_t i = 0; i < n; i++) {
          edges[i] = i == edgeIdx
                         ? Edge<Node>::zero
                         : e.p->e[i]; // optimization -> node cannot occur below
                                      // again, since dd is assumed to be free
        }
      } else {
        for (std::size_t i = 0; i < n; i++) {
          edges[i] = deleteEdge(e.p->e[i], v, edgeIdx, nodes);
        }
      }

      newedge = makeDDNode(e.p->v, edges);
      nodes[e.p] = newedge;
    }

    if (newedge.w.approximatelyOne()) {
      newedge.w = e.w;
    } else {
      auto w = cn.getTemporary();
      dd::ComplexNumbers::mul(w, newedge.w, e.w);
      newedge.w = cn.lookup(w);
    }

    return newedge;
  }

  ///
  /// Compute table definitions
  ///
public:
  void clearComputeTables() {
    vectorAdd.clear();
    matrixAdd.clear();
    matrixTranspose.clear();
    conjugateMatrixTranspose.clear();
    matrixMatrixMultiplication.clear();
    matrixVectorMultiplication.clear();
    vectorInnerProduct.clear();
    vectorKronecker.clear();
    matrixKronecker.clear();

    toffoliTable.clear();

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
                         std::mt19937_64& mt, fp epsilon = 0.001) {
    if (std::abs(ComplexNumbers::mag2(rootEdge.w) - 1.0) > epsilon) {
      if (rootEdge.w.approximatelyZero()) {
        throw std::runtime_error(
            "Numerical instabilities led to a 0-vector! Abort simulation!");
      }
      std::cerr << "WARNING in MAll: numerical instability occurred during "
                   "simulation: |alpha|^2 + |beta|^2 = "
                << ComplexNumbers::mag2(rootEdge.w) << ", but should be 1!\n";
    }

    vEdge cur = rootEdge;
    const auto numberOfQubits = static_cast<QubitCount>(rootEdge.p->v + 1);

    std::string result(numberOfQubits, '0');

    std::uniform_real_distribution<fp> dist(0.0, 1.0L);

    for (Qubit i = rootEdge.p->v; i >= 0; --i) {
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
        result[static_cast<std::size_t>(cur.p->v)] = '1';
        cur = cur.p->e.at(1);
      }
    }

    if (collapse) {
      decRef(rootEdge);

      vEdge e = vEdge::one;
      std::array<vEdge, 2> edges{};

      for (Qubit p = 0; p < numberOfQubits; p++) {
        if (result[static_cast<std::size_t>(p)] == '0') {
          edges[0] = e;
          edges[1] = vEdge::zero;
        } else {
          edges[0] = vEdge::zero;
          edges[1] = e;
        }
        e = makeDDNode(p, edges, false);
      }
      incRef(e);
      rootEdge = e;
      garbageCollect();
    }

    return std::string{result.rbegin(), result.rend()};
  }

private:
  double assignProbabilities(const vEdge& edge,
                             std::unordered_map<vNode*, fp>& probs) {
    auto it = probs.find(edge.p);
    if (it != probs.end()) {
      return ComplexNumbers::mag2(edge.w) * it->second;
    }
    double sum{1};
    if (!edge.isTerminal()) {
      sum = assignProbabilities(edge.p->e.at(0), probs) +
            assignProbabilities(edge.p->e.at(1), probs);
    }

    probs.insert({edge.p, sum});

    return ComplexNumbers::mag2(edge.w) * sum;
  }

public:
  std::pair<dd::fp, dd::fp>
  determineMeasurementProbabilities(const vEdge& rootEdge, const Qubit index,
                                    const bool assumeProbabilityNormalization) {
    std::map<vNode*, fp> probsMone;
    std::set<vNode*> visited;
    std::queue<vNode*> q;

    probsMone[rootEdge.p] = ComplexNumbers::mag2(rootEdge.w);
    visited.insert(rootEdge.p);
    q.push(rootEdge.p);

    while (q.front()->v != index) {
      vNode* ptr = q.front();
      q.pop();
      const fp prob = probsMone[ptr];

      if (!ptr->e.at(0).w.approximatelyZero()) {
        const fp tmp1 = prob * ComplexNumbers::mag2(ptr->e.at(0).w);

        if (visited.find(ptr->e.at(0).p) != visited.end()) {
          probsMone[ptr->e.at(0).p] = probsMone[ptr->e.at(0).p] + tmp1;
        } else {
          probsMone[ptr->e.at(0).p] = tmp1;
          visited.insert(ptr->e.at(0).p);
          q.push(ptr->e.at(0).p);
        }
      }

      if (!ptr->e.at(1).w.approximatelyZero()) {
        const fp tmp1 = prob * ComplexNumbers::mag2(ptr->e.at(1).w);

        if (visited.find(ptr->e.at(1).p) != visited.end()) {
          probsMone[ptr->e.at(1).p] = probsMone[ptr->e.at(1).p] + tmp1;
        } else {
          probsMone[ptr->e.at(1).p] = tmp1;
          visited.insert(ptr->e.at(1).p);
          q.push(ptr->e.at(1).p);
        }
      }
    }

    fp pzero{0};
    fp pone{0};

    if (assumeProbabilityNormalization) {
      while (!q.empty()) {
        vNode* ptr = q.front();
        q.pop();

        if (!ptr->e.at(0).w.approximatelyZero()) {
          pzero += probsMone[ptr] * ComplexNumbers::mag2(ptr->e.at(0).w);
        }

        if (!ptr->e.at(1).w.approximatelyZero()) {
          pone += probsMone[ptr] * ComplexNumbers::mag2(ptr->e.at(1).w);
        }
      }
    } else {
      std::unordered_map<vNode*, fp> probs;
      assignProbabilities(rootEdge, probs);

      while (!q.empty()) {
        vNode* ptr = q.front();
        q.pop();

        if (!ptr->e.at(0).w.approximatelyZero()) {
          pzero += probsMone[ptr] * probs[ptr->e.at(0).p] *
                   ComplexNumbers::mag2(ptr->e.at(0).w);
        }

        if (!ptr->e.at(1).w.approximatelyZero()) {
          pone += probsMone[ptr] * probs[ptr->e.at(1).p] *
                  ComplexNumbers::mag2(ptr->e.at(1).w);
        }
      }
    }
    return {pzero, pone};
  }

  char measureOneCollapsing(vEdge& rootEdge, const Qubit index,
                            const bool assumeProbabilityNormalization,
                            std::mt19937_64& mt, fp epsilon = 0.001) {
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
    GateMatrix measurementMatrix{complex_zero, complex_zero, complex_zero,
                                 complex_zero};

    std::uniform_real_distribution<fp> dist(0.0, 1.0L);

    fp threshold = dist(mt);
    fp normalizationFactor; // NOLINT(cppcoreguidelines-init-variables) always
                            // assigned a value in the following block
    char result; // NOLINT(cppcoreguidelines-init-variables) always assigned a
                 // value in the following block

    if (threshold < pzero / sum) {
      measurementMatrix[0] = complex_one;
      normalizationFactor = pzero;
      result = '0';
    } else {
      measurementMatrix[3] = complex_one;
      normalizationFactor = pone;
      result = '1';
    }

    mEdge measurementGate =
        makeGateDD(measurementMatrix,
                   static_cast<dd::QubitCount>(rootEdge.p->v + 1), index);

    vEdge e = multiply(measurementGate, rootEdge);

    Complex c = cn.getTemporary(std::sqrt(1.0 / normalizationFactor), 0);
    ComplexNumbers::mul(c, e.w, c);
    e.w = cn.lookup(c);
    incRef(e);
    decRef(rootEdge);
    rootEdge = e;

    return result;
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

  template <class Edge> Edge add(const Edge& x, const Edge& y) {
    [[maybe_unused]] const auto before = cn.cacheCount();

    auto result = add2(x, y);

    if (result.w != Complex::zero) {
      cn.returnToCache(result.w);
      result.w = cn.lookup(result.w);
    }

    [[maybe_unused]] const auto after = cn.complexCache.getCount();
    assert(after == before);

    return result;
  }

  template <class Node>
  Edge<Node> add2(const Edge<Node>& x, const Edge<Node>& y) {
    if (x.p == nullptr) {
      return y;
    }
    if (y.p == nullptr) {
      return x;
    }

    if (x.w.exactlyZero()) {
      if (y.w.exactlyZero()) {
        return Edge<Node>::zero;
      }
      auto r = y;
      r.w = cn.getCached(CTEntry::val(y.w.r), CTEntry::val(y.w.i));
      return r;
    }
    if (y.w.exactlyZero()) {
      auto r = x;
      r.w = cn.getCached(CTEntry::val(x.w.r), CTEntry::val(x.w.i));
      return r;
    }
    if (x.p == y.p) {
      auto r = y;
      r.w = cn.addCached(x.w, y.w);
      if (r.w.approximatelyZero()) {
        cn.returnToCache(r.w);
        return Edge<Node>::zero;
      }
      return r;
    }

    auto& computeTable = getAddComputeTable<Node>();
    auto r = computeTable.lookup({x.p, x.w}, {y.p, y.w});
    //           if (r.p != nullptr && false) { // activate for debugging
    //           caching only
    if (r.p != nullptr) {
      if (r.w.approximatelyZero()) {
        return Edge<Node>::zero;
      }
      return {r.p, cn.getCached(r.w)};
    }

    const Qubit w = (x.isTerminal() || (!y.isTerminal() && y.p->v > x.p->v))
                        ? y.p->v
                        : x.p->v;

    constexpr std::size_t n = std::tuple_size_v<decltype(x.p->e)>;
    std::array<Edge<Node>, n> edge{};
    for (std::size_t i = 0U; i < n; i++) {
      Edge<Node> e1{};
      if (!x.isTerminal() && x.p->v == w) {
        e1 = x.p->e[i];

        if (e1.w != Complex::zero) {
          e1.w = cn.mulCached(e1.w, x.w);
        }
      } else {
        e1 = x;
        if (y.p->e[i].p == nullptr) {
          e1 = {nullptr, Complex::zero};
        }
      }
      Edge<Node> e2{};
      if (!y.isTerminal() && y.p->v == w) {
        e2 = y.p->e[i];

        if (e2.w != Complex::zero) {
          e2.w = cn.mulCached(e2.w, y.w);
        }
      } else {
        e2 = y;
        if (x.p->e[i].p == nullptr) {
          e2 = {nullptr, Complex::zero};
        }
      }

      if constexpr (std::is_same_v<Node, dNode>) {
        dEdge::applyDmChangesToEdges(e1, e2);
        edge[i] = add2(e1, e2);
        dEdge::revertDmChangesToEdges(e1, e2);
      } else {
        edge[i] = add2(e1, e2);
      }

      if (!x.isTerminal() && x.p->v == w && e1.w != Complex::zero) {
        cn.returnToCache(e1.w);
      }

      if (!y.isTerminal() && y.p->v == w && e2.w != Complex::zero) {
        cn.returnToCache(e2.w);
      }
    }

    auto e = makeDDNode(w, edge, true);

    //           if (r.p != nullptr && e.p != r.p){ // activate for debugging
    //           caching only
    //               std::cout << "Caching error detected in add" << std::endl;
    //           }

    computeTable.insert({x.p, x.w}, {y.p, y.w}, {e.p, e.w});
    return e;
  }

  ///
  /// Matrix (conjugate) transpose
  ///
  UnaryComputeTable<mEdge, mEdge, Config::CT_MAT_TRANS_NBUCKET>
      matrixTranspose{};
  UnaryComputeTable<mEdge, mEdge, Config::CT_MAT_CONJ_TRANS_NBUCKET>
      conjugateMatrixTranspose{};

  mEdge transpose(const mEdge& a) {
    if (a.p == nullptr || a.isTerminal() || a.p->isSymmetric()) {
      return a;
    }

    // check in compute table
    auto r = matrixTranspose.lookup(a);
    if (r.p != nullptr) {
      return r;
    }

    std::array<mEdge, NEDGE> e{};
    // transpose sub-matrices and rearrange as required
    for (auto i = 0U; i < RADIX; ++i) {
      for (auto j = 0U; j < RADIX; ++j) {
        e[RADIX * i + j] = transpose(a.p->e[RADIX * j + i]);
      }
    }
    // create new top node
    r = makeDDNode(a.p->v, e);
    // adjust top weight
    auto c = cn.getTemporary();
    ComplexNumbers::mul(c, r.w, a.w);
    r.w = cn.lookup(c);

    // put in compute table
    matrixTranspose.insert(a, r);
    return r;
  }
  mEdge conjugateTranspose(const mEdge& a) {
    if (a.p == nullptr) {
      return a;
    }
    if (a.isTerminal()) { // terminal case
      auto r = a;
      r.w = ComplexNumbers::conj(a.w);
      return r;
    }

    // check if in compute table
    auto r = conjugateMatrixTranspose.lookup(a);
    if (r.p != nullptr) {
      return r;
    }

    std::array<mEdge, NEDGE> e{};
    // conjugate transpose submatrices and rearrange as required
    for (auto i = 0U; i < RADIX; ++i) {
      for (auto j = 0U; j < RADIX; ++j) {
        e[RADIX * i + j] = conjugateTranspose(a.p->e[RADIX * j + i]);
      }
    }
    // create new top node
    r = makeDDNode(a.p->v, e);

    auto c = cn.getTemporary();
    // adjust top weight including conjugate
    ComplexNumbers::mul(c, r.w, ComplexNumbers::conj(a.w));
    r.w = cn.lookup(c);

    // put it in the compute table
    conjugateMatrixTranspose.insert(a, r);
    return r;
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

  template <class LeftOperandNode, class RightOperandNode>
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
                                bool generateDensityMatrix = false) {
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

  template <class LeftOperand, class RightOperand>
  RightOperand multiply(const LeftOperand& x, const RightOperand& y,
                        dd::Qubit start = 0,
                        [[maybe_unused]] bool generateDensityMatrix = false) {
    [[maybe_unused]] const auto before = cn.cacheCount();

    Qubit var = -1;
    RightOperand e;

    if constexpr (std::is_same_v<LeftOperand, dEdge>) {
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
      if (!y.isTerminal() && (y.p->v) > var) {
        var = y.p->v;
      }
      e = multiply2(x, y, var, start);
    }

    if (!e.w.exactlyZero() && !e.w.exactlyOne()) {
      cn.returnToCache(e.w);
      e.w = cn.lookup(e.w);
    }

    [[maybe_unused]] const auto after = cn.cacheCount();
    assert(before == after);

    return e;
  }

private:
  template <class LeftOperandNode, class RightOperandNode>
  Edge<RightOperandNode>
  multiply2(const Edge<LeftOperandNode>& x, const Edge<RightOperandNode>& y,
            Qubit var, Qubit start = 0,
            [[maybe_unused]] bool generateDensityMatrix = false) {
    using LEdge = Edge<LeftOperandNode>;
    using REdge = Edge<RightOperandNode>;
    using ResultEdge = Edge<RightOperandNode>;

    if (x.p == nullptr) {
      return {nullptr, Complex::zero};
    }
    if (y.p == nullptr) {
      return y;
    }

    if (x.w.exactlyZero() || y.w.exactlyZero()) {
      return ResultEdge::zero;
    }

    if (var == start - 1) {
      return ResultEdge::terminal(cn.mulCached(x.w, y.w));
    }

    auto xCopy = x;
    xCopy.w = Complex::one;
    auto yCopy = y;
    yCopy.w = Complex::one;

    auto& computeTable =
        getMultiplicationComputeTable<LeftOperandNode, RightOperandNode>();
    auto r = computeTable.lookup(xCopy, yCopy, generateDensityMatrix);
    //            if (r.p != nullptr && false) { // activate for debugging
    //            caching only
    if (r.p != nullptr) {
      if (r.w.approximatelyZero()) {
        return ResultEdge::zero;
      }
      auto e = ResultEdge{r.p, cn.getCached(r.w)};
      ComplexNumbers::mul(e.w, e.w, x.w);
      ComplexNumbers::mul(e.w, e.w, y.w);
      if (e.w.approximatelyZero()) {
        cn.returnToCache(e.w);
        return ResultEdge::zero;
      }
      return e;
    }

    constexpr std::size_t n = std::tuple_size_v<decltype(y.p->e)>;

    ResultEdge e{};
    if constexpr (std::is_same_v<RightOperandNode, mCachedEdge>) {
      // This branch is only taken for matrices
      if (x.p->v == var && x.p->v == y.p->v) {
        if (x.p->isIdentity()) {
          if constexpr (n == NEDGE) {
            // additionally check if y is the identity in case of matrix
            // multiplication
            if (y.p->isIdentity()) {
              e = makeIdent(start, var);
            } else {
              e = yCopy;
            }
          } else {
            e = yCopy;
          }
          computeTable.insert(xCopy, yCopy, {e.p, e.w});
          e.w = cn.mulCached(x.w, y.w);
          if (e.w.approximatelyZero()) {
            cn.returnToCache(e.w);
            return ResultEdge::zero;
          }
          return e;
        }

        if constexpr (n == NEDGE) {
          // additionally check if y is the identity in case of matrix
          // multiplication
          if (y.p->isIdentity()) {
            e = xCopy;
            computeTable.insert(xCopy, yCopy, {e.p, e.w});
            e.w = cn.mulCached(x.w, y.w);

            if (e.w.approximatelyZero()) {
              cn.returnToCache(e.w);
              return ResultEdge::zero;
            }
            return e;
          }
        }
      }
    }

    constexpr std::size_t rows = RADIX;
    constexpr std::size_t cols = n == NEDGE ? RADIX : 1U;

    std::array<ResultEdge, n> edge{};
    for (auto i = 0U; i < rows; i++) {
      for (auto j = 0U; j < cols; j++) {
        auto idx = cols * i + j;
        edge[idx] = ResultEdge::zero;
        for (auto k = 0U; k < rows; k++) {
          LEdge e1{};
          if (!x.isTerminal() && x.p->v == var) {
            e1 = x.p->e[rows * i + k];
          } else {
            e1 = xCopy;
          }

          REdge e2{};
          if (!y.isTerminal() && y.p->v == var) {
            e2 = y.p->e[j + cols * k];
          } else {
            e2 = yCopy;
          }

          if constexpr (std::is_same_v<LeftOperandNode, dNode>) {
            dEdge m;
            dEdge::applyDmChangesToEdges(e1, e2);
            if (!generateDensityMatrix || idx == 1) {
              // When generateDensityMatrix is false or I have the first edge I
              // don't optimize anything and set generateDensityMatrix to false
              // for all child edges
              m = multiply2(e1, e2, static_cast<Qubit>(var - 1), start, false);
            } else if (idx == 2) {
              // When I have the second edge and generateDensityMatrix == false,
              // then edge[2] == edge[1]
              if (k == 0) {
                if (edge[1].w.approximatelyZero()) {
                  edge[2] = ResultEdge::zero;
                } else {
                  edge[2] =
                      ResultEdge{edge[1].p, cn.getCached(edge[1].w.r->value,
                                                         edge[1].w.i->value)};
                }
              }
              continue;
            } else {
              m = multiply2(e1, e2, static_cast<Qubit>(var - 1), start,
                            generateDensityMatrix);
            }

            if (k == 0 || edge[idx].w.exactlyZero()) {
              edge[idx] = m;
            } else if (!m.w.exactlyZero()) {
              dEdge::applyDmChangesToEdges(edge[idx], m);
              auto oldE = edge[idx];
              edge[idx] = add2(edge[idx], m);
              dEdge::revertDmChangesToEdges(edge[idx], e2);
              cn.returnToCache(oldE.w);
              cn.returnToCache(m.w);
            }
            // Undo modifications on density matrices
            dEdge::revertDmChangesToEdges(e1, e2);
          } else {
            auto m = multiply2(e1, e2, static_cast<Qubit>(var - 1), start);

            if (k == 0 || edge[idx].w.exactlyZero()) {
              edge[idx] = m;
            } else if (!m.w.exactlyZero()) {
              auto oldE = edge[idx];
              edge[idx] = add2(edge[idx], m);
              cn.returnToCache(oldE.w);
              cn.returnToCache(m.w);
            }
          }
        }
      }
    }
    e = makeDDNode(var, edge, true, generateDensityMatrix);

    //            if (r.p != nullptr && e.p != r.p) { // activate for debugging
    //            caching
    //                std::cout << "Caching error detected in mul" << std::endl;
    //            }

    computeTable.insert(xCopy, yCopy, {e.p, e.w});

    if (!e.w.exactlyZero() && (x.w.exactlyOne() || !y.w.exactlyZero())) {
      if (e.w.exactlyOne()) {
        e.w = cn.mulCached(x.w, y.w);
      } else {
        ComplexNumbers::mul(e.w, e.w, x.w);
        ComplexNumbers::mul(e.w, e.w, y.w);
      }
      if (e.w.approximatelyZero()) {
        cn.returnToCache(e.w);
        return ResultEdge::zero;
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
    if (x.p == nullptr || y.p == nullptr || x.w.approximatelyZero() ||
        y.w.approximatelyZero()) { // the 0 case
      return {0, 0};
    }

    [[maybe_unused]] const auto before = cn.cacheCount();

    auto w = x.p->v;
    if (y.p->v > w) {
      w = y.p->v;
    }
    auto xCopy = x;
    xCopy.w = ComplexNumbers::conj(
        x.w); // Overall normalization factor needs to be conjugated
              // before input into recursive private function
    const ComplexValue ip = innerProduct(xCopy, y, static_cast<Qubit>(w + 1));

    [[maybe_unused]] const auto after = cn.cacheCount();
    assert(after == before);

    return ip;
  }

  fp fidelity(const vEdge& x, const vEdge& y) {
    const auto fid = innerProduct(x, y);
    return fid.r * fid.r + fid.i * fid.i;
  }

  [[gnu::pure]] dd::fp
  fidelityOfMeasurementOutcomes(const vEdge& e,
                                const ProbabilityVector& probs) {
    if (e.w.approximatelyZero()) {
      return 0.;
    }
    return fidelityOfMeasurementOutcomesRecursive(e, probs, 0);
  }

  [[gnu::pure]] dd::fp fidelityOfMeasurementOutcomesRecursive(
      const vEdge& e, const ProbabilityVector& probs, const std::size_t i) {
    const auto topw = dd::ComplexNumbers::mag(e.w);
    if (e.isTerminal()) {
      if (auto it = probs.find(i); it != probs.end()) {
        return topw * std::sqrt(it->second);
      }
      return 0.;
    }

    std::size_t leftIdx = i;
    dd::fp leftContribution = 0.;
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

    const dd::fp fidelity = topw * (leftContribution + rightContribution);
    return fidelity;
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
    if (x.p == nullptr || y.p == nullptr || x.w.approximatelyZero() ||
        y.w.approximatelyZero()) { // the 0 case
      return {0.0, 0.0};
    }

    if (var == 0) { // Multiplies terminal weights
      auto c = cn.getTemporary();
      ComplexNumbers::mul(c, x.w, y.w);
      return {c.r->value, c.i->value};
    }

    auto xCopy = x;
    xCopy.w = Complex::one; // Set to one to generate more lookup hits
    auto yCopy = y;
    yCopy.w = Complex::one;

    auto r = vectorInnerProduct.lookup(xCopy, yCopy);
    if (r.p != nullptr) {
      auto c = cn.getTemporary(r.w);
      ComplexNumbers::mul(c, c, x.w);
      ComplexNumbers::mul(c, c, y.w);
      return {CTEntry::val(c.r), CTEntry::val(c.i)};
    }

    auto w = static_cast<Qubit>(var - 1);

    ComplexValue sum{0.0, 0.0};
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
      auto cv = innerProduct(e1, e2, w);
      sum.r += cv.r;
      sum.i += cv.i;
    }
    r.p = vNode::terminal;
    r.w = sum;

    vectorInnerProduct.insert(xCopy, yCopy, r);
    auto c = cn.getTemporary(sum);
    ComplexNumbers::mul(c, c, x.w);
    ComplexNumbers::mul(c, c, y.w);
    return {CTEntry::val(c.r), CTEntry::val(c.i)};
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
    if (x.p->v != y.p->v) {
      throw std::invalid_argument(
          "Observable and state must act on the same number of qubits to "
          "compute the expectation value.");
    }

    auto yPrime = multiply(x, y);
    const ComplexValue expValue = innerProduct(y, yPrime);

    assert(CTEntry::approximatelyZero(expValue.i));

    garbageCollect();

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

  template <class Edge>
  Edge kronecker(const Edge& x, const Edge& y, bool incIdx = true) {
    if constexpr (std::is_same_v<Edge, dEdge>) {
      throw std::invalid_argument(
          "Kronecker is currently not supported for density matrices");
    }

    auto e = kronecker2(x, y, incIdx);

    if (e.w != Complex::zero && !e.w.exactlyOne()) {
      cn.returnToCache(e.w);
      e.w = cn.lookup(e.w);
    }

    return e;
  }

  // extent the DD pointed to by `e` with `h` identities on top and `l`
  // identities at the bottom
  mEdge extend(const mEdge& e, Qubit h, Qubit l = 0) {
    auto f =
        (l > 0) ? kronecker(e, makeIdent(static_cast<dd::QubitCount>(l))) : e;
    auto g =
        (h > 0) ? kronecker(makeIdent(static_cast<dd::QubitCount>(h)), f) : f;
    return g;
  }

private:
  template <class Node>
  Edge<Node> kronecker2(const Edge<Node>& x, const Edge<Node>& y,
                        bool incIdx = true) {
    if (x.w.approximatelyZero() || y.w.approximatelyZero()) {
      return Edge<Node>::zero;
    }

    if (x.isTerminal()) {
      auto r = y;
      r.w = cn.mulCached(x.w, y.w);
      return r;
    }

    auto& computeTable = getKroneckerComputeTable<Node>();
    auto r = computeTable.lookup(x, y);
    if (r.p != nullptr) {
      if (r.w.approximatelyZero()) {
        return Edge<Node>::zero;
      }
      return {r.p, cn.getCached(r.w)};
    }

    constexpr std::size_t n = std::tuple_size_v<decltype(x.p->e)>;
    // special case handling for matrices
    if constexpr (n == NEDGE) {
      if (x.p->isIdentity()) {
        auto idx = incIdx ? static_cast<Qubit>(y.p->v + 1) : y.p->v;
        auto e = makeDDNode(
            idx, std::array{y, Edge<Node>::zero, Edge<Node>::zero, y});
        for (auto i = 0; i < x.p->v; ++i) {
          idx = incIdx ? static_cast<Qubit>(e.p->v + 1) : e.p->v;
          e = makeDDNode(idx,
                         std::array{e, Edge<Node>::zero, Edge<Node>::zero, e});
        }

        e.w = cn.getCached(CTEntry::val(y.w.r), CTEntry::val(y.w.i));
        computeTable.insert(x, y, {e.p, e.w});
        return e;
      }
    }

    std::array<Edge<Node>, n> edge{};
    for (auto i = 0U; i < n; ++i) {
      edge[i] = kronecker2(x.p->e[i], y, incIdx);
    }

    auto idx = incIdx ? static_cast<Qubit>(y.p->v + x.p->v + 1) : x.p->v;
    auto e = makeDDNode(idx, edge, true);
    ComplexNumbers::mul(e.w, e.w, x.w);
    computeTable.insert(x, y, {e.p, e.w});
    return e;
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
    auto eliminate = std::vector<bool>(nqubits, true);
    [[maybe_unused]] const auto before = cn.cacheCount();
    const auto res = partialTrace(a, eliminate);
    [[maybe_unused]] const auto after = cn.cacheCount();
    assert(before == after);
    return {CTEntry::val(res.w.r), CTEntry::val(res.w.i)};
  }
  bool isCloseToIdentity(const mEdge& m, dd::fp tol = 1e-10) {
    std::unordered_set<decltype(m.p)> visited{};
    visited.reserve(mUniqueTable.getActiveNodeCount());
    return isCloseToIdentityRecursive(m, visited, tol);
  }

private:
  /// TODO: introduce a compute table for the trace?
  mEdge trace(const mEdge& a, const std::vector<bool>& eliminate,
              std::size_t alreadyEliminated = 0) {
    if (a.w.approximatelyZero()) {
      return mEdge::zero;
    }

    if (std::none_of(eliminate.begin(), eliminate.end(),
                     [](bool v) { return v; })) {
      return a;
    }
    auto v = a.p->v;
    // Base case
    if (v == -1) {
      if (a.isTerminal()) {
        return a;
      }
      throw std::runtime_error("Expected terminal node in trace.");
    }

    if (eliminate[static_cast<std::size_t>(v)]) {
      auto elims = alreadyEliminated + 1;
      auto r = mEdge::zero;

      auto t0 = trace(a.p->e[0], eliminate, elims);
      r = add2(r, t0);
      auto r1 = r;

      auto t1 = trace(a.p->e[3], eliminate, elims);
      r = add2(r, t1);
      auto r2 = r;

      if (r.w.exactlyOne()) {
        r.w = a.w;
      } else {
        auto c = cn.getTemporary();
        ComplexNumbers::mul(c, r.w, a.w);
        r.w =
            cn.lookup(c); // better safe than sorry. this may result in complex
                          // values with magnitude > 1 in the complex table
      }

      if (r1.w != Complex::zero) {
        cn.returnToCache(r1.w);
      }

      if (r2.w != Complex::zero) {
        cn.returnToCache(r2.w);
      }

      return r;
    }

    std::array<mEdge, NEDGE> edge{};
    std::transform(a.p->e.cbegin(), a.p->e.cend(), edge.begin(),
                   [&](const mEdge& e) -> mEdge {
                     return trace(e, eliminate, alreadyEliminated);
                   });
    auto adjustedV =
        static_cast<Qubit>(static_cast<std::size_t>(a.p->v) -
                           (static_cast<std::size_t>(std::count(
                                eliminate.begin(), eliminate.end(), true)) -
                            alreadyEliminated));
    auto r = makeDDNode(adjustedV, edge);

    if (r.w.exactlyOne()) {
      r.w = a.w;
    } else {
      auto c = cn.getTemporary();
      ComplexNumbers::mul(c, r.w, a.w);
      r.w = cn.lookup(c);
    }
    return r;
  }

  bool isCloseToIdentityRecursive(const mEdge& m,
                                  std::unordered_set<decltype(m.p)>& visited,
                                  dd::fp tol) {
    // immediately return if this node has already been visited
    if (visited.find(m.p) != visited.end()) {
      return true;
    }

    // immediately return of this node is identical to the identity
    if (m.p->isIdentity()) {
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

  ///
  /// Toffoli gates
  ///
public:
  ToffoliTable<mEdge> toffoliTable{};

  ///
  /// Identity matrices
  ///
  // create n-qubit identity DD. makeIdent(n) === makeIdent(0, n-1)
  mEdge makeIdent(QubitCount n) {
    return makeIdent(0, static_cast<Qubit>(n - 1));
  }
  mEdge makeIdent(Qubit leastSignificantQubit, Qubit mostSignificantQubit) {
    if (mostSignificantQubit < leastSignificantQubit) {
      return mEdge::one;
    }

    if (leastSignificantQubit == 0 &&
        idTable[static_cast<std::size_t>(mostSignificantQubit)].p != nullptr) {
      return idTable[static_cast<std::size_t>(mostSignificantQubit)];
    }
    if (mostSignificantQubit >= 1 &&
        (idTable[static_cast<std::size_t>(mostSignificantQubit - 1)]).p !=
            nullptr) {
      idTable[static_cast<std::size_t>(mostSignificantQubit)] = makeDDNode(
          mostSignificantQubit,
          std::array{
              idTable[static_cast<std::size_t>(mostSignificantQubit - 1)],
              mEdge::zero, mEdge::zero,
              idTable[static_cast<std::size_t>(mostSignificantQubit - 1)]});
      return idTable[static_cast<std::size_t>(mostSignificantQubit)];
    }

    auto e =
        makeDDNode(leastSignificantQubit, std::array{mEdge::one, mEdge::zero,
                                                     mEdge::zero, mEdge::one});
    for (auto k = static_cast<std::size_t>(leastSignificantQubit + 1);
         k <= static_cast<std::make_unsigned_t<Qubit>>(mostSignificantQubit);
         k++) {
      e = makeDDNode(static_cast<Qubit>(k),
                     std::array{e, mEdge::zero, mEdge::zero, e});
    }
    if (leastSignificantQubit == 0) {
      idTable[static_cast<std::size_t>(mostSignificantQubit)] = e;
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

  mEdge createInitialMatrix(dd::QubitCount n,
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
  /// Decision diagram size
  ///
  template <class Edge> unsigned int size(const Edge& e) {
    static constexpr unsigned int NODECOUNT_BUCKETS = 200000;
    static std::unordered_set<decltype(e.p)> visited{NODECOUNT_BUCKETS}; // 2e6
    visited.max_load_factor(10);
    visited.clear();
    return nodeCount(e, visited);
  }

private:
  template <class Edge>
  unsigned int nodeCount(const Edge& e,
                         std::unordered_set<decltype(e.p)>& v) const {
    v.insert(e.p);
    unsigned int sum = 1;
    if (!e.isTerminal()) {
      for (const auto& edge : e.p->e) {
        if (edge.p != nullptr && !v.count(edge.p)) {
          sum += nodeCount(edge, v);
        }
      }
    }
    return sum;
  }

  ///
  /// Ancillary and garbage reduction
  ///
public:
  mEdge reduceAncillae(mEdge& e, const std::vector<bool>& ancillary,
                       bool regular = true) {
    // return if no more garbage left
    if (std::none_of(ancillary.begin(), ancillary.end(),
                     [](bool v) { return v; }) ||
        e.p == nullptr) {
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
    decRef(e);
    incRef(f);
    return f;
  }

  // Garbage reduction works for reversible circuits --- to be thoroughly tested
  // for quantum circuits
  vEdge reduceGarbage(vEdge& e, const std::vector<bool>& garbage) {
    // return if no more garbage left
    if (std::none_of(garbage.begin(), garbage.end(),
                     [](bool v) { return v; }) ||
        e.p == nullptr) {
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
    decRef(e);
    incRef(f);
    return f;
  }
  mEdge reduceGarbage(mEdge& e, const std::vector<bool>& garbage,
                      bool regular = true) {
    // return if no more garbage left
    if (std::none_of(garbage.begin(), garbage.end(),
                     [](bool v) { return v; }) ||
        e.p == nullptr) {
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
    decRef(e);
    incRef(f);
    return f;
  }

private:
  mEdge reduceAncillaeRecursion(mEdge& e, const std::vector<bool>& ancillary,
                                Qubit lowerbound, bool regular = true) {
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
    if (f.p->v >= 0 && ancillary[static_cast<std::size_t>(f.p->v)]) {
      if (regular) {
        if (f.p->e[1].w != Complex::zero || f.p->e[3].w != Complex::zero) {
          f = makeDDNode(f.p->v, std::array{f.p->e[0], mEdge::zero, f.p->e[2],
                                            mEdge::zero});
        }
      } else {
        if (f.p->e[2].w != Complex::zero || f.p->e[3].w != Complex::zero) {
          f = makeDDNode(f.p->v, std::array{f.p->e[0], f.p->e[1], mEdge::zero,
                                            mEdge::zero});
        }
      }
    }

    auto c = cn.mulCached(f.w, e.w);
    f.w = cn.lookup(c);
    cn.returnToCache(c);
    return f;
  }

  vEdge reduceGarbageRecursion(vEdge& e, const std::vector<bool>& garbage,
                               Qubit lowerbound) {
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
    if (f.p->v >= 0 && garbage[static_cast<std::size_t>(f.p->v)]) {
      if (f.p->e[1].w != Complex::zero) {
        vEdge g{};
        if (f.p->e[0].w == Complex::zero && f.p->e[1].w != Complex::zero) {
          g = f.p->e[1];
        } else if (f.p->e[1].w != Complex::zero) {
          g = add(f.p->e[0], f.p->e[1]);
        } else {
          g = f.p->e[0];
        }
        f = makeDDNode(e.p->v, std::array{g, vEdge::zero});
      }
    }

    auto c = cn.mulCached(f.w, e.w);
    f.w = cn.lookup(c);
    cn.returnToCache(c);

    // Quick-fix for normalization bug
    if (ComplexNumbers::mag2(f.w) > 1.0) {
      f.w = Complex::one;
    }

    return f;
  }
  mEdge reduceGarbageRecursion(mEdge& e, const std::vector<bool>& garbage,
                               Qubit lowerbound, bool regular = true) {
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
    if (f.p->v >= 0 && garbage[static_cast<std::size_t>(f.p->v)]) {
      if (regular) {
        if (f.p->e[2].w != Complex::zero || f.p->e[3].w != Complex::zero) {
          mEdge g{};
          if (f.p->e[0].w == Complex::zero && f.p->e[2].w != Complex::zero) {
            g = f.p->e[2];
          } else if (f.p->e[2].w != Complex::zero) {
            g = add(f.p->e[0], f.p->e[2]);
          } else {
            g = f.p->e[0];
          }
          mEdge h{};
          if (f.p->e[1].w == Complex::zero && f.p->e[3].w != Complex::zero) {
            h = f.p->e[3];
          } else if (f.p->e[3].w != Complex::zero) {
            h = add(f.p->e[1], f.p->e[3]);
          } else {
            h = f.p->e[1];
          }
          f = makeDDNode(e.p->v, std::array{g, h, mEdge::zero, mEdge::zero});
        }
      } else {
        if (f.p->e[1].w != Complex::zero || f.p->e[3].w != Complex::zero) {
          mEdge g{};
          if (f.p->e[0].w == Complex::zero && f.p->e[1].w != Complex::zero) {
            g = f.p->e[1];
          } else if (f.p->e[1].w != Complex::zero) {
            g = add(f.p->e[0], f.p->e[1]);
          } else {
            g = f.p->e[0];
          }
          mEdge h{};
          if (f.p->e[2].w == Complex::zero && f.p->e[3].w != Complex::zero) {
            h = f.p->e[3];
          } else if (f.p->e[3].w != Complex::zero) {
            h = add(f.p->e[2], f.p->e[3]);
          } else {
            h = f.p->e[2];
          }
          f = makeDDNode(e.p->v, std::array{g, mEdge::zero, h, mEdge::zero});
        }
      }
    }

    auto c = cn.mulCached(f.w, e.w);
    f.w = cn.lookup(c);
    cn.returnToCache(c);

    // Quick-fix for normalization bug
    if (ComplexNumbers::mag2(f.w) > 1.0) {
      f.w = Complex::one;
    }

    return f;
  }

  ///
  /// Vector and matrix extraction from DDs
  ///
public:
  /// Get a single element of the vector or matrix represented by the dd with
  /// root edge e \tparam Edge type of edge to use (vector or matrix) \param e
  /// edge to traverse \param elements string {0, 1, 2, 3}^n describing which
  /// outgoing edge should be followed
  ///        (for vectors entries are limited to 0 and 1)
  ///        If string is longer than required, the additional characters are
  ///        ignored.
  /// \return the complex amplitude of the specified element
  template <class Edge>
  ComplexValue getValueByPath(const Edge& e, const std::string& elements) {
    if (e.isTerminal()) {
      return {CTEntry::val(e.w.r), CTEntry::val(e.w.i)};
    }

    auto c = cn.getTemporary(1, 0);
    auto r = e;
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-do-while)
    do {
      ComplexNumbers::mul(c, c, r.w);
      auto tmp = static_cast<std::size_t>(
          elements.at(static_cast<std::size_t>(r.p->v)) - '0');
      assert(tmp <= r.p->e.size());
      r = r.p->e.at(tmp);
    } while (!r.isTerminal());
    ComplexNumbers::mul(c, c, r.w);

    return {CTEntry::val(c.r), CTEntry::val(c.i)};
  }
  ComplexValue getValueByPath(const vEdge& e, std::size_t i) {
    if (e.isTerminal()) {
      return {CTEntry::val(e.w.r), CTEntry::val(e.w.i)};
    }
    return getValueByPath(e, Complex::one, i);
  }
  ComplexValue getValueByPath(const vEdge& e, const Complex& amp,
                              std::size_t i) {
    auto c = cn.mulCached(e.w, amp);

    if (e.isTerminal()) {
      cn.returnToCache(c);
      return {CTEntry::val(c.r), CTEntry::val(c.i)};
    }

    const bool one = (i & (1ULL << e.p->v)) != 0U;

    ComplexValue r{};
    if (!one && !e.p->e[0].w.approximatelyZero()) {
      r = getValueByPath(e.p->e[0], c, i);
    } else if (one && !e.p->e[1].w.approximatelyZero()) {
      r = getValueByPath(e.p->e[1], c, i);
    }
    cn.returnToCache(c);
    return r;
  }
  ComplexValue getValueByPath(const mEdge& e, std::size_t i, std::size_t j) {
    if (e.isTerminal()) {
      return {CTEntry::val(e.w.r), CTEntry::val(e.w.i)};
    }
    return getValueByPath(e, Complex::one, i, j);
  }
  ComplexValue getValueByPath(const mEdge& e, const Complex& amp, std::size_t i,
                              std::size_t j) {
    auto c = cn.mulCached(e.w, amp);

    if (e.isTerminal()) {
      cn.returnToCache(c);
      return {CTEntry::val(c.r), CTEntry::val(c.i)};
    }

    const bool row = (i & (1ULL << e.p->v)) != 0U;
    const bool col = (j & (1ULL << e.p->v)) != 0U;

    ComplexValue r{};
    if (!row && !col && !e.p->e[0].w.approximatelyZero()) {
      r = getValueByPath(e.p->e[0], c, i, j);
    } else if (!row && col && !e.p->e[1].w.approximatelyZero()) {
      r = getValueByPath(e.p->e[1], c, i, j);
    } else if (row && !col && !e.p->e[2].w.approximatelyZero()) {
      r = getValueByPath(e.p->e[2], c, i, j);
    } else if (row && col && !e.p->e[3].w.approximatelyZero()) {
      r = getValueByPath(e.p->e[3], c, i, j);
    }
    cn.returnToCache(c);
    return r;
  }

  std::map<std::string, dd::fp>
  getProbVectorFromDensityMatrix(dEdge e, double measurementThreshold) {
    dEdge::alignDensityEdge(e);
    if (std::pow(2, e.p->v + 1) >=
        static_cast<double>(std::numeric_limits<std::size_t>::max())) {
      throw std::runtime_error(
          std::string{"Density matrix is too large to measure!"});
    }

    const std::size_t statesToMeasure = 2ULL << e.p->v;
    std::map<std::string, dd::fp> measuredResult = {};
    for (std::size_t m = 0; m < statesToMeasure; m++) {
      std::size_t currentResult = m;
      auto globalProbability = dd::CTEntry::val(e.w.r);
      auto resultString = intToString(m, '1', e.p->v + 1);
      dEdge cur = e;
      for (dd::Qubit i = 0; i < e.p->v + 1; ++i) {
        if (cur.p->v == -1 || globalProbability <= measurementThreshold) {
          globalProbability = 0;
          break;
        }
        assert(dd::CTEntry::approximatelyZero(cur.p->e.at(0).w.i) &&
               dd::CTEntry::approximatelyZero(cur.p->e.at(3).w.i));
        auto p0 = dd::CTEntry::val(cur.p->e.at(0).w.r);
        auto p1 = dd::CTEntry::val(cur.p->e.at(3).w.r);

        if (currentResult % 2 == 0) {
          cur = cur.p->e.at(0);
          globalProbability *= p0;
        } else {
          cur = cur.p->e.at(3);
          globalProbability *= p1;
        }
        currentResult = currentResult >> 1;
      }
      if (globalProbability > 0) { // No need to track probabilities of 0
        measuredResult.insert({resultString, globalProbability});
      }
    }
    return measuredResult;
  }

  [[nodiscard]] std::string intToString(std::size_t targetNumber, char value,
                                        dd::Qubit size) const {
    std::string path(static_cast<std::size_t>(size), '0');
    for (auto i = 1; i <= size; i++) {
      if ((targetNumber % 2) != 0U) {
        path[static_cast<std::size_t>(size - i)] = value;
      }
      targetNumber = targetNumber >> 1U;
    }
    return path;
  }

  CVec getVector(const vEdge& e) {
    const std::size_t dim = 2ULL << e.p->v;
    // allocate resulting vector
    auto vec = CVec(dim, {0.0, 0.0});
    getVector(e, Complex::one, 0, vec);
    return vec;
  }
  void getVector(const vEdge& e, const Complex& amp, std::size_t i, CVec& vec) {
    // calculate new accumulated amplitude
    auto c = cn.mulCached(e.w, amp);

    // base case
    if (e.isTerminal()) {
      vec.at(i) = {CTEntry::val(c.r), CTEntry::val(c.i)};
      cn.returnToCache(c);
      return;
    }

    const std::size_t x = i | (1ULL << e.p->v);

    // recursive case
    if (!e.p->e[0].w.approximatelyZero()) {
      getVector(e.p->e[0], c, i, vec);
    }
    if (!e.p->e[1].w.approximatelyZero()) {
      getVector(e.p->e[1], c, x, vec);
    }
    cn.returnToCache(c);
  }

  void printVector(const vEdge& e) {
    const std::size_t element = 2ULL << e.p->v;
    for (auto i = 0ULL; i < element; i++) {
      const auto amplitude = getValueByPath(e, i);
      for (Qubit j = e.p->v; j >= 0; j--) {
        std::cout << ((i >> j) & 1ULL);
      }
      constexpr auto precision = 3;
      // set fixed width to maximum of a printed number
      // (-) 0.precision plus/minus 0.precision i
      constexpr auto width = 1 + 2 + precision + 1 + 2 + precision + 1;
      std::cout << ": " << std::setw(width)
                << ComplexValue::toString(amplitude.r, amplitude.i, false,
                                          precision)
                << "\n";
    }
    std::cout << std::flush;
  }

  void printMatrix(const mEdge& e) {
    const std::size_t element = 2ULL << e.p->v;
    for (auto i = 0ULL; i < element; i++) {
      for (auto j = 0ULL; j < element; j++) {
        const auto amplitude = getValueByPath(e, i, j);
        constexpr auto precision = 3;
        // set fixed width to maximum of a printed number
        // (-) 0.precision plus/minus 0.precision i
        constexpr auto width = 1 + 2 + precision + 1 + 2 + precision + 1;
        std::cout << std::setw(width)
                  << ComplexValue::toString(amplitude.r, amplitude.i, false,
                                            precision)
                  << " ";
      }
      std::cout << "\n";
    }
    std::cout << std::flush;
  }

  CMat getMatrix(const mEdge& e) {
    const std::size_t dim = 2ULL << e.p->v;
    // allocate resulting matrix
    auto mat = CMat(dim, CVec(dim, {0.0, 0.0}));
    getMatrix(e, Complex::one, 0, 0, mat);
    return mat;
  }
  void getMatrix(const mEdge& e, const Complex& amp, std::size_t i,
                 std::size_t j, CMat& mat) {
    // calculate new accumulated amplitude
    auto c = cn.mulCached(e.w, amp);

    // base case
    if (e.isTerminal()) {
      mat.at(i).at(j) = {CTEntry::val(c.r), CTEntry::val(c.i)};
      cn.returnToCache(c);
      return;
    }

    const std::size_t x = i | (1ULL << e.p->v);
    const std::size_t y = j | (1ULL << e.p->v);

    // recursive case
    if (!e.p->e[0].w.approximatelyZero()) {
      getMatrix(e.p->e[0], c, i, j, mat);
    }
    if (!e.p->e[1].w.approximatelyZero()) {
      getMatrix(e.p->e[1], c, i, y, mat);
    }
    if (!e.p->e[2].w.approximatelyZero()) {
      getMatrix(e.p->e[2], c, x, j, mat);
    }
    if (!e.p->e[3].w.approximatelyZero()) {
      getMatrix(e.p->e[3], c, x, y, mat);
    }
    cn.returnToCache(c);
  }

  CMat getDensityMatrix(dEdge& e) {
    dEdge::applyDmChangesToEdge(e);
    const std::size_t dim = 2ULL << e.p->v;
    // allocate resulting matrix
    auto mat = CMat(dim, CVec(dim, {0.0, 0.0}));
    getDensityMatrix(e, Complex::one, 0, 0, mat);
    dd::dEdge::revertDmChangesToEdge(e);
    return mat;
  }

  void getDensityMatrix(dEdge& e, const Complex& amp, std::size_t i,
                        std::size_t j, CMat& mat) {
    // calculate new accumulated amplitude
    auto c = cn.mulCached(e.w, amp);

    // base case
    if (e.isTerminal()) {
      mat.at(i).at(j) = {CTEntry::val(c.r), CTEntry::val(c.i)};
      cn.returnToCache(c);
      return;
    }

    const std::size_t x = i | (1ULL << e.p->v);
    const std::size_t y = j | (1ULL << e.p->v);

    // recursive case
    if (!e.p->e[0].w.approximatelyZero()) {
      dEdge::applyDmChangesToEdge(e.p->e[0]);
      getDensityMatrix(e.p->e[0], c, i, j, mat);
      dd::dEdge::revertDmChangesToEdge(e.p->e[0]);
    }
    if (!e.p->e[1].w.approximatelyZero()) {
      dEdge::applyDmChangesToEdge(e.p->e[1]);
      getDensityMatrix(e.p->e[1], c, i, y, mat);
      dd::dEdge::revertDmChangesToEdge(e.p->e[1]);
    }
    if (!e.p->e[2].w.approximatelyZero()) {
      dEdge::applyDmChangesToEdge(e.p->e[2]);
      getDensityMatrix(e.p->e[2], c, x, j, mat);
      dd::dEdge::revertDmChangesToEdge(e.p->e[2]);
    }
    if (!e.p->e[3].w.approximatelyZero()) {
      dEdge::applyDmChangesToEdge(e.p->e[3]);
      getDensityMatrix(e.p->e[3], c, x, y, mat);
      dd::dEdge::revertDmChangesToEdge(e.p->e[3]);
    }

    cn.returnToCache(c);
  }

  void exportAmplitudesRec(const vEdge& edge, std::ostream& oss,
                           const std::string& path, Complex& amplitude,
                           dd::QubitCount level, bool binary = false) {
    if (edge.isTerminal()) {
      auto amp = cn.getTemporary();
      dd::ComplexNumbers::mul(amp, amplitude, edge.w);
      for (std::size_t i = 0; i < (1ULL << level); i++) {
        if (binary) {
          amp.writeBinary(oss);
        } else {
          oss << amp.toString(false, 16) << "\n";
        }
      }

      return;
    }

    auto a = cn.mulCached(amplitude, edge.w);
    exportAmplitudesRec(edge.p->e[0], oss, path + "0", a, level - 1, binary);
    exportAmplitudesRec(edge.p->e[1], oss, path + "1", a, level - 1, binary);
    cn.returnToCache(a);
  }
  void exportAmplitudes(const vEdge& edge, std::ostream& oss, dd::QubitCount nq,
                        bool binary = false) {
    if (edge.isTerminal()) {
      // TODO special treatment
      return;
    }
    auto weight = cn.getCached(1., 0.);
    exportAmplitudesRec(edge, oss, "", weight, nq, binary);
    cn.returnToCache(weight);
  }
  void exportAmplitudes(const vEdge& edge, const std::string& outputFilename,
                        dd::QubitCount nq, bool binary = false) {
    std::ofstream init(outputFilename);
    std::ostringstream oss{};

    exportAmplitudes(edge, oss, nq, binary);

    init << oss.str() << std::flush;
    init.close();
  }

  void exportAmplitudesRec(const vEdge& edge,
                           std::vector<std::complex<dd::fp>>& amplitudes,
                           Complex& amplitude, dd::QubitCount level,
                           std::size_t idx) {
    if (edge.isTerminal()) {
      auto amp = cn.getTemporary();
      dd::ComplexNumbers::mul(amp, amplitude, edge.w);
      idx <<= level;
      for (std::size_t i = 0; i < (1ULL << level); i++) {
        amplitudes[idx++] =
            std::complex<dd::fp>{dd::ComplexTable<>::Entry::val(amp.r),
                                 dd::ComplexTable<>::Entry::val(amp.i)};
      }

      return;
    }

    auto a = cn.mulCached(amplitude, edge.w);
    exportAmplitudesRec(edge.p->e[0], amplitudes, a, level - 1, idx << 1);
    exportAmplitudesRec(edge.p->e[1], amplitudes, a, level - 1,
                        (idx << 1) | 1ULL);
    cn.returnToCache(a);
  }
  void exportAmplitudes(const vEdge& edge,
                        std::vector<std::complex<dd::fp>>& amplitudes,
                        dd::QubitCount nq) {
    if (edge.isTerminal()) {
      // TODO special treatment
      return;
    }
    auto weight = cn.getCached(1., 0.);
    exportAmplitudesRec(edge, amplitudes, weight, nq, 0);
    cn.returnToCache(weight);
  }

  void addAmplitudesRec(const vEdge& edge,
                        std::vector<std::complex<dd::fp>>& amplitudes,
                        ComplexValue& amplitude, dd::QubitCount level,
                        std::size_t idx) {
    auto ar = dd::ComplexTable<>::Entry::val(edge.w.r);
    auto ai = dd::ComplexTable<>::Entry::val(edge.w.i);
    ComplexValue amp{ar * amplitude.r - ai * amplitude.i,
                     ar * amplitude.i + ai * amplitude.r};

    if (edge.isTerminal()) {
      idx <<= level;
      for (std::size_t i = 0; i < (1ULL << level); i++) {
        auto temp = std::complex<dd::fp>{amp.r + amplitudes[idx].real(),
                                         amp.i + amplitudes[idx].imag()};
        amplitudes[idx++] = temp;
      }

      return;
    }

    addAmplitudesRec(edge.p->e[0], amplitudes, amp, level - 1, idx << 1);
    addAmplitudesRec(edge.p->e[1], amplitudes, amp, level - 1, idx << 1 | 1ULL);
  }
  void addAmplitudes(const vEdge& edge,
                     std::vector<std::complex<dd::fp>>& amplitudes,
                     dd::QubitCount nq) {
    if (edge.isTerminal()) {
      // TODO special treatment
      return;
    }
    ComplexValue a{1., 0.};
    addAmplitudesRec(edge, amplitudes, a, nq, 0);
  }

  // transfers a decision diagram from another package to this package
  template <class Edge> Edge transfer(Edge& original) {
    // POST ORDER TRAVERSAL USING ONE STACK
    // https://www.geeksforgeeks.org/iterative-postorder-traversal-using-stack/
    Edge root{};
    std::stack<Edge*> stack;

    std::unordered_map<decltype(original.p), decltype(original.p)> mappedNode{};

    Edge* currentEdge = &original;
    if (!currentEdge->isTerminal()) {
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
          Edge* temp = stack.top();
          stack.pop();
          stack.push(currentEdge);
          currentEdge = temp;
        } else {
          if (mappedNode.find(currentEdge->p) != mappedNode.end()) {
            currentEdge = nullptr;
            continue;
          }
          std::array<Edge, n> edges{};
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

      auto w = cn.getCached(dd::ComplexTable<>::Entry::val(original.w.r),
                            dd::ComplexTable<>::Entry::val(original.w.i));
      dd::ComplexNumbers::mul(w, root.w, w);
      root.w = cn.lookup(w);
      cn.returnToCache(w);
    } else {
      root.p = original.p; // terminal -> static
      root.w = cn.lookup(original.w);
    }
    return root;
  }

  ///
  /// Deserialization
  /// Note: do not rely on the binary format being portable across different
  /// architectures/platforms
  ///

  template <class Node, class Edge = Edge<Node>,
            std::size_t N = std::tuple_size_v<decltype(Node::e)>>
  Edge deserialize(std::istream& is, bool readBinary = false) {
    auto result = Edge::zero;
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

    auto w = cn.getCached(rootweight.r, rootweight.i);
    ComplexNumbers::mul(w, result.w, w);
    result.w = cn.lookup(w);
    cn.returnToCache(w);

    return result;
  }

  template <class Node, class Edge = Edge<Node>>
  Edge deserialize(const std::string& inputFilename, bool readBinary) {
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
  Edge deserializeNode(std::int64_t index, Qubit v,
                       std::array<std::int64_t, N>& edgeIdx,
                       std::array<ComplexValue, N>& edgeWeight,
                       std::unordered_map<std::int64_t, Node*>& nodes) {
    if (index == -1) {
      return Edge::zero;
    }

    std::array<Edge, N> edges{};
    for (auto i = 0U; i < N; ++i) {
      if (edgeIdx[i] == -2) {
        edges[i] = Edge::zero;
      } else {
        if (edgeIdx[i] == -1) {
          edges[i] = Edge::one;
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

  ///
  /// Debugging
  ///
public:
  template <class Node> void debugnode(const Node* p) const {
    if (Node::isTerminal(p)) {
      std::clog << "terminal\n";
      return;
    }
    std::clog << "Debug node: " << debugnodeLine(p) << "\n";
    for (const auto& edge : p->e) {
      std::clog << "  " << std::hexfloat << std::setw(22)
                << CTEntry::val(edge.w.r) << " " << std::setw(22)
                << CTEntry::val(edge.w.i) << std::defaultfloat << "i --> "
                << debugnodeLine(edge.p) << "\n";
    }
    std::clog << std::flush;
  }

  template <class Node> std::string debugnodeLine(const Node* p) const {
    if (Node::isTerminal(p)) {
      return "terminal";
    }
    std::stringstream sst;
    sst << "0x" << std::hex << reinterpret_cast<std::uintptr_t>(p) << std::dec
        << "[v=" << static_cast<std::int64_t>(p->v) << " ref=" << p->ref
        << " hash=" << UniqueTable<Node>::hash(p) << "]";
    return sst.str();
  }

  template <class Edge> bool isLocallyConsistent(const Edge& e) {
    // NOLINTNEXTLINE(clang-diagnostic-float-equal)
    assert(Complex::one.r->value == 1 && Complex::one.i->value == 0);
    // NOLINTNEXTLINE(clang-diagnostic-float-equal)
    assert(Complex::zero.r->value == 0 && Complex::zero.i->value == 0);

    const bool result = isLocallyConsistent2(e);
    return result;
  }

  template <class Edge> bool isGloballyConsistent(const Edge& e) {
    std::map<ComplexTable<>::Entry*, std::size_t> weightCounter{};
    std::map<decltype(e.p), std::size_t> nodeCounter{};
    fillConsistencyCounter(e, weightCounter, nodeCounter);
    checkConsistencyCounter(e, weightCounter, nodeCounter);
    return true;
  }

private:
  template <class Edge> bool isLocallyConsistent2(const Edge& e) {
    const auto* ptrR = CTEntry::getAlignedPointer(e.w.r);
    const auto* ptrI = CTEntry::getAlignedPointer(e.w.i);

    if ((ptrR->refCount == 0 || ptrI->refCount == 0) && e.w != Complex::one &&
        e.w != Complex::zero) {
      std::clog << "\nLOCAL INCONSISTENCY FOUND\nOffending Number: " << e.w
                << " (" << ptrR->refCount << ", " << ptrI->refCount << ")\n\n";
      debugnode(e.p);
      return false;
    }

    if (e.isTerminal()) {
      return true;
    }

    if (!e.isTerminal() && e.p->ref == 0) {
      std::clog << "\nLOCAL INCONSISTENCY FOUND: RC==0\n";
      debugnode(e.p);
      return false;
    }

    for (const auto& child : e.p->e) {
      if (child.p->v + 1 != e.p->v && !child.isTerminal()) {
        std::clog << "\nLOCAL INCONSISTENCY FOUND: Wrong V\n";
        debugnode(e.p);
        return false;
      }
      if (!child.isTerminal() && child.p->ref == 0) {
        std::clog << "\nLOCAL INCONSISTENCY FOUND: RC==0\n";
        debugnode(e.p);
        return false;
      }
      if (!isLocallyConsistent2(child)) {
        return false;
      }
    }
    return true;
  }

  template <class Edge>
  void fillConsistencyCounter(
      const Edge& edge,
      std::map<ComplexTable<>::Entry*, std::size_t>& weightMap,
      std::map<decltype(edge.p), std::size_t>& nodeMap) {
    weightMap[CTEntry::getAlignedPointer(edge.w.r)]++;
    weightMap[CTEntry::getAlignedPointer(edge.w.i)]++;

    if (edge.isTerminal()) {
      return;
    }
    nodeMap[edge.p]++;
    for (auto& child : edge.p->e) {
      if (nodeMap[child.p] == 0) {
        fillConsistencyCounter(child, weightMap, nodeMap);
      } else {
        nodeMap[child.p]++;
        weightMap[CTEntry::getAlignedPointer(child.w.r)]++;
        weightMap[CTEntry::getAlignedPointer(child.w.i)]++;
      }
    }
  }

  template <class Edge>
  void checkConsistencyCounter(
      const Edge& edge,
      const std::map<ComplexTable<>::Entry*, std::size_t>& weightMap,
      const std::map<decltype(edge.p), std::size_t>& nodeMap) {
    auto* rPtr = CTEntry::getAlignedPointer(edge.w.r);
    auto* iPtr = CTEntry::getAlignedPointer(edge.w.i);

    if (weightMap.at(rPtr) > rPtr->refCount && rPtr != Complex::one.r &&
        rPtr != Complex::zero.i && rPtr != &ComplexTable<>::sqrt2_2) {
      std::clog << "\nOffending weight: " << edge.w << "\n";
      std::clog << "Bits: " << std::hexfloat << CTEntry::val(edge.w.r) << "r "
                << CTEntry::val(edge.w.i) << std::defaultfloat << "i\n";
      debugnode(edge.p);
      throw std::runtime_error("Ref-Count mismatch for " +
                               std::to_string(rPtr->value) +
                               "(r): " + std::to_string(weightMap.at(rPtr)) +
                               " occurrences in DD but Ref-Count is only " +
                               std::to_string(rPtr->refCount));
    }

    if (weightMap.at(iPtr) > iPtr->refCount && iPtr != Complex::zero.i &&
        iPtr != Complex::one.r && iPtr != &ComplexTable<>::sqrt2_2) {
      std::clog << "\nOffending weight: " << edge.w << "\n";
      std::clog << "Bits: " << std::hexfloat << CTEntry::val(edge.w.r) << "r "
                << CTEntry::val(edge.w.i) << std::defaultfloat << "i\n";
      debugnode(edge.p);
      throw std::runtime_error("Ref-Count mismatch for " +
                               std::to_string(iPtr->value) +
                               "(i): " + std::to_string(weightMap.at(iPtr)) +
                               " occurrences in DD but Ref-Count is only " +
                               std::to_string(iPtr->refCount));
    }

    if (edge.isTerminal()) {
      return;
    }

    if (nodeMap.at(edge.p) != edge.p->ref) {
      debugnode(edge.p);
      throw std::runtime_error(
          "Ref-Count mismatch for node: " + std::to_string(nodeMap.at(edge.p)) +
          " occurrences in DD but Ref-Count is " + std::to_string(edge.p->ref));
    }
    for (auto child : edge.p->e) {
      if (!child.isTerminal() && child.p->v != edge.p->v - 1) {
        std::clog << "child.p->v == " << child.p->v << "\n";
        std::clog << " edge.p->v == " << edge.p->v << "\n";
        debugnode(child.p);
        debugnode(edge.p);
        throw std::runtime_error("Variable level ordering seems wrong");
      }
      checkConsistencyCounter(child, weightMap, nodeMap);
    }
  }

  ///
  /// Printing and Statistics
  ///
public:
  // print information on package and its members
  static void printInformation() {
    std::cout << "\n  compiled: " << __DATE__ << " " << __TIME__
              << "\n  Complex size: " << sizeof(Complex) << " bytes (aligned "
              << alignof(Complex) << " bytes)"
              << "\n  ComplexValue size: " << sizeof(ComplexValue)
              << " bytes (aligned " << alignof(ComplexValue) << " bytes)"
              << "\n  ComplexNumbers size: " << sizeof(ComplexNumbers)
              << " bytes (aligned " << alignof(ComplexNumbers) << " bytes)"
              << "\n  vEdge size: " << sizeof(vEdge) << " bytes (aligned "
              << alignof(vEdge) << " bytes)"
              << "\n  vNode size: " << sizeof(vNode) << " bytes (aligned "
              << alignof(vNode) << " bytes)"
              << "\n  mEdge size: " << sizeof(mEdge) << " bytes (aligned "
              << alignof(mEdge) << " bytes)"
              << "\n  mNode size: " << sizeof(mNode) << " bytes (aligned "
              << alignof(mNode) << " bytes)"
              << "\n  CT Vector Add size: "
              << sizeof(typename decltype(vectorAdd)::Entry)
              << " bytes (aligned "
              << alignof(typename decltype(vectorAdd)::Entry) << " bytes)"
              << "\n  CT Matrix Add size: "
              << sizeof(typename decltype(matrixAdd)::Entry)
              << " bytes (aligned "
              << alignof(typename decltype(matrixAdd)::Entry) << " bytes)"
              << "\n  CT Matrix Transpose size: "
              << sizeof(typename decltype(matrixTranspose)::Entry)
              << " bytes (aligned "
              << alignof(typename decltype(matrixTranspose)::Entry) << " bytes)"
              << "\n  CT Conjugate Matrix Transpose size: "
              << sizeof(typename decltype(conjugateMatrixTranspose)::Entry)
              << " bytes (aligned "
              << alignof(typename decltype(conjugateMatrixTranspose)::Entry)
              << " bytes)"
              << "\n  CT Matrix Multiplication size: "
              << sizeof(typename decltype(matrixMatrixMultiplication)::Entry)
              << " bytes (aligned "
              << alignof(typename decltype(matrixMatrixMultiplication)::Entry)
              << " bytes)"
              << "\n  CT Matrix Vector Multiplication size: "
              << sizeof(typename decltype(matrixVectorMultiplication)::Entry)
              << " bytes (aligned "
              << alignof(typename decltype(matrixVectorMultiplication)::Entry)
              << " bytes)"
              << "\n  CT Vector Inner Product size: "
              << sizeof(typename decltype(vectorInnerProduct)::Entry)
              << " bytes (aligned "
              << alignof(typename decltype(vectorInnerProduct)::Entry)
              << " bytes)"
              << "\n  CT Vector Kronecker size: "
              << sizeof(typename decltype(vectorKronecker)::Entry)
              << " bytes (aligned "
              << alignof(typename decltype(vectorKronecker)::Entry) << " bytes)"
              << "\n  CT Matrix Kronecker size: "
              << sizeof(typename decltype(matrixKronecker)::Entry)
              << " bytes (aligned "
              << alignof(typename decltype(matrixKronecker)::Entry) << " bytes)"
              << "\n  ToffoliTable::Entry size: "
              << sizeof(ToffoliTable<mEdge>::Entry) << " bytes (aligned "
              << alignof(ToffoliTable<mEdge>::Entry) << " bytes)"
              << "\n  Package size: " << sizeof(Package) << " bytes (aligned "
              << alignof(Package) << " bytes)"
              << "\n"
              << std::flush;
  }

  // print unique and compute table statistics
  void statistics() {
    std::cout << "DD statistics:" << std::endl << "[vUniqueTable] ";
    vUniqueTable.printStatistics();
    std::cout << "[mUniqueTable] ";
    mUniqueTable.printStatistics();
    std::cout << "[CT Vector Add] ";
    vectorAdd.printStatistics();
    std::cout << "[CT Matrix Add] ";
    matrixAdd.printStatistics();
    std::cout << "[CT Matrix Transpose] ";
    matrixTranspose.printStatistics();
    std::cout << "[CT Conjugate Matrix Transpose] ";
    conjugateMatrixTranspose.printStatistics();
    std::cout << "[CT Matrix Multiplication] ";
    matrixMatrixMultiplication.printStatistics();
    std::cout << "[CT Matrix Vector Multiplication] ";
    matrixVectorMultiplication.printStatistics();
    std::cout << "[CT Inner Product] ";
    vectorInnerProduct.printStatistics();
    std::cout << "[CT Vector Kronecker] ";
    vectorKronecker.printStatistics();
    std::cout << "[CT Matrix Kronecker] ";
    matrixKronecker.printStatistics();
    std::cout << "[Toffoli Table] ";
    toffoliTable.printStatistics();
    std::cout << "[Stochastic Noise Table] ";
    stochasticNoiseOperationCache.printStatistics();
    std::cout << "[CT Density Add] ";
    densityAdd.printStatistics();
    std::cout << "[CT Density Mul] ";
    densityDensityMultiplication.printStatistics();
    std::cout << "[CT Density Noise] ";
    densityNoise.printStatistics();
    std::cout << "[ComplexTable] ";
    cn.complexTable.printStatistics();
  }
};

} // namespace dd
