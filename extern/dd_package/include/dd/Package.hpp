/*
 * This file is part of the JKQ DD Package which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum_dd/ for more information.
 */

#ifndef DDpackage_H
#define DDpackage_H

#include "Complex.hpp"
#include "ComplexCache.hpp"
#include "ComplexNumbers.hpp"
#include "ComplexTable.hpp"
#include "ComplexValue.hpp"
#include "ComputeTable.hpp"
#include "Control.hpp"
#include "Definitions.hpp"
#include "Edge.hpp"
#include "GateMatrixDefinitions.hpp"
#include "NoiseOperationTable.hpp"
#include "ToffoliTable.hpp"
#include "UnaryComputeTable.hpp"
#include "UniqueTable.hpp"

#include <algorithm>
#include <array>
#include <bitset>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <regex>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace dd {
    class Package {
        ///
        /// Complex number handling
        ///
    public:
        ComplexNumbers cn{};

        ///
        /// Construction, destruction, information and reset
        ///
    public:
        static constexpr std::size_t maxPossibleQubits = static_cast<std::make_unsigned_t<Qubit>>(std::numeric_limits<Qubit>::max()) + 1U;
        static constexpr std::size_t defaultQubits     = 128;
        explicit Package(std::size_t nq = defaultQubits):
            cn(ComplexNumbers()), nqubits(nq) {
            resize(nq);
        };
        ~Package()                      = default;
        Package(const Package& package) = delete;
        Package& operator=(const Package& package) = delete;

        // resize the package instance
        void resize(std::size_t nq) {
            if (nq > maxPossibleQubits) {
                throw std::invalid_argument("Requested too many qubits from package. Qubit datatype only allows up to " +
                                            std::to_string(maxPossibleQubits) + " qubits, while " +
                                            std::to_string(nq) + " were requested. Please recompile the package with a wider Qubit type!");
            }
            nqubits = nq;
            vUniqueTable.resize(nqubits);
            mUniqueTable.resize(nqubits);
            noiseOperationTable.resize(nqubits);
            IdTable.resize(nqubits);
        }

        // reset package state
        void reset() {
            clearUniqueTables();
            clearComputeTables();
        }

        // getter for qubits
        [[nodiscard]] auto qubits() const { return nqubits; }

    private:
        std::size_t nqubits;

        ///
        /// Vector nodes, edges and quantum states
        ///
    public:
        struct vNode {
            std::array<Edge<vNode>, RADIX> e{};    // edges out of this node
            vNode*                         next{}; // used to link nodes in unique table
            RefCount                       ref{};  // reference count
            Qubit                          v{};    // variable index (nonterminal) value (-1 for terminal)

            static vNode            terminalNode;
            constexpr static vNode* terminal{&terminalNode};

            static constexpr bool isTerminal(const vNode* p) { return p == terminal; }
        };
        using vEdge       = Edge<vNode>;
        using vCachedEdge = CachedEdge<vNode>;

        vEdge normalize(const vEdge& e, bool cached) {
            auto argmax = -1;

            auto zero = std::array{e.p->e[0].w.approximatelyZero(), e.p->e[1].w.approximatelyZero()};

            // make sure to release cached numbers approximately zero, but not exactly zero
            if (cached) {
                for (auto i = 0U; i < RADIX; i++) {
                    if (zero[i] && e.p->e[i].w != Complex::zero) {
                        cn.returnToCache(e.p->e[i].w);
                        e.p->e[i] = vEdge::zero;
                    }
                }
            }

            fp sum = 0.;
            fp div = 0.;
            for (auto i = 0U; i < RADIX; ++i) {
                if (e.p->e[i].p == nullptr || zero[i]) {
                    continue;
                }

                if (argmax == -1) {
                    argmax = static_cast<decltype(argmax)>(i);
                    div    = ComplexNumbers::mag2(e.p->e[i].w);
                    sum    = div;
                } else {
                    sum += ComplexNumbers::mag2(e.p->e[i].w);
                }
            }

            // all equal to zero
            if (argmax == -1) {
                if (!cached && !e.isTerminal()) {
                    // If it is not a cached computation, the node has to be put back into the chain
                    vUniqueTable.returnNode(e.p);
                }
                return vEdge::zero;
            }

            sum = std::sqrt(sum / div);

            auto  r   = e;
            auto& max = r.p->e[argmax];
            if (cached && max.w != Complex::one) {
                r.w = max.w;
                r.w.r->value *= sum;
                r.w.i->value *= sum;
            } else {
                r.w = cn.lookup(CTEntry::val(max.w.r) * sum, CTEntry::val(max.w.i) * sum);
                if (r.w.approximatelyZero()) {
                    return vEdge::zero;
                }
            }
            max.w = cn.lookup(static_cast<fp>(1.0) / sum, 0.);
            if (max.w == Complex::zero)
                max = vEdge::zero;

            auto  argmin = (argmax + 1) % 2;
            auto& min    = r.p->e[argmin];
            if (!zero[argmin]) {
                if (cached) {
                    cn.returnToCache(min.w);
                    ComplexNumbers::div(min.w, min.w, r.w);
                    min.w = cn.lookup(min.w);
                    if (min.w == Complex::zero) {
                        min = vEdge::zero;
                    }
                } else {
                    auto c = cn.getTemporary();
                    ComplexNumbers::div(c, min.w, r.w);
                    min.w = cn.lookup(c);
                    if (min.w == Complex::zero) {
                        min = vEdge::zero;
                    }
                }
            }

            return r;
        }

        // generate |0...0> with n qubits
        vEdge makeZeroState(QubitCount n, std::size_t start = 0) {
            if (n + start > nqubits) {
                throw std::runtime_error("Requested state with " +
                                         std::to_string(n + start) +
                                         " qubits, but current package configuration only supports up to " +
                                         std::to_string(nqubits) +
                                         " qubits. Please allocate a larger package instance.");
            }
            auto f = vEdge::one;
            for (std::size_t p = start; p < n + start; p++) {
                f = makeDDNode(static_cast<Qubit>(p), std::array{f, vEdge::zero});
            }
            return f;
        }
        // generate computational basis state |i> with n qubits
        vEdge makeBasisState(QubitCount n, const std::vector<bool>& state, std::size_t start = 0) {
            if (n + start > nqubits) {
                throw std::runtime_error("Requested state with " +
                                         std::to_string(n + start) +
                                         " qubits, but current package configuration only supports up to " +
                                         std::to_string(nqubits) +
                                         " qubits. Please allocate a larger package instance.");
            }
            auto f = vEdge::one;
            for (std::size_t p = start; p < n + start; ++p) {
                if (state[p] == 0) {
                    f = makeDDNode(static_cast<Qubit>(p), std::array{f, vEdge::zero});
                } else {
                    f = makeDDNode(static_cast<Qubit>(p), std::array{vEdge::zero, f});
                }
            }
            return f;
        }
        // generate general basis state with n qubits
        vEdge makeBasisState(QubitCount n, const std::vector<BasisStates>& state, std::size_t start = 0) {
            if (n + start > nqubits) {
                throw std::runtime_error("Requested state with " +
                                         std::to_string(n + start) +
                                         " qubits, but current package configuration only supports up to " +
                                         std::to_string(nqubits) +
                                         " qubits. Please allocate a larger package instance.");
            }
            if (state.size() < n) {
                throw std::runtime_error("Insufficient qubit states provided. Requested " + std::to_string(n) + ", but received " + std::to_string(state.size()));
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
                        f = makeDDNode(static_cast<Qubit>(p), std::array<vEdge, RADIX>{{{f.p, cn.lookup(dd::SQRT2_2, 0)}, {f.p, cn.lookup(dd::SQRT2_2, 0)}}});
                        break;
                    case BasisStates::minus:
                        f = makeDDNode(static_cast<Qubit>(p), std::array<vEdge, RADIX>{{{f.p, cn.lookup(dd::SQRT2_2, 0)}, {f.p, cn.lookup(-dd::SQRT2_2, 0)}}});
                        break;
                    case BasisStates::right:
                        f = makeDDNode(static_cast<Qubit>(p), std::array<vEdge, RADIX>{{{f.p, cn.lookup(dd::SQRT2_2, 0)}, {f.p, cn.lookup(0, dd::SQRT2_2)}}});
                        break;
                    case BasisStates::left:
                        f = makeDDNode(static_cast<Qubit>(p), std::array<vEdge, RADIX>{{{f.p, cn.lookup(dd::SQRT2_2, 0)}, {f.p, cn.lookup(0, -dd::SQRT2_2)}}});
                        break;
                }
            }
            return f;
        }

        ///
        /// Matrix nodes, edges and quantum gates
        ///
    public:
        struct mNode {
            std::array<Edge<mNode>, NEDGE> e{};           // edges out of this node
            mNode*                         next{};        // used to link nodes in unique table
            RefCount                       ref{};         // reference count
            Qubit                          v{};           // variable index (nonterminal) value (-1 for terminal)
            bool                           symm  = false; // node is symmetric
            bool                           ident = false; // node resembles identity

            static mNode            terminalNode;
            constexpr static mNode* terminal{&terminalNode};

            static constexpr bool isTerminal(const mNode* p) { return p == terminal; }
        };
        using mEdge       = Edge<mNode>;
        using mCachedEdge = CachedEdge<mNode>;

        mEdge normalize(const mEdge& e, bool cached) {
            auto argmax = -1;

            auto zero = std::array{e.p->e[0].w.approximatelyZero(),
                                   e.p->e[1].w.approximatelyZero(),
                                   e.p->e[2].w.approximatelyZero(),
                                   e.p->e[3].w.approximatelyZero()};

            // make sure to release cached numbers approximately zero, but not exactly zero
            if (cached) {
                for (auto i = 0U; i < NEDGE; i++) {
                    if (zero[i] && e.p->e[i].w != Complex::zero) {
                        cn.returnToCache(e.p->e[i].w);
                        e.p->e[i] = mEdge::zero;
                    }
                }
            }

            fp   max  = 0;
            auto maxc = Complex::one;
            // determine max amplitude
            for (auto i = 0U; i < NEDGE; ++i) {
                if (zero[i]) continue;
                if (argmax == -1) {
                    argmax = static_cast<decltype(argmax)>(i);
                    max    = ComplexNumbers::mag2(e.p->e[i].w);
                    maxc   = e.p->e[i].w;
                } else {
                    auto mag = ComplexNumbers::mag2(e.p->e[i].w);
                    if (mag - max > ComplexTable<>::tolerance()) {
                        argmax = static_cast<decltype(argmax)>(i);
                        max    = mag;
                        maxc   = e.p->e[i].w;
                    }
                }
            }

            // all equal to zero
            if (argmax == -1) {
                if (!cached && !e.isTerminal()) {
                    // If it is not a cached computation, the node has to be put back into the chain
                    mUniqueTable.returnNode(e.p);
                }
                return mEdge::zero;
            }

            auto r = e;
            // divide each entry by max
            for (auto i = 0U; i < NEDGE; ++i) {
                if (static_cast<decltype(argmax)>(i) == argmax) {
                    if (cached) {
                        if (r.w == Complex::one)
                            r.w = maxc;
                        else
                            ComplexNumbers::mul(r.w, r.w, maxc);
                    } else {
                        if (r.w == Complex::one) {
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
                        if (cached && r.p->e[i].w != Complex::zero)
                            cn.returnToCache(r.p->e[i].w);
                        r.p->e[i] = mEdge::zero;
                        continue;
                    }
                    if (cached && !zero[i] && r.p->e[i].w != Complex::one) {
                        cn.returnToCache(r.p->e[i].w);
                    }
                    if (r.p->e[i].w.approximatelyOne())
                        r.p->e[i].w = Complex::one;
                    auto c = cn.getTemporary();
                    ComplexNumbers::div(c, r.p->e[i].w, maxc);
                    r.p->e[i].w = cn.lookup(c);
                }
            }
            return r;
        }

        // build matrix representation for a single gate on an n-qubit circuit
        mEdge makeGateDD(const std::array<ComplexValue, NEDGE>& mat, QubitCount n, Qubit target, std::size_t start = 0) {
            return makeGateDD(mat, n, Controls{}, target, start);
        }
        mEdge makeGateDD(const std::array<ComplexValue, NEDGE>& mat, QubitCount n, const Control& control, Qubit target, std::size_t start = 0) {
            return makeGateDD(mat, n, Controls{control}, target, start);
        }
        mEdge makeGateDD(const std::array<ComplexValue, NEDGE>& mat, QubitCount n, const Controls& controls, Qubit target, std::size_t start = 0) {
            if (n + start > nqubits) {
                throw std::runtime_error("Requested gate with " +
                                         std::to_string(n + start) +
                                         " qubits, but current package configuration only supports up to " +
                                         std::to_string(nqubits) +
                                         " qubits. Please allocate a larger package instance.");
            }
            std::array<mEdge, NEDGE> em{};
            auto                     it = controls.begin();
            for (auto i = 0U; i < NEDGE; ++i) {
                if (mat[i].r == 0 && mat[i].i == 0) {
                    em[i] = mEdge::zero;
                } else {
                    em[i] = mEdge::terminal(cn.lookup(mat[i]));
                }
            }

            //process lines below target
            auto z = static_cast<Qubit>(start);
            for (; z < target; z++) {
                for (auto i1 = 0U; i1 < RADIX; i1++) {
                    for (auto i2 = 0U; i2 < RADIX; i2++) {
                        auto i = i1 * RADIX + i2;
                        if (it != controls.end() && it->qubit == z) {
                            if (it->type == Control::Type::neg) { // neg. control
                                em[i] = makeDDNode(z, std::array{em[i], mEdge::zero, mEdge::zero, (i1 == i2) ? makeIdent(static_cast<Qubit>(start), static_cast<Qubit>(z - 1)) : mEdge::zero});
                            } else { // pos. control
                                em[i] = makeDDNode(z, std::array{(i1 == i2) ? makeIdent(static_cast<Qubit>(start), static_cast<Qubit>(z - 1)) : mEdge::zero, mEdge::zero, mEdge::zero, em[i]});
                            }
                        } else { // not connected
                            em[i] = makeDDNode(z, std::array{em[i], mEdge::zero, mEdge::zero, em[i]});
                        }
                    }
                }
                if (it != controls.end() && it->qubit == z) {
                    ++it;
                }
            }

            // target line
            auto e = makeDDNode(z, em);

            //process lines above target
            for (; z < static_cast<Qubit>(n - 1 + start); z++) {
                auto q = static_cast<Qubit>(z + 1);
                if (it != controls.end() && it->qubit == q) {
                    if (it->type == Control::Type::neg) { // neg. control
                        e = makeDDNode(q, std::array{e, mEdge::zero, mEdge::zero, makeIdent(static_cast<Qubit>(start), static_cast<Qubit>(q - 1))});
                    } else { // pos. control
                        e = makeDDNode(q, std::array{makeIdent(static_cast<Qubit>(start), static_cast<Qubit>(q - 1)), mEdge::zero, mEdge::zero, e});
                    }
                    ++it;
                } else { // not connected
                    e = makeDDNode(q, std::array{e, mEdge::zero, mEdge::zero, e});
                }
            }
            return e;
        }

        mEdge makeSWAPDD(QubitCount n, const Controls& controls, Qubit target0, Qubit target1, std::size_t start = 0) {
            auto c = controls;
            c.insert(Control{target0});
            mEdge e = makeGateDD(Xmat, n, c, target1, start);
            c.erase(Control{target0});
            c.insert(Control{target1});
            e = multiply(e, multiply(makeGateDD(Xmat, n, c, target0, start), e));
            return e;
        }

        mEdge makePeresDD(QubitCount n, const Controls& controls, Qubit target0, Qubit target1, std::size_t start = 0) {
            auto c = controls;
            c.insert(Control{target1});
            mEdge e = makeGateDD(Xmat, n, c, target0, start);
            e       = multiply(makeGateDD(Xmat, n, controls, target1, start), e);
            return e;
        }

        mEdge makePeresdagDD(QubitCount n, const Controls& controls, Qubit target0, Qubit target1, std::size_t start = 0) {
            mEdge e = makeGateDD(Xmat, n, controls, target1, start);
            auto  c = controls;
            c.insert(Control{target1});
            e = multiply(makeGateDD(Xmat, n, c, target0, start), e);
            return e;
        }

        mEdge makeiSWAPDD(QubitCount n, const Controls& controls, Qubit target0, Qubit target1, std::size_t start = 0) {
            mEdge e = makeGateDD(Smat, n, controls, target1, start);              // S q[1]
            e       = multiply(e, makeGateDD(Smat, n, controls, target0, start)); // S q[0]
            e       = multiply(e, makeGateDD(Hmat, n, controls, target0, start)); // H q[0]
            auto c  = controls;
            c.insert(Control{target0});
            e = multiply(e, makeGateDD(Xmat, n, c, target1, start)); // CX q[0], q[1]
            c.erase(Control{target0});
            c.insert(Control{target1});
            e = multiply(e, makeGateDD(Xmat, n, c, target0, start));        // CX q[1], q[0]
            e = multiply(e, makeGateDD(Hmat, n, controls, target1, start)); // H q[1]
            return e;
        }

        mEdge makeiSWAPinvDD(QubitCount n, const Controls& controls, Qubit target0, Qubit target1, std::size_t start = 0) {
            mEdge e = makeGateDD(Hmat, n, controls, target1, start); // H q[1]
            auto  c = controls;
            c.insert(Control{target1});
            e = multiply(e, makeGateDD(Xmat, n, c, target0, start)); // CX q[1], q[0]
            c.erase(Control{target1});
            c.insert(Control{target0});
            e = multiply(e, makeGateDD(Xmat, n, c, target1, start));           // CX q[0], q[1]
            e = multiply(e, makeGateDD(Hmat, n, controls, target0, start));    // H q[0]
            e = multiply(e, makeGateDD(Sdagmat, n, controls, target0, start)); // Sdag q[0]
            e = multiply(e, makeGateDD(Sdagmat, n, controls, target1, start)); // Sdag q[1]
            return e;
        }

    private:
        // check whether node represents a symmetric matrix or the identity
        void checkSpecialMatrices(mNode* p) {
            if (p->v == -1)
                return;

            p->ident = false; // assume not identity
            p->symm  = false; // assume symmetric

            // check if matrix is symmetric
            if (!p->e[0].p->symm || !p->e[3].p->symm) return;
            if (transpose(p->e[1]) != p->e[2]) return;
            p->symm = true;

            // check if matrix resembles identity
            if (!(p->e[0].p->ident) || (p->e[1].w) != Complex::zero || (p->e[2].w) != Complex::zero || (p->e[0].w) != Complex::one ||
                (p->e[3].w) != Complex::one || !(p->e[3].p->ident))
                return;
            p->ident = true;
        }

        ///
        /// Unique tables, Reference counting and garbage collection
        ///
    public:
        // unique tables
        template<class Node>
        [[nodiscard]] UniqueTable<Node>& getUniqueTable();

        template<class Node>
        void incRef(const Edge<Node>& e) {
            getUniqueTable<Node>().incRef(e);
        }
        template<class Node>
        void decRef(const Edge<Node>& e) {
            getUniqueTable<Node>().decRef(e);
        }

        UniqueTable<vNode> vUniqueTable{nqubits};
        UniqueTable<mNode> mUniqueTable{nqubits};

        bool garbageCollect(bool force = false) {
            // return immediately if no table needs collection
            if (!force &&
                !vUniqueTable.possiblyNeedsCollection() &&
                !mUniqueTable.possiblyNeedsCollection() &&
                !cn.complexTable.possiblyNeedsCollection()) {
                return false;
            }

            auto vCollect = vUniqueTable.garbageCollect(force);
            auto mCollect = mUniqueTable.garbageCollect(force);
            auto cCollect = cn.garbageCollect(force);

            // invalidate all compute tables involving vectors if any vector node has been collected
            if (vCollect > 0) {
                vectorAdd.clear();
                vectorInnerProduct.clear();
                vectorKronecker.clear();
                matrixVectorMultiplication.clear();
            }
            // invalidate all compute tables involving matrices if any matrix node has been collected
            if (mCollect > 0) {
                matrixAdd.clear();
                matrixTranspose.clear();
                conjugateMatrixTranspose.clear();
                matrixKronecker.clear();
                matrixVectorMultiplication.clear();
                matrixMatrixMultiplication.clear();
                toffoliTable.clear();
                clearIdentityTable();
                noiseOperationTable.clear();
            }
            // invalidate all compute tables where any component of the entry contains numbers from the complex table if any complex numbers were collected
            if (cCollect > 0) {
                matrixVectorMultiplication.clear();
                matrixMatrixMultiplication.clear();
                matrixTranspose.clear();
                conjugateMatrixTranspose.clear();
                vectorInnerProduct.clear();
                vectorKronecker.clear();
                matrixKronecker.clear();
                noiseOperationTable.clear();
            }
            return vCollect > 0 || mCollect > 0 || cCollect > 0;
        }

        void clearUniqueTables() {
            vUniqueTable.clear();
            mUniqueTable.clear();
        }

        // create a normalized DD node and return an edge pointing to it. The node is not recreated if it already exists.
        template<class Node>
        Edge<Node> makeDDNode(Qubit var, const std::array<Edge<Node>, std::tuple_size_v<decltype(Node::e)>>& edges, bool cached = false) {
            auto&      uniqueTable = getUniqueTable<Node>();
            Edge<Node> e{uniqueTable.getNode(), Complex::one};
            e.p->v = var;
            e.p->e = edges;

            assert(e.p->ref == 0);
            for ([[maybe_unused]] const auto& edge: edges)
                assert(edge.p->v == var - 1 || edge.isTerminal());

            // normalize it
            e = normalize(e, cached);
            assert(e.p->v == var || e.isTerminal());

            // look it up in the unique tables
            auto l = uniqueTable.lookup(e, false);
            assert(l.p->v == var || l.isTerminal());

            // set specific node properties for matrices
            if constexpr (std::tuple_size_v<decltype(Node::e)> == NEDGE) {
                if (l.p == e.p)
                    checkSpecialMatrices(l.p);
            }

            return l;
        }

        template<class Node>
        Edge<Node> deleteEdge(const Edge<Node>& e, dd::Qubit v, std::size_t edgeIdx) {
            std::unordered_map<Node*, Edge<Node>> nodes{};
            return deleteEdge(e, v, edgeIdx, nodes);
        }

    private:
        template<class Node>
        Edge<Node> deleteEdge(const Edge<Node>& e, dd::Qubit v, std::size_t edgeIdx, std::unordered_map<Node*, Edge<Node>>& nodes) {
            if (e.p == nullptr || e.isTerminal()) {
                return e;
            }

            const auto& nodeit = nodes.find(e.p);
            Edge<Node>  newedge{};
            if (nodeit != nodes.end()) {
                newedge = nodeit->second;
            } else {
                constexpr std::size_t     N = std::tuple_size_v<decltype(e.p->e)>;
                std::array<Edge<Node>, N> edges{};
                if (e.p->v == v) {
                    for (std::size_t i = 0; i < N; i++) {
                        edges[i] = i == edgeIdx ? Edge<Node>::zero : e.p->e[i]; // optimization -> node cannot occur below again, since dd is assumed to be free
                    }
                } else {
                    for (std::size_t i = 0; i < N; i++) {
                        edges[i] = deleteEdge(e.p->e[i], v, edgeIdx, nodes);
                    }
                }

                newedge    = makeDDNode(e.p->v, edges);
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

            noiseOperationTable.clear();
        }

        ///
        /// Addition
        ///
    public:
        ComputeTable<vCachedEdge, vCachedEdge, vCachedEdge> vectorAdd{};
        ComputeTable<mCachedEdge, mCachedEdge, mCachedEdge> matrixAdd{};

        template<class Node>
        [[nodiscard]] ComputeTable<CachedEdge<Node>, CachedEdge<Node>, CachedEdge<Node>>& getAddComputeTable();

        template<class Edge>
        Edge add(const Edge& x, const Edge& y) {
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

    public:
        template<class Node>
        Edge<Node> add2(const Edge<Node>& x, const Edge<Node>& y) {
            if (x.p == nullptr) return y;
            if (y.p == nullptr) return x;

            if (x.w == Complex::zero) {
                if (y.w == Complex::zero) return y;
                auto r = y;
                r.w    = cn.getCached(CTEntry::val(y.w.r), CTEntry::val(y.w.i));
                return r;
            }
            if (y.w == Complex::zero) {
                auto r = x;
                r.w    = cn.getCached(CTEntry::val(x.w.r), CTEntry::val(x.w.i));
                return r;
            }
            if (x.p == y.p) {
                auto r = y;
                r.w    = cn.addCached(x.w, y.w);
                if (r.w.approximatelyZero()) {
                    cn.returnToCache(r.w);
                    return Edge<Node>::zero;
                }
                return r;
            }

            auto& computeTable = getAddComputeTable<Node>();
            auto  r            = computeTable.lookup({x.p, x.w}, {y.p, y.w});
            if (r.p != nullptr) {
                if (r.w.approximatelyZero()) {
                    return Edge<Node>::zero;
                } else {
                    return {r.p, cn.getCached(r.w)};
                }
            }

            Qubit w;
            if (x.isTerminal()) {
                w = y.p->v;
            } else {
                w = x.p->v;
                if (!y.isTerminal() && y.p->v > w) {
                    w = y.p->v;
                }
            }

            constexpr std::size_t     N = std::tuple_size_v<decltype(x.p->e)>;
            std::array<Edge<Node>, N> edge{};
            for (auto i = 0U; i < N; i++) {
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

                edge[i] = add2(e1, e2);

                if (!x.isTerminal() && x.p->v == w && e1.w != Complex::zero) {
                    cn.returnToCache(e1.w);
                }

                if (!y.isTerminal() && y.p->v == w && e2.w != Complex::zero) {
                    cn.returnToCache(e2.w);
                }
            }

            auto e = makeDDNode(w, edge, true);
            computeTable.insert({x.p, x.w}, {y.p, y.w}, {e.p, e.w});
            return e;
        }

        ///
        /// Matrix (conjugate) transpose
        ///
    public:
        UnaryComputeTable<mEdge, mEdge, 4096> matrixTranspose{};
        UnaryComputeTable<mEdge, mEdge, 4096> conjugateMatrixTranspose{};

        mEdge transpose(const mEdge& a) {
            if (a.p == nullptr || a.isTerminal() || a.p->symm) {
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
            if (a.p == nullptr)
                return a;
            if (a.isTerminal()) { // terminal case
                auto r = a;
                r.w    = ComplexNumbers::conj(a.w);
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
    public:
        ComputeTable<mEdge, vEdge, vCachedEdge> matrixVectorMultiplication{};
        ComputeTable<mEdge, mEdge, mCachedEdge> matrixMatrixMultiplication{};

        template<class LeftOperandNode, class RightOperandNode>
        [[nodiscard]] ComputeTable<Edge<LeftOperandNode>, Edge<RightOperandNode>, CachedEdge<RightOperandNode>>& getMultiplicationComputeTable();

        template<class LeftOperand, class RightOperand>
        RightOperand multiply(const LeftOperand& x, const RightOperand& y, dd::Qubit start = 0) {
            [[maybe_unused]] const auto before = cn.cacheCount();

            Qubit var = -1;
            if (!x.isTerminal()) {
                var = x.p->v;
            }
            if (!y.isTerminal() && (y.p->v) > var) {
                var = y.p->v;
            }

            auto e = multiply2(x, y, var, start);

            if (e.w != Complex::zero && e.w != Complex::one) {
                cn.returnToCache(e.w);
                e.w = cn.lookup(e.w);
            }

            [[maybe_unused]] const auto after = cn.cacheCount();
            assert(before == after);

            return e;
        }

    private:
        template<class LeftOperandNode, class RightOperandNode>
        Edge<RightOperandNode> multiply2(const Edge<LeftOperandNode>& x, const Edge<RightOperandNode>& y, Qubit var, Qubit start = 0) {
            using LEdge      = Edge<LeftOperandNode>;
            using REdge      = Edge<RightOperandNode>;
            using ResultEdge = Edge<RightOperandNode>;

            if (x.p == nullptr) return {nullptr, Complex::zero};
            if (y.p == nullptr) return y;

            if (x.w == Complex::zero || y.w == Complex::zero) {
                return ResultEdge::zero;
            }

            if (var == start - 1) {
                return ResultEdge::terminal(cn.mulCached(x.w, y.w));
            }

            auto xCopy = x;
            xCopy.w    = Complex::one;
            auto yCopy = y;
            yCopy.w    = Complex::one;

            auto& computeTable = getMultiplicationComputeTable<LeftOperandNode, RightOperandNode>();
            auto  r            = computeTable.lookup(xCopy, yCopy);
            if (r.p != nullptr) {
                if (r.w.approximatelyZero()) {
                    return ResultEdge::zero;
                } else {
                    auto e = ResultEdge{r.p, cn.getCached(r.w)};
                    ComplexNumbers::mul(e.w, e.w, x.w);
                    ComplexNumbers::mul(e.w, e.w, y.w);
                    if (e.w.approximatelyZero()) {
                        cn.returnToCache(e.w);
                        return ResultEdge::zero;
                    }
                    return e;
                }
            }

            constexpr std::size_t N = std::tuple_size_v<decltype(y.p->e)>;

            ResultEdge e{};
            if (x.p->v == var && x.p->v == y.p->v) {
                if (x.p->ident) {
                    if constexpr (N == NEDGE) {
                        // additionally check if y is the identity in case of matrix multiplication
                        if (y.p->ident) {
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

                if constexpr (N == NEDGE) {
                    // additionally check if y is the identity in case of matrix multiplication
                    if (y.p->ident) {
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

            constexpr std::size_t ROWS = RADIX;
            constexpr std::size_t COLS = N == NEDGE ? RADIX : 1U;

            std::array<ResultEdge, N> edge{};
            for (auto i = 0U; i < ROWS; i++) {
                for (auto j = 0U; j < COLS; j++) {
                    auto idx  = COLS * i + j;
                    edge[idx] = ResultEdge::zero;
                    for (auto k = 0U; k < ROWS; k++) {
                        LEdge e1{};
                        if (!x.isTerminal() && x.p->v == var) {
                            e1 = x.p->e[ROWS * i + k];
                        } else {
                            e1 = xCopy;
                        }

                        REdge e2{};
                        if (!y.isTerminal() && y.p->v == var) {
                            e2 = y.p->e[j + COLS * k];
                        } else {
                            e2 = yCopy;
                        }

                        auto m = multiply2(e1, e2, static_cast<Qubit>(var - 1), start);

                        if (k == 0 || edge[idx].w == Complex::zero) {
                            edge[idx] = m;
                        } else if (m.w != Complex::zero) {
                            auto old_e = edge[idx];
                            edge[idx]  = add2(edge[idx], m);
                            cn.returnToCache(old_e.w);
                            cn.returnToCache(m.w);
                        }
                    }
                }
            }
            e = makeDDNode(var, edge, true);

            computeTable.insert(xCopy, yCopy, {e.p, e.w});

            if (e.w != Complex::zero && (x.w != Complex::one || y.w != Complex::one)) {
                if (e.w == Complex::one) {
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
        /// Inner product and fidelity
        ///
    public:
        ComputeTable<vEdge, vEdge, vCachedEdge, 4096> vectorInnerProduct{};

        ComplexValue innerProduct(const vEdge& x, const vEdge& y) {
            if (x.p == nullptr || y.p == nullptr || x.w.approximatelyZero() || y.w.approximatelyZero()) { // the 0 case
                return {0, 0};
            }

            [[maybe_unused]] const auto before = cn.cacheCount();

            auto w = x.p->v;
            if (y.p->v > w) {
                w = y.p->v;
            }
            const ComplexValue ip = innerProduct(x, y, static_cast<Qubit>(w + 1));

            [[maybe_unused]] const auto after = cn.cacheCount();
            assert(after == before);

            return ip;
        }
        fp fidelity(const vEdge& x, const vEdge& y) {
            const auto fid = innerProduct(x, y);
            return fid.r * fid.r + fid.i * fid.i;
        }

    private:
        ComplexValue innerProduct(const vEdge& x, const vEdge& y, Qubit var) {
            if (x.p == nullptr || y.p == nullptr || x.w.approximatelyZero() || y.w.approximatelyZero()) { // the 0 case
                return {0.0, 0.0};
            }

            if (var == 0) {
                auto c = cn.getTemporary();
                ComplexNumbers::mul(c, x.w, y.w);
                return {c.r->value, c.i->value};
            }

            auto xCopy = x;
            xCopy.w    = Complex::one;
            auto yCopy = y;
            yCopy.w    = Complex::one;

            auto r = vectorInnerProduct.lookup(xCopy, yCopy);
            if (r.p != nullptr) {
                auto c = cn.getTemporary(r.w);
                ComplexNumbers::mul(c, c, x.w);
                ComplexNumbers::mul(c, c, y.w);
                return {CTEntry::val(c.r), CTEntry::val(c.i)};
            }

            auto w = static_cast<Qubit>(var - 1);

            ComplexValue sum{0.0, 0.0};
            for (auto i = 0U; i < RADIX; i++) {
                vEdge e1{};
                if (!x.isTerminal() && x.p->v == w) {
                    e1 = x.p->e[i];
                } else {
                    e1 = xCopy;
                }
                vEdge e2{};
                if (!y.isTerminal() && y.p->v == w) {
                    e2   = y.p->e[i];
                    e2.w = ComplexNumbers::conj(e2.w);
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

        ///
        /// Kronecker/tensor product
        ///
    public:
        ComputeTable<vEdge, vEdge, vCachedEdge, 4096> vectorKronecker{};
        ComputeTable<mEdge, mEdge, mCachedEdge, 4096> matrixKronecker{};

        template<class Node>
        [[nodiscard]] ComputeTable<Edge<Node>, Edge<Node>, CachedEdge<Node>, 4096>& getKroneckerComputeTable();

        template<class Edge>
        Edge kronecker(const Edge& x, const Edge& y, bool incIdx = true) {
            auto e = kronecker2(x, y, incIdx);

            if (e.w != Complex::zero && e.w != Complex::one) {
                cn.returnToCache(e.w);
                e.w = cn.lookup(e.w);
            }

            return e;
        }

        // extent the DD pointed to by `e` with `h` identities on top and `l` identities at the bottom
        mEdge extend(const mEdge& e, Qubit h, Qubit l = 0) {
            auto f = (l > 0) ? kronecker(e, makeIdent(l)) : e;
            auto g = (h > 0) ? kronecker(makeIdent(h), f) : f;
            return g;
        }

    private:
        template<class Node>
        Edge<Node> kronecker2(const Edge<Node>& x, const Edge<Node>& y, bool incIdx = true) {
            if (x.w.approximatelyZero() || y.w.approximatelyZero())
                return Edge<Node>::zero;

            if (x.isTerminal()) {
                auto r = y;
                r.w    = cn.mulCached(x.w, y.w);
                return r;
            }

            auto& computeTable = getKroneckerComputeTable<Node>();
            auto  r            = computeTable.lookup(x, y);
            if (r.p != nullptr) {
                if (r.w.approximatelyZero()) {
                    return Edge<Node>::zero;
                } else {
                    return {r.p, cn.getCached(r.w)};
                }
            }

            constexpr std::size_t N = std::tuple_size_v<decltype(x.p->e)>;
            // special case handling for matrices
            if constexpr (N == NEDGE) {
                if (x.p->ident) {
                    auto idx = incIdx ? static_cast<Qubit>(y.p->v + 1) : y.p->v;
                    auto e   = makeDDNode(idx, std::array{y, Edge<Node>::zero, Edge<Node>::zero, y});
                    for (auto i = 0; i < x.p->v; ++i) {
                        idx = incIdx ? static_cast<Qubit>(e.p->v + 1) : e.p->v;
                        e   = makeDDNode(idx, std::array{e, Edge<Node>::zero, Edge<Node>::zero, e});
                    }

                    e.w = cn.getCached(CTEntry::val(y.w.r), CTEntry::val(y.w.i));
                    computeTable.insert(x, y, {e.p, e.w});
                    return e;
                }
            }

            std::array<Edge<Node>, N> edge{};
            for (auto i = 0U; i < N; ++i) {
                edge[i] = kronecker2(x.p->e[i], y, incIdx);
            }

            auto idx = incIdx ? static_cast<Qubit>(y.p->v + x.p->v + 1) : x.p->v;
            auto e   = makeDDNode(idx, edge, true);
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
            const auto                  result = trace(a, eliminate);
            [[maybe_unused]] const auto after  = cn.cacheCount();
            assert(before == after);
            return result;
        }
        ComplexValue trace(const mEdge& a) {
            auto                        eliminate = std::vector<bool>(nqubits, true);
            [[maybe_unused]] const auto before    = cn.cacheCount();
            const auto                  res       = partialTrace(a, eliminate);
            [[maybe_unused]] const auto after     = cn.cacheCount();
            assert(before == after);
            return {CTEntry::val(res.w.r), CTEntry::val(res.w.i)};
        }

    private:
        /// TODO: introduce a compute table for the trace?
        mEdge trace(const mEdge& a, const std::vector<bool>& eliminate, std::size_t alreadyEliminated = 0) {
            auto v = a.p->v;

            if (a.w.approximatelyZero()) return mEdge::zero;

            if (std::none_of(eliminate.begin(), eliminate.end(), [](bool v) { return v; })) return a;

            // Base case
            if (v == -1) {
                if (a.isTerminal()) return a;
                throw std::runtime_error("Expected terminal node in trace.");
            }

            if (eliminate[v]) {
                auto elims = alreadyEliminated + 1;
                auto r     = mEdge::zero;

                auto t0 = trace(a.p->e[0], eliminate, elims);
                r       = add2(r, t0);
                auto r1 = r;

                auto t1 = trace(a.p->e[3], eliminate, elims);
                r       = add2(r, t1);
                auto r2 = r;

                if (r.w == Complex::one) {
                    r.w = a.w;
                } else {
                    auto c = cn.getTemporary();
                    ComplexNumbers::mul(c, r.w, a.w);
                    r.w = cn.lookup(c); // better safe than sorry. this may result in complex values with magnitude > 1 in the complex table
                }

                if (r1.w != Complex::zero) {
                    cn.returnToCache(r1.w);
                }

                if (r2.w != Complex::zero) {
                    cn.returnToCache(r2.w);
                }

                return r;
            } else {
                auto                     adjustedV = static_cast<Qubit>(a.p->v - (std::count(eliminate.begin(), eliminate.end(), true) - alreadyEliminated));
                std::array<mEdge, NEDGE> edge{};
                std::transform(a.p->e.cbegin(),
                               a.p->e.cend(),
                               edge.begin(),
                               [&](const mEdge& e) -> mEdge { return trace(e, eliminate, alreadyEliminated); });
                auto r = makeDDNode(adjustedV, edge);

                if (r.w == Complex::one) {
                    r.w = a.w;
                } else {
                    auto c = cn.getTemporary();
                    ComplexNumbers::mul(c, r.w, a.w);
                    r.w = cn.lookup(c);
                }
                return r;
            }
        }

        ///
        /// Toffoli gates
        ///
    public:
        ToffoliTable<mEdge> toffoliTable{};

        ///
        /// Identity matrices
        ///
    public:
        // create n-qubit identity DD. makeIdent(n) === makeIdent(0, n-1)
        mEdge makeIdent(QubitCount n) { return makeIdent(0, static_cast<Qubit>(n - 1)); }
        mEdge makeIdent(Qubit leastSignificantQubit, Qubit mostSignificantQubit) {
            if (mostSignificantQubit < leastSignificantQubit)
                return mEdge::one;

            if (leastSignificantQubit == 0 && IdTable[mostSignificantQubit].p != nullptr) {
                return IdTable[mostSignificantQubit];
            }
            if (mostSignificantQubit >= 1 && (IdTable[mostSignificantQubit - 1]).p != nullptr) {
                IdTable[mostSignificantQubit] = makeDDNode(mostSignificantQubit,
                                                           std::array{IdTable[mostSignificantQubit - 1],
                                                                      mEdge::zero,
                                                                      mEdge::zero,
                                                                      IdTable[mostSignificantQubit - 1]});
                return IdTable[mostSignificantQubit];
            }

            auto e = makeDDNode(leastSignificantQubit, std::array{mEdge::one, mEdge::zero, mEdge::zero, mEdge::one});
            for (std::size_t k = leastSignificantQubit + 1; k <= std::make_unsigned_t<Qubit>(mostSignificantQubit); k++) {
                e = makeDDNode(static_cast<Qubit>(k), std::array{e, mEdge::zero, mEdge::zero, e});
            }
            if (leastSignificantQubit == 0)
                IdTable[mostSignificantQubit] = e;
            return e;
        }

        // identity table access and reset
        [[nodiscard]] const auto& getIdentityTable() const { return IdTable; }

        void clearIdentityTable() {
            for (auto& entry: IdTable) entry.p = nullptr;
        }

    private:
        std::vector<mEdge> IdTable{};

        ///
        /// Noise Operations
        ///
    public:
        NoiseOperationTable<mEdge> noiseOperationTable{nqubits};

        ///
        /// Decision diagram size
        ///
    public:
        template<class Edge>
        unsigned int size(const Edge& e) {
            static constexpr unsigned int            NODECOUNT_BUCKETS = 200000;
            static std::unordered_set<decltype(e.p)> visited{NODECOUNT_BUCKETS}; // 2e6
            visited.max_load_factor(10);
            visited.clear();
            return nodeCount(e, visited);
        }

    private:
        template<class Edge>
        unsigned int nodeCount(const Edge& e, std::unordered_set<decltype(e.p)>& v) const {
            v.insert(e.p);
            unsigned int sum = 1;
            if (!e.isTerminal()) {
                for (const auto& edge: e.p->e) {
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
        mEdge reduceAncillae(mEdge& e, const std::vector<bool>& ancillary, bool regular = true) {
            // return if no more garbage left
            if (std::none_of(ancillary.begin(), ancillary.end(), [](bool v) { return v; }) || e.p == nullptr) return e;
            Qubit lowerbound = 0;
            for (auto i = 0U; i < ancillary.size(); ++i) {
                if (ancillary[i]) {
                    lowerbound = static_cast<Qubit>(i);
                    break;
                }
            }
            if (e.p->v < lowerbound) return e;
            auto f = reduceAncillaeRecursion(e, ancillary, lowerbound, regular);
            decRef(e);
            incRef(f);
            return f;
        }

        // Garbage reduction works for reversible circuits --- to be thoroughly tested for quantum circuits
        vEdge reduceGarbage(vEdge& e, const std::vector<bool>& garbage) {
            // return if no more garbage left
            if (std::none_of(garbage.begin(), garbage.end(), [](bool v) { return v; }) || e.p == nullptr) return e;
            Qubit lowerbound = 0;
            for (auto i = 0U; i < garbage.size(); ++i) {
                if (garbage[i]) {
                    lowerbound = static_cast<Qubit>(i);
                    break;
                }
            }
            if (e.p->v < lowerbound) return e;
            auto f = reduceGarbageRecursion(e, garbage, lowerbound);
            decRef(e);
            incRef(f);
            return f;
        }
        mEdge reduceGarbage(mEdge& e, const std::vector<bool>& garbage, bool regular = true) {
            // return if no more garbage left
            if (std::none_of(garbage.begin(), garbage.end(), [](bool v) { return v; }) || e.p == nullptr) return e;
            Qubit lowerbound = 0;
            for (auto i = 0U; i < garbage.size(); ++i) {
                if (garbage[i]) {
                    lowerbound = static_cast<Qubit>(i);
                    break;
                }
            }
            if (e.p->v < lowerbound) return e;
            auto f = reduceGarbageRecursion(e, garbage, lowerbound, regular);
            decRef(e);
            incRef(f);
            return f;
        }

    private:
        mEdge reduceAncillaeRecursion(mEdge& e, const std::vector<bool>& ancillary, Qubit lowerbound, bool regular = true) {
            if (e.p->v < lowerbound) return e;

            auto f = e;

            std::array<mEdge, NEDGE> edges{};
            std::bitset<NEDGE>       handled{};
            for (auto i = 0U; i < NEDGE; ++i) {
                if (!handled.test(i)) {
                    if (e.p->e[i].isTerminal()) {
                        edges[i] = e.p->e[i];
                    } else {
                        edges[i] = reduceAncillaeRecursion(f.p->e[i], ancillary, lowerbound, regular);
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
            if (f.p->v >= 0 && ancillary[f.p->v]) {
                if (regular) {
                    if (f.p->e[1].w != Complex::zero || f.p->e[3].w != Complex::zero) {
                        f = makeDDNode(f.p->v, std::array{f.p->e[0], mEdge::zero, f.p->e[2], mEdge::zero});
                    }
                } else {
                    if (f.p->e[2].w != Complex::zero || f.p->e[3].w != Complex::zero) {
                        f = makeDDNode(f.p->v, std::array{f.p->e[0], f.p->e[1], mEdge::zero, mEdge::zero});
                    }
                }
            }

            auto c = cn.mulCached(f.w, e.w);
            f.w    = cn.lookup(c);
            cn.returnToCache(c);
            return f;
        }

        vEdge reduceGarbageRecursion(vEdge& e, const std::vector<bool>& garbage, Qubit lowerbound) {
            if (e.p->v < lowerbound) return e;

            auto f = e;

            std::array<vEdge, RADIX> edges{};
            std::bitset<RADIX>       handled{};
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
            if (f.p->v >= 0 && garbage[f.p->v]) {
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
            f.w    = cn.lookup(c);
            cn.returnToCache(c);

            // Quick-fix for normalization bug
            if (ComplexNumbers::mag2(f.w) > 1.0)
                f.w = Complex::one;

            return f;
        }
        mEdge reduceGarbageRecursion(mEdge& e, const std::vector<bool>& garbage, Qubit lowerbound, bool regular = true) {
            if (e.p->v < lowerbound) return e;

            auto f = e;

            std::array<mEdge, NEDGE> edges{};
            std::bitset<NEDGE>       handled{};
            for (auto i = 0U; i < NEDGE; ++i) {
                if (!handled.test(i)) {
                    if (e.p->e[i].isTerminal()) {
                        edges[i] = e.p->e[i];
                    } else {
                        edges[i] = reduceGarbageRecursion(f.p->e[i], garbage, lowerbound, regular);
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
            if (f.p->v >= 0 && garbage[f.p->v]) {
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
            f.w    = cn.lookup(c);
            cn.returnToCache(c);

            // Quick-fix for normalization bug
            if (ComplexNumbers::mag2(f.w) > 1.0)
                f.w = Complex::one;

            return f;
        }

        ///
        /// Vector and matrix extraction from DDs
        ///
    public:
        /// Get a single element of the vector or matrix represented by the dd with root edge e
        /// \tparam Edge type of edge to use (vector or matrix)
        /// \param e edge to traverse
        /// \param elements string {0, 1, 2, 3}^n describing which outgoing edge should be followed
        ///        (for vectors entries are limitted to 0 and 1)
        ///        If string is longer than required, the additional characters are ignored.
        /// \return the complex amplitude of the specified element
        template<class Edge>
        ComplexValue getValueByPath(const Edge& e, const std::string& elements) {
            if (e.isTerminal()) {
                return {CTEntry::val(e.w.r), CTEntry::val(e.w.i)};
            }

            auto c = cn.getTemporary(1, 0);
            auto r = e;
            do {
                ComplexNumbers::mul(c, c, r.w);
                std::size_t tmp = elements.at(r.p->v) - '0';
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
        ComplexValue getValueByPath(const vEdge& e, const Complex& amp, std::size_t i) {
            auto c = cn.mulCached(e.w, amp);

            if (e.isTerminal()) {
                cn.returnToCache(c);
                return {CTEntry::val(c.r), CTEntry::val(c.i)};
            }

            bool one = i & (1 << e.p->v);

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
        ComplexValue getValueByPath(const mEdge& e, const Complex& amp, std::size_t i, std::size_t j) {
            auto c = cn.mulCached(e.w, amp);

            if (e.isTerminal()) {
                cn.returnToCache(c);
                return {CTEntry::val(c.r), CTEntry::val(c.i)};
            }

            bool row = i & (1 << e.p->v);
            bool col = j & (1 << e.p->v);

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

        CVec getVector(const vEdge& e) {
            std::size_t dim = 1 << (e.p->v + 1);
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

            std::size_t x = i | (1 << e.p->v);

            // recursive case
            if (!e.p->e[0].w.approximatelyZero())
                getVector(e.p->e[0], c, i, vec);
            if (!e.p->e[1].w.approximatelyZero())
                getVector(e.p->e[1], c, x, vec);
            cn.returnToCache(c);
        }
        void printVector(const vEdge& e) {
            unsigned long long element = 2u << e.p->v;
            for (unsigned long long i = 0; i < element; i++) {
                auto amplitude = getValueByPath(e, i);
                for (Qubit j = e.p->v; j >= 0; j--) {
                    std::cout << ((i >> j) & 1u);
                }
                constexpr auto precision = 3;
                // set fixed width to maximum of a printed number
                // (-) 0.precision plus/minus 0.precision i
                constexpr auto width = 1 + 2 + precision + 1 + 2 + precision + 1;
                std::cout << ": " << std::setw(width) << ComplexValue::toString(amplitude.r, amplitude.i, false, precision) << "\n";
            }
            std::cout << std::flush;
        }

        void printMatrix(const mEdge& e) {
            unsigned long long element = 2u << e.p->v;
            for (unsigned long long i = 0; i < element; i++) {
                for (unsigned long long j = 0; j < element; j++) {
                    auto           amplitude = getValueByPath(e, i, j);
                    constexpr auto precision = 3;
                    // set fixed width to maximum of a printed number
                    // (-) 0.precision plus/minus 0.precision i
                    constexpr auto width = 1 + 2 + precision + 1 + 2 + precision + 1;
                    std::cout << std::setw(width) << ComplexValue::toString(amplitude.r, amplitude.i, false, precision) << " ";
                }
                std::cout << "\n";
            }
            std::cout << std::flush;
        }

        CMat getMatrix(const mEdge& e) {
            std::size_t dim = 1 << (e.p->v + 1);
            // allocate resulting matrix
            auto mat = CMat(dim, CVec(dim, {0.0, 0.0}));
            getMatrix(e, Complex::one, 0, 0, mat);
            return mat;
        }
        void getMatrix(const mEdge& e, const Complex& amp, std::size_t i, std::size_t j, CMat& mat) {
            // calculate new accumulated amplitude
            auto c = cn.mulCached(e.w, amp);

            // base case
            if (e.isTerminal()) {
                mat.at(i).at(j) = {CTEntry::val(c.r), CTEntry::val(c.i)};
                cn.returnToCache(c);
                return;
            }

            std::size_t x = i | (1 << e.p->v);
            std::size_t y = j | (1 << e.p->v);

            // recursive case
            if (!e.p->e[0].w.approximatelyZero())
                getMatrix(e.p->e[0], c, i, j, mat);
            if (!e.p->e[1].w.approximatelyZero())
                getMatrix(e.p->e[1], c, i, y, mat);
            if (!e.p->e[2].w.approximatelyZero())
                getMatrix(e.p->e[2], c, x, j, mat);
            if (!e.p->e[3].w.approximatelyZero())
                getMatrix(e.p->e[3], c, x, y, mat);
            cn.returnToCache(c);
        }

        void exportAmplitudesRec(const dd::Package::vEdge& edge, std::ostream& oss, const std::string& path, Complex& amplitude, dd::QubitCount level, bool binary = false) {
            if (edge.isTerminal()) {
                auto amp = cn.getTemporary();
                dd::ComplexNumbers::mul(amp, amplitude, edge.w);
                for (std::size_t i = 0; i < (1UL << level); i++) {
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
        void exportAmplitudes(const dd::Package::vEdge& edge, std::ostream& oss, dd::QubitCount nq, bool binary = false) {
            if (edge.isTerminal()) {
                // TODO special treatment
                return;
            }
            auto weight = cn.getCached(1., 0.);
            exportAmplitudesRec(edge, oss, "", weight, nq, binary);
            cn.returnToCache(weight);
        }
        void exportAmplitudes(const dd::Package::vEdge& edge, const std::string& outputFilename, dd::QubitCount nq, bool binary = false) {
            std::ofstream      init(outputFilename);
            std::ostringstream oss{};

            exportAmplitudes(edge, oss, nq, binary);

            init << oss.str() << std::flush;
            init.close();
        }

        void exportAmplitudesRec(const dd::Package::vEdge& edge, std::vector<ComplexValue>& amplitudes, Complex& amplitude, dd::QubitCount level, std::size_t idx) {
            if (edge.isTerminal()) {
                auto amp = cn.getTemporary();
                dd::ComplexNumbers::mul(amp, amplitude, edge.w);
                idx <<= level;
                for (std::size_t i = 0; i < (1UL << level); i++) {
                    amplitudes[idx++] = dd::ComplexValue{dd::ComplexTable<>::Entry::val(amp.r), dd::ComplexTable<>::Entry::val(amp.i)};
                }

                return;
            }

            auto a = cn.mulCached(amplitude, edge.w);
            exportAmplitudesRec(edge.p->e[0], amplitudes, a, level - 1, idx << 1);
            exportAmplitudesRec(edge.p->e[1], amplitudes, a, level - 1, (idx << 1) | 1);
            cn.returnToCache(a);
        }
        void exportAmplitudes(const dd::Package::vEdge& edge, std::vector<ComplexValue>& amplitudes, dd::QubitCount nq) {
            if (edge.isTerminal()) {
                // TODO special treatment
                return;
            }
            auto weight = cn.getCached(1., 0.);
            exportAmplitudesRec(edge, amplitudes, weight, nq, 0);
            cn.returnToCache(weight);
        }

        void addAmplitudesRec(const dd::Package::vEdge& edge, std::vector<ComplexValue>& amplitudes, ComplexValue& amplitude, dd::QubitCount level, std::size_t idx) {
            auto         ar = dd::ComplexTable<>::Entry::val(edge.w.r);
            auto         ai = dd::ComplexTable<>::Entry::val(edge.w.i);
            ComplexValue amp{ar * amplitude.r - ai * amplitude.i, ar * amplitude.i + ai * amplitude.r};

            if (edge.isTerminal()) {
                idx <<= level;
                for (std::size_t i = 0; i < (1UL << level); i++) {
                    auto temp         = dd::ComplexValue{amp.r + amplitudes[idx].r, amp.i + amplitudes[idx].i};
                    amplitudes[idx++] = temp;
                }

                return;
            }

            addAmplitudesRec(edge.p->e[0], amplitudes, amp, level - 1, idx << 1);
            addAmplitudesRec(edge.p->e[1], amplitudes, amp, level - 1, idx << 1 | 1);
        }
        void addAmplitudes(const dd::Package::vEdge& edge, std::vector<ComplexValue>& amplitudes, dd::QubitCount nq) {
            if (edge.isTerminal()) {
                // TODO special treatment
                return;
            }
            ComplexValue a{1., 0.};
            addAmplitudesRec(edge, amplitudes, a, nq, 0);
        }

        // transfers a decision diagram from another package to this package
        template<class Edge>
        Edge transfer(Edge& original) {
            // POST ORDER TRAVERSAL USING ONE STACK   https://www.geeksforgeeks.org/iterative-postorder-traversal-using-stack/
            Edge              root{};
            std::stack<Edge*> stack;

            std::unordered_map<decltype(original.p), decltype(original.p)> mapped_node{};

            Edge* currentEdge = &original;
            if (!currentEdge->isTerminal()) {
                constexpr std::size_t N = std::tuple_size_v<decltype(original.p->e)>;
                do {
                    while (currentEdge != nullptr && !currentEdge->isTerminal()) {
                        for (short i = N - 1; i > 0; --i) {
                            auto& edge = currentEdge->p->e[i];
                            if (edge.isTerminal()) {
                                continue;
                            }
                            if (edge.w.approximatelyZero()) {
                                continue;
                            }
                            if (mapped_node.find(edge.p) != mapped_node.end()) {
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
                    for (std::size_t i = 1; i < N && !hasChild; ++i) {
                        auto& edge = currentEdge->p->e[i];
                        if (edge.w.approximatelyZero()) {
                            continue;
                        }
                        if (mapped_node.find(edge.p) != mapped_node.end()) {
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
                        if (mapped_node.find(currentEdge->p) != mapped_node.end()) {
                            currentEdge = nullptr;
                            continue;
                        }
                        std::array<Edge, N> edges{};
                        for (std::size_t i = 0; i < N; i++) {
                            if (currentEdge->p->e[i].isTerminal()) {
                                edges[i].p = currentEdge->p->e[i].p;
                            } else {
                                edges[i].p = mapped_node[currentEdge->p->e[i].p];
                            }
                            edges[i].w = cn.lookup(currentEdge->p->e[i].w);
                        }
                        root                        = makeDDNode(currentEdge->p->v, edges);
                        mapped_node[currentEdge->p] = root.p;
                        currentEdge                 = nullptr;
                    }
                } while (!stack.empty());

                auto w = cn.getCached(dd::ComplexTable<>::Entry::val(original.w.r), dd::ComplexTable<>::Entry::val(original.w.i));
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
        /// Note: do not rely on the binary format being portable across different architectures/platforms
        ///
    public:
        template<class Node, class Edge = Edge<Node>, std::size_t N = std::tuple_size_v<decltype(Node::e)>>
        Edge deserialize(std::istream& is, bool readBinary = false) {
            auto         result = Edge::zero;
            ComplexValue rootweight{};

            std::unordered_map<std::int_least64_t, Node*> nodes{};
            std::int_least64_t                            node_index;
            Qubit                                         v;
            std::array<ComplexValue, N>                   edge_weights{};
            std::array<std::int_least64_t, N>             edge_indices{};
            edge_indices.fill(-2);

            if (readBinary) {
                std::remove_const_t<decltype(SERIALIZATION_VERSION)> version;
                is.read(reinterpret_cast<char*>(&version), sizeof(decltype(SERIALIZATION_VERSION)));
                if (version != SERIALIZATION_VERSION) {
                    throw std::runtime_error("Wrong Version of serialization file version. version of file: " + std::to_string(version) + "; current version: " + std::to_string(SERIALIZATION_VERSION));
                }

                if (!is.eof()) {
                    rootweight.readBinary(is);
                }

                while (is.read(reinterpret_cast<char*>(&node_index), sizeof(decltype(node_index)))) {
                    is.read(reinterpret_cast<char*>(&v), sizeof(decltype(v)));
                    for (auto i = 0U; i < N; i++) {
                        is.read(reinterpret_cast<char*>(&edge_indices[i]), sizeof(decltype(edge_indices[i])));
                        edge_weights[i].readBinary(is);
                    }
                    result = deserializeNode(node_index, v, edge_indices, edge_weights, nodes);
                }
            } else {
                std::string version;
                std::getline(is, version);
                if (std::stoi(version) != SERIALIZATION_VERSION) {
                    throw std::runtime_error("Wrong Version of serialization file version. version of file: " + version + "; current version: " + std::to_string(SERIALIZATION_VERSION));
                }

                std::string line;
                std::string complex_real_regex = R"(([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?(?![ \d\.]*(?:[eE][+-])?\d*[iI]))?)";
                std::string complex_imag_regex = R"(( ?[+-]? ?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)?[iI])?)";
                std::string edge_regex         = " \\(((-?\\d+) (" + complex_real_regex + complex_imag_regex + "))?\\)";
                std::regex  complex_weight_regex(complex_real_regex + complex_imag_regex);
                std::string line_construct = "(\\d+) (\\d+)";
                for (auto i = 0U; i < N; ++i) {
                    line_construct += "(?:" + edge_regex + ")";
                }
                line_construct += " *(?:#.*)?";
                std::regex  line_regex(line_construct);
                std::smatch m;

                if (std::getline(is, line)) {
                    if (!std::regex_match(line, m, complex_weight_regex)) {
                        throw std::runtime_error("Regex did not match second line: " + line);
                    }
                    rootweight.from_string(m.str(1), m.str(2));
                }

                while (std::getline(is, line)) {
                    if (line.empty() || line.size() == 1) continue;

                    if (!std::regex_match(line, m, line_regex)) {
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
                    node_index = std::stoi(m.str(1));
                    v          = static_cast<Qubit>(std::stoi(m.str(2)));

                    for (auto edge_idx = 3U, i = 0U; i < N; i++, edge_idx += 5) {
                        if (m.str(edge_idx).empty()) continue;

                        edge_indices[i] = std::stoi(m.str(edge_idx + 1));
                        edge_weights[i].from_string(m.str(edge_idx + 3), m.str(edge_idx + 4));
                    }

                    result = deserializeNode(node_index, v, edge_indices, edge_weights, nodes);
                }
            }

            auto w = cn.getCached(rootweight.r, rootweight.i);
            ComplexNumbers::mul(w, result.w, w);
            result.w = cn.lookup(w);
            cn.returnToCache(w);

            return result;
        }

        template<class Node, class Edge = Edge<Node>>
        Edge deserialize(const std::string& inputFilename, bool readBinary) {
            auto ifs = std::ifstream(inputFilename, std::ios::binary);

            if (!ifs.good()) {
                throw std::invalid_argument("Cannot open serialized file: " + inputFilename);
            }

            return deserialize<Node>(ifs, readBinary);
        }

    private:
        template<class Node, class Edge = Edge<Node>, std::size_t N = std::tuple_size_v<decltype(Node::e)>>
        Edge deserializeNode(std::int_least64_t index, Qubit v, std::array<std::int_least64_t, N>& edge_idx, std::array<ComplexValue, N>& edge_weight, std::unordered_map<std::int_least64_t, Node*>& nodes) {
            if (index == -1) {
                return Edge::zero;
            }

            std::array<Edge, N> edges{};
            for (auto i = 0U; i < N; ++i) {
                if (edge_idx[i] == -2) {
                    edges[i] = Edge::zero;
                } else {
                    if (edge_idx[i] == -1) {
                        edges[i] = Edge::one;
                    } else {
                        edges[i].p = nodes[edge_idx[i]];
                    }
                    edges[i].w = cn.lookup(edge_weight[i]);
                }
            }

            auto newedge = makeDDNode(v, edges);
            nodes[index] = newedge.p;

            // reset
            edge_idx.fill(-2);

            return newedge;
        }

        ///
        /// Debugging
        ///
    public:
        template<class Node>
        void debugnode(const Node* p) const {
            if (Node::isTerminal(p)) {
                std::clog << "terminal\n";
                return;
            }
            std::clog << "Debug node: " << debugnode_line(p) << "\n";
            for (const auto& edge: p->e) {
                std::clog << "  " << std::hexfloat
                          << std::setw(22) << CTEntry::val(edge.w.r) << " "
                          << std::setw(22) << CTEntry::val(edge.w.i) << std::defaultfloat
                          << "i --> " << debugnode_line(edge.p) << "\n";
            }
            std::clog << std::flush;
        }

        template<class Node>
        std::string debugnode_line(const Node* p) const {
            if (Node::isTerminal(p)) {
                return "terminal";
            }
            std::stringstream sst;
            sst << "0x" << std::hex << reinterpret_cast<std::uintptr_t>(p) << std::dec
                << "[v=" << static_cast<std::int_fast64_t>(p->v)
                << " ref=" << p->ref
                << " hash=" << UniqueTable<Node>::hash(p)
                << "]";
            return sst.str();
        }

        template<class Edge>
        bool isLocallyConsistent(const Edge& e) {
            assert(Complex::one.r->value == 1 && Complex::one.i->value == 0);
            assert(Complex::zero.r->value == 0 && Complex::zero.i->value == 0);

            const bool result = isLocallyConsistent2(e);
            return result;
        }

        template<class Edge>
        bool isGloballyConsistent(const Edge& e) {
            std::map<ComplexTable<>::Entry*, std::size_t> weight_counter{};
            std::map<decltype(e.p), std::size_t>          node_counter{};
            fillConsistencyCounter(e, weight_counter, node_counter);
            checkConsistencyCounter(e, weight_counter, node_counter);
            return true;
        }

    private:
        template<class Edge>
        bool isLocallyConsistent2(const Edge& e) {
            const auto ptr_r = CTEntry::getAlignedPointer(e.w.r);
            const auto ptr_i = CTEntry::getAlignedPointer(e.w.i);

            if ((ptr_r->refCount == 0 || ptr_i->refCount == 0) && e.w != Complex::one && e.w != Complex::zero) {
                std::clog << "\nLOCAL INCONSISTENCY FOUND\nOffending Number: " << e.w << " (" << ptr_r->refCount << ", " << ptr_i->refCount << ")\n\n";
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

            for (const auto& child: e.p->e) {
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

        template<class Edge>
        void fillConsistencyCounter(const Edge& edge, std::map<ComplexTable<>::Entry*, std::size_t>& weight_map, std::map<decltype(edge.p), std::size_t>& node_map) {
            weight_map[CTEntry::getAlignedPointer(edge.w.r)]++;
            weight_map[CTEntry::getAlignedPointer(edge.w.i)]++;

            if (edge.isTerminal()) {
                return;
            }
            node_map[edge.p]++;
            for (auto& child: edge.p->e) {
                if (node_map[child.p] == 0) {
                    fillConsistencyCounter(child, weight_map, node_map);
                } else {
                    node_map[child.p]++;
                    weight_map[CTEntry::getAlignedPointer(child.w.r)]++;
                    weight_map[CTEntry::getAlignedPointer(child.w.i)]++;
                }
            }
        }

        template<class Edge>
        void checkConsistencyCounter(const Edge& edge, const std::map<ComplexTable<>::Entry*, std::size_t>& weight_map, const std::map<decltype(edge.p), std::size_t>& node_map) {
            auto* r_ptr = CTEntry::getAlignedPointer(edge.w.r);
            auto* i_ptr = CTEntry::getAlignedPointer(edge.w.i);

            if (weight_map.at(r_ptr) > r_ptr->refCount && r_ptr != Complex::one.r && r_ptr != Complex::zero.i && r_ptr != &ComplexTable<>::sqrt2_2) {
                std::clog << "\nOffending weight: " << edge.w << "\n";
                std::clog << "Bits: " << std::hexfloat << CTEntry::val(edge.w.r) << "r " << CTEntry::val(edge.w.i) << std::defaultfloat << "i\n";
                debugnode(edge.p);
                throw std::runtime_error("Ref-Count mismatch for " + std::to_string(r_ptr->value) + "(r): " + std::to_string(weight_map.at(r_ptr)) + " occurences in DD but Ref-Count is only " + std::to_string(r_ptr->refCount));
            }

            if (weight_map.at(i_ptr) > i_ptr->refCount && i_ptr != Complex::zero.i && i_ptr != Complex::one.r && i_ptr != &ComplexTable<>::sqrt2_2) {
                std::clog << "\nOffending weight: " << edge.w << "\n";
                std::clog << "Bits: " << std::hexfloat << CTEntry::val(edge.w.r) << "r " << CTEntry::val(edge.w.i) << std::defaultfloat << "i\n";
                debugnode(edge.p);
                throw std::runtime_error("Ref-Count mismatch for " + std::to_string(i_ptr->value) + "(i): " + std::to_string(weight_map.at(i_ptr)) + " occurences in DD but Ref-Count is only " + std::to_string(i_ptr->refCount));
            }

            if (edge.isTerminal()) {
                return;
            }

            if (node_map.at(edge.p) != edge.p->ref) {
                debugnode(edge.p);
                throw std::runtime_error("Ref-Count mismatch for node: " + std::to_string(node_map.at(edge.p)) + " occurences in DD but Ref-Count is " + std::to_string(edge.p->ref));
            }
            for (auto child: edge.p->e) {
                if (!child.isTerminal() && child.p->v != edge.p->v - 1) {
                    std::clog << "child.p->v == " << child.p->v << "\n";
                    std::clog << " edge.p->v == " << edge.p->v << "\n";
                    debugnode(child.p);
                    debugnode(edge.p);
                    throw std::runtime_error("Variable level ordering seems wrong");
                }
                checkConsistencyCounter(child, weight_map, node_map);
            }
        }

        ///
        /// Printing and Statistics
        ///
    public:
        // print information on package and its members
        static void printInformation() {
            std::cout << "\n  compiled: " << __DATE__ << " " << __TIME__
                      << "\n  Complex size: " << sizeof(Complex) << " bytes (aligned " << alignof(Complex) << " bytes)"
                      << "\n  ComplexValue size: " << sizeof(ComplexValue) << " bytes (aligned " << alignof(ComplexValue) << " bytes)"
                      << "\n  ComplexNumbers size: " << sizeof(ComplexNumbers) << " bytes (aligned " << alignof(ComplexNumbers) << " bytes)"
                      << "\n  vEdge size: " << sizeof(vEdge) << " bytes (aligned " << alignof(vEdge) << " bytes)"
                      << "\n  vNode size: " << sizeof(vNode) << " bytes (aligned " << alignof(vNode) << " bytes)"
                      << "\n  mEdge size: " << sizeof(mEdge) << " bytes (aligned " << alignof(mEdge) << " bytes)"
                      << "\n  mNode size: " << sizeof(mNode) << " bytes (aligned " << alignof(mNode) << " bytes)"
                      << "\n  CT Vector Add size: " << sizeof(decltype(vectorAdd)::Entry) << " bytes (aligned " << alignof(decltype(vectorAdd)::Entry) << " bytes)"
                      << "\n  CT Matrix Add size: " << sizeof(decltype(matrixAdd)::Entry) << " bytes (aligned " << alignof(decltype(matrixAdd)::Entry) << " bytes)"
                      << "\n  CT Matrix Transpose size: " << sizeof(decltype(matrixTranspose)::Entry) << " bytes (aligned " << alignof(decltype(matrixTranspose)::Entry) << " bytes)"
                      << "\n  CT Conjugate Matrix Transpose size: " << sizeof(decltype(conjugateMatrixTranspose)::Entry) << " bytes (aligned " << alignof(decltype(conjugateMatrixTranspose)::Entry) << " bytes)"
                      << "\n  CT Matrix Multiplication size: " << sizeof(decltype(matrixMatrixMultiplication)::Entry) << " bytes (aligned " << alignof(decltype(matrixMatrixMultiplication)::Entry) << " bytes)"
                      << "\n  CT Matrix Vector Multiplication size: " << sizeof(decltype(matrixVectorMultiplication)::Entry) << " bytes (aligned " << alignof(decltype(matrixVectorMultiplication)::Entry) << " bytes)"
                      << "\n  CT Vector Inner Product size: " << sizeof(decltype(vectorInnerProduct)::Entry) << " bytes (aligned " << alignof(decltype(vectorInnerProduct)::Entry) << " bytes)"
                      << "\n  CT Vector Kronecker size: " << sizeof(decltype(vectorKronecker)::Entry) << " bytes (aligned " << alignof(decltype(vectorKronecker)::Entry) << " bytes)"
                      << "\n  CT Matrix Kronecker size: " << sizeof(decltype(matrixKronecker)::Entry) << " bytes (aligned " << alignof(decltype(matrixKronecker)::Entry) << " bytes)"
                      << "\n  ToffoliTable::Entry size: " << sizeof(ToffoliTable<mEdge>::Entry) << " bytes (aligned " << alignof(ToffoliTable<mEdge>::Entry) << " bytes)"
                      << "\n  Package size: " << sizeof(Package) << " bytes (aligned " << alignof(Package) << " bytes)"
                      << "\n"
                      << std::flush;
        }

        // print unique and compute table statistics
        void statistics() {
            std::cout << "DD statistics:" << std::endl
                      << "[vUniqueTable] ";
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
            std::cout << "[Operation Table] ";
            noiseOperationTable.printStatistics();
            std::cout << "[ComplexTable] ";
            cn.complexTable.printStatistics();
        }
    };

    inline Package::vNode Package::vNode::terminalNode{{{{nullptr, Complex::zero}, {nullptr, Complex::zero}}},
                                                       nullptr,
                                                       0,
                                                       -1};

    inline Package::mNode Package::mNode::terminalNode{
            {{{nullptr, Complex::zero}, {nullptr, Complex::zero}, {nullptr, Complex::zero}, {nullptr, Complex::zero}}},
            nullptr,
            0,
            -1,
            true,
            true};

    template<>
    [[nodiscard]] inline UniqueTable<Package::vNode>& Package::getUniqueTable() { return vUniqueTable; }

    template<>
    [[nodiscard]] inline UniqueTable<Package::mNode>& Package::getUniqueTable() { return mUniqueTable; }

    template<>
    [[nodiscard]] inline ComputeTable<Package::vCachedEdge, Package::vCachedEdge, Package::vCachedEdge>& Package::getAddComputeTable() { return vectorAdd; }

    template<>
    [[nodiscard]] inline ComputeTable<Package::mCachedEdge, Package::mCachedEdge, Package::mCachedEdge>& Package::getAddComputeTable() { return matrixAdd; }

    template<>
    [[nodiscard]] inline ComputeTable<Package::mEdge, Package::vEdge, Package::vCachedEdge>& Package::getMultiplicationComputeTable() { return matrixVectorMultiplication; }

    template<>
    [[nodiscard]] inline ComputeTable<Package::mEdge, Package::mEdge, Package::mCachedEdge>& Package::getMultiplicationComputeTable() { return matrixMatrixMultiplication; }

    template<>
    [[nodiscard]] inline ComputeTable<Package::vEdge, Package::vEdge, Package::vCachedEdge, 4096>& Package::getKroneckerComputeTable() { return vectorKronecker; }

    template<>
    [[nodiscard]] inline ComputeTable<Package::mEdge, Package::mEdge, Package::mCachedEdge, 4096>& Package::getKroneckerComputeTable() { return matrixKronecker; }
} // namespace dd
#endif
