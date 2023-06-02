/*
 * This file is part of the MQT DD Package which is released under the MIT license.
 * See file README.md or go to https://www.cda.cit.tum.de/research/quantum_dd/ for more information.
 */

#ifndef DD_PACKAGE_EDGE_HPP
#define DD_PACKAGE_EDGE_HPP

#include "Complex.hpp"
#include "ComplexValue.hpp"
#include "Definitions.hpp"

#include <array>
#include <cstddef>
#include <utility>

namespace dd {
    template<class Node>
    struct Edge {
        Node*   p;
        Complex w;

        /// Comparing two DD edges with another involves comparing the respective pointers
        /// and checking whether the corresponding weights are "close enough" according to a given tolerance
        /// this notion of equivalence is chosen to counter floating point inaccuracies
        constexpr bool operator==(const Edge& other) const {
            return p == other.p && w.approximatelyEquals(other.w);
        }
        constexpr bool operator!=(const Edge& other) const {
            return !operator==(other);
        }

        [[nodiscard]] constexpr bool isTerminal() const { return Node::isTerminal(p); }

        // edges pointing to zero and one terminals
        static const inline Edge one{Node::terminal, Complex::one};   // NOLINT(readability-identifier-naming) automatic renaming does not work reliably, so skip linting
        static const inline Edge zero{Node::terminal, Complex::zero}; // NOLINT(readability-identifier-naming) automatic renaming does not work reliably, so skip linting

        [[nodiscard]] static constexpr Edge terminal(const Complex& w) { return {Node::terminal, w}; }
        [[nodiscard]] constexpr bool        isZeroTerminal() const { return Node::isTerminal(p) && w == Complex::zero; }
        [[nodiscard]] constexpr bool        isOneTerminal() const { return Node::isTerminal(p) && w == Complex::one; }

        [[maybe_unused]] static inline void setDensityConjugateTrue(Edge& e) { Node::setConjugateTempFlagTrue(e.p); }
        [[maybe_unused]] static inline void setFirstEdgeDensityPathTrue(Edge& e) { Node::setNonReduceTempFlagTrue(e.p); }
        [[maybe_unused]] static inline void setDensityMatrixTrue(Edge& e) { Node::setDensityMatTempFlagTrue(e.p); }
        [[maybe_unused]] static inline void alignDensityEdge(Edge& e) { Node::alignDensityNode(e.p); }

        static inline void revertDmChangesToEdges(Edge& x, Edge& y) {
            revertDmChangesToEdge(x);
            revertDmChangesToEdge(y);
        }
        static inline void revertDmChangesToEdge(Edge& x) {
            // Align the node pointer
            Node::revertDmChangesToNode(x.p);
        }

        static inline void applyDmChangesToEdges(Edge& x, Edge& y) {
            applyDmChangesToEdge(x);
            applyDmChangesToEdge(y);
        }

        static inline void applyDmChangesToEdge(Edge& x) {
            // Apply density matrix changes to node pointer
            Node::applyDmChangesToNode(x.p);
        }
    };

    template<typename Node>
    struct CachedEdge {
        Node*        p{};
        ComplexValue w{};

        CachedEdge() = default;
        CachedEdge(Node* n, const ComplexValue& v):
            p(n), w(v) {}
        CachedEdge(Node* n, const Complex& c):
            p(n) {
            w.r = CTEntry::val(c.r);
            w.i = CTEntry::val(c.i);
        }

        /// Comparing two DD edges with another involves comparing the respective pointers
        /// and checking whether the corresponding weights are "close enough" according to a given tolerance
        /// this notion of equivalence is chosen to counter floating point inaccuracies
        bool operator==(const CachedEdge& other) const {
            return p == other.p && w.approximatelyEquals(other.w);
        }
        bool operator!=(const CachedEdge& other) const {
            return !operator==(other);
        }
    };
} // namespace dd

namespace std {
    template<class Node>
    struct hash<dd::Edge<Node>> {
        std::size_t operator()(dd::Edge<Node> const& e) const noexcept {
            auto h1 = dd::murmur64(reinterpret_cast<std::size_t>(e.p));
            auto h2 = std::hash<dd::Complex>{}(e.w);
            return dd::combineHash(h1, h2);
        }
    };

    template<class Node>
    struct hash<dd::CachedEdge<Node>> {
        std::size_t operator()(dd::CachedEdge<Node> const& e) const noexcept {
            auto h1 = dd::murmur64(reinterpret_cast<std::size_t>(e.p));
            auto h2 = std::hash<dd::ComplexValue>{}(e.w);
            return dd::combineHash(h1, h2);
        }
    };
} // namespace std

#endif //DD_PACKAGE_EDGE_HPP
