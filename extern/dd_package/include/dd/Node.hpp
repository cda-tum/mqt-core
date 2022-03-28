/*
* This file is part of the MQT DD Package which is released under the MIT license.
* See file README.md or go to https://www.cda.cit.tum.de/research/quantum_dd/ for more information.
*/

#pragma once

#include "Definitions.hpp"
#include "Edge.hpp"

#include <array>

namespace dd {
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

    inline vNode vNode::terminalNode{{{{nullptr, Complex::zero}, {nullptr, Complex::zero}}}, nullptr, 0U, -1};

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

    inline mNode mNode::terminalNode{{{{nullptr, Complex::zero}, {nullptr, Complex::zero}, {nullptr, Complex::zero}, {nullptr, Complex::zero}}}, nullptr, 0U, -1, true, true};
} // namespace dd
