/*
 * This file is part of the JKQ DD Package which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum_dd/ for more information.
 */

#ifndef DDexport_H
#define DDexport_H

#include "Complex.hpp"
#include "ComplexNumbers.hpp"
#include "Definitions.hpp"
#include "Edge.hpp"
#include "Package.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <queue>
#include <stack>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace dd {

    inline std::string colorFromPhase(const Complex& a) {
        auto phase = dd::ComplexNumbers::arg(a);
        auto twopi = 2 * dd::PI;
        phase      = (phase) / twopi;
        if (phase < 0)
            phase += 1.;
        std::ostringstream oss{};
        oss << std::fixed << std::setprecision(3) << phase << " " << 0.667 << " " << 0.75;
        return oss.str();
    }
    inline fp thicknessFromMagnitude(const Complex& a) {
        return 3.0 * std::max(dd::ComplexNumbers::mag(a), 0.10);
    }

    template<class Edge>
    static std::ostream& header(const Edge& e, std::ostream& os, bool edgeLabels) {
        os << "digraph \"DD\" {graph[];node[shape=plain];edge[arrowhead=none]\n";
        os << "root [label=\"\",shape=point,style=invis]\n";
        os << "t [label=<<font point-size=\"20\">1</font>>,shape=box,tooltip=\"1\",width=0.3,height=0.3]\n";
        auto toplabel = (reinterpret_cast<std::uintptr_t>(e.p) & 0x001fffffU) >> 1U;
        auto mag      = thicknessFromMagnitude(e.w);
        os << "root->";
        if (e.isTerminal()) {
            os << "t";
        } else {
            os << toplabel;
        }
        os << "[penwidth=\"" << mag << "\",tooltip=\"" << e.w << "\"";
        if (!e.w.approximatelyOne()) {
            os << ",style=dashed";
        }
        if (edgeLabels) {
            os << ",label=<<font point-size=\"8\">&nbsp;" << e.w << "</font>>";
        }

        os << "]\n";

        return os;
    }
    template<class Edge>
    static std::ostream& coloredHeader(const Edge& e, std::ostream& os, bool edgeLabels) {
        os << "digraph \"DD\" {graph[];node[shape=plain];edge[arrowhead=none]\n";
        os << "root [label=\"\",shape=point,style=invis]\n";
        os << "t [label=<<font point-size=\"20\">1</font>>,shape=box,tooltip=\"1\",width=0.3,height=0.3]\n";

        auto toplabel = (reinterpret_cast<std::uintptr_t>(e.p) & 0x001fffffU) >> 1U;
        auto mag      = thicknessFromMagnitude(e.w);
        auto color    = colorFromPhase(e.w);
        os << "root->";
        if (e.isTerminal()) {
            os << "t";
        } else {
            os << toplabel;
        }
        os << "[penwidth=\"" << mag << "\",tooltip=\"" << e.w << "\",color=\"" << color << "\"";
        if (edgeLabels) {
            os << ",label=<<font point-size=\"8\">&nbsp;" << e.w << "</font>>";
        }
        os << "]\n";
        return os;
    }
    template<class Edge>
    static std::ostream& memoryHeader(const Edge& e, std::ostream& os, bool edgeLabels) {
        os << "digraph \"DD\" {graph[];node[shape=plain];edge[arrowhead=none]\n";
        os << "root [label=\"\",shape=point,style=invis]\n";
        os << "t [label=<<font point-size=\"20\">1</font>>,shape=box,tooltip=\"1\",width=0.3,height=0.3]\n";

        auto toplabel = (reinterpret_cast<std::uintptr_t>(e.p) & 0x001fffffU) >> 1U;
        auto mag      = thicknessFromMagnitude(e.w);
        auto color    = colorFromPhase(e.w);
        os << "root->";
        if (e.isTerminal()) {
            os << "t";
        } else {
            os << toplabel;
        }
        os << "[penwidth=\"" << mag << "\",tooltip=\"" << e.w.toString(false, 4) << "\" color=\"" << color << "\"";
        if (edgeLabels) {
            os << ",label=<<font point-size=\"8\">&nbsp;[";
            if (e.w == Complex::zero) {
                os << "0";
            } else if (e.w == Complex::one) {
                os << "1";
            } else {
                if (e.w.r == &ComplexTable<>::zero) {
                    os << "0";
                } else if (e.w.r == &ComplexTable<>::sqrt2_2) {
                    os << u8"\u221a\u00bd";
                } else if (e.w.r == &ComplexTable<>::one) {
                    os << "1";
                } else {
                    os << std::hex << reinterpret_cast<std::uintptr_t>(e.w.r) << std::dec;
                }
                os << " ";
                if (e.w.i == &ComplexTable<>::zero) {
                    os << "0";
                } else if (e.w.i == &ComplexTable<>::sqrt2_2) {
                    os << u8"\u221a\u00bd";
                } else if (e.w.i == &ComplexTable<>::one) {
                    os << "1";
                } else {
                    os << std::hex << reinterpret_cast<std::uintptr_t>(e.w.i) << std::dec;
                }
            }
            os << "]</font>>";
        }
        os << "]\n";
        return os;
    }

    [[maybe_unused]] static std::ostream& modernNode(const Package::mEdge& e, std::ostream& os) {
        auto nodelabel = (reinterpret_cast<std::uintptr_t>(e.p) & 0x001fffffU) >> 1U; // this allows for 2^20 (roughly 1e6) unique nodes
        os << nodelabel << "[label=<";
        os << R"(<font point-size="10"><table border="1" cellspacing="0" cellpadding="2" style="rounded">)";
        os << R"(<tr><td colspan="2" rowspan="2" port="0" href="javascript:;" border="0" tooltip=")" << e.p->e[0].w << "\">" << (e.p->e[0].w.approximatelyZero() ? "&nbsp;0 " : "<font color=\"white\">&nbsp;0 </font>")
           << R"(</td><td sides="R"></td><td sides="L"></td>)"
           << R"(<td colspan="2" rowspan="2" port="1" href="javascript:;" border="0" tooltip=")" << e.p->e[1].w << "\">" << (e.p->e[1].w.approximatelyZero() ? "&nbsp;0 " : "<font color=\"white\">&nbsp;0 </font>") << R"(</td></tr>)";
        os << R"(<tr><td sides="R"></td><td sides="L"></td></tr>)";
        os << R"(<tr><td colspan="2" sides="B"></td><td colspan="2" rowspan="2" border="0"><font point-size="24">q<sub><font point-size="16">)" << static_cast<std::size_t>(e.p->v) << R"(</font></sub></font></td><td colspan="2" sides="B"></td></tr>)";
        os << R"(<tr><td sides="T" colspan="2"></td><td sides="T" colspan="2"></td></tr>)";
        os << R"(<tr><td colspan="2" rowspan="2" port="2" href="javascript:;" border="0" tooltip=")" << e.p->e[2].w << "\">" << (e.p->e[2].w.approximatelyZero() ? "&nbsp;0 " : "<font color=\"white\">&nbsp;0 </font>")
           << R"(</td><td sides="R"></td><td sides="L"></td>)"
           << R"(<td colspan="2" rowspan="2" port="3" href="javascript:;" border="0" tooltip=")" << e.p->e[3].w << "\">" << (e.p->e[3].w.approximatelyZero() ? "&nbsp;0 " : "<font color=\"white\">&nbsp;0 </font>") << "</td></tr>";
        os << R"(<tr><td sides="R"></td><td sides="L"></td></tr>)";
        os << "</table></font>>,tooltip=\"q" << static_cast<std::size_t>(e.p->v) << "\"]\n";
        return os;
    }
    [[maybe_unused]] static std::ostream& modernNode(const Package::vEdge& e, std::ostream& os) {
        auto nodelabel = (reinterpret_cast<std::uintptr_t>(e.p) & 0x001fffffU) >> 1U; // this allows for 2^20 (roughly 1e6) unique nodes
        os << nodelabel << "[label=<";
        os << R"(<font point-size="8"><table border="1" cellspacing="0" cellpadding="0" style="rounded">)";
        os << R"(<tr><td colspan="2" border="0" cellpadding="1"><font point-size="20">q<sub><font point-size="12">)" << static_cast<std::size_t>(e.p->v) << R"(</font></sub></font></td></tr><tr>)";
        os << R"(<td height="6" width="14" port="0" tooltip=")" << e.p->e[0].w << R"(" href="javascript:;" sides="RT">)" << (e.p->e[0].w.approximatelyZero() ? "&nbsp;0 " : R"(<font color="white">&nbsp;0 </font>)") << "</td>";
        os << R"(<td height="6" width="14" port="1" tooltip=")" << e.p->e[1].w << R"(" href="javascript:;" sides="LT">)" << (e.p->e[1].w.approximatelyZero() ? "&nbsp;0 " : R"(<font color="white">&nbsp;0 </font>)") << "</td>";
        os << "</tr></table></font>>,tooltip=\"q" << static_cast<std::size_t>(e.p->v) << "\"]\n";
        return os;
    }
    [[maybe_unused]] static std::ostream& classicNode(const Package::mEdge& e, std::ostream& os) {
        auto nodelabel = (reinterpret_cast<std::uintptr_t>(e.p) & 0x001fffffU) >> 1U; // this allows for 2^20 (roughly 1e6) unique nodes
        os << nodelabel << "[shape=circle, width=0.53, fixedsize=true, label=<";
        os << R"(<font point-size="6"><table border="0" cellspacing="0" cellpadding="0">)";
        os << R"(<tr><td colspan="4"><font point-size="18">q<sub><font point-size="10">)" << static_cast<std::size_t>(e.p->v) << R"(</font></sub></font></td></tr><tr>)";
        os << R"(<td port="0" tooltip=")" << e.p->e[0].w << R"(" href="javascript:;">)";
        if (e.p->e[0].w.approximatelyZero()) {
            os << R"(<font point-size="8">&nbsp;0 </font>)";
        } else {
            os << R"(<font color="white">&nbsp;0 </font>)";
        }
        os << "</td>";
        os << "<td></td><td></td>";
        os << R"(<td port="3" tooltip=")" << e.p->e[3].w << R"(" href="javascript:;">)";
        if (e.p->e[3].w.approximatelyZero()) {
            os << R"(<font point-size="8">&nbsp;0 </font>)";
        } else {
            os << R"(<font color="white">&nbsp;0 </font>)";
        }
        os << "</td>";
        os << "</tr><tr><td></td>";
        os << R"(<td port="1" tooltip=")" << e.p->e[1].w << R"(" href="javascript:;">)";
        if (e.p->e[1].w.approximatelyZero()) {
            os << R"(<font point-size="8">&nbsp;0 </font>)";
        } else {
            os << R"(<font color="white">&nbsp;0 </font>)";
        }
        os << "</td>";
        os << R"(<td port="2" tooltip=")" << e.p->e[2].w << R"(" href="javascript:;">)";
        if (e.p->e[2].w.approximatelyZero()) {
            os << R"(<font point-size="8">&nbsp;0 </font>)";
        } else {
            os << R"(<font color="white">&nbsp;0 </font>)";
        }
        os << "</td>";
        os << "<td></td></tr></table></font>>,tooltip=\"q" << static_cast<std::size_t>(e.p->v) << "\"]\n";
        return os;
    }
    [[maybe_unused]] static std::ostream& classicNode(const Package::vEdge& e, std::ostream& os) {
        auto nodelabel = (reinterpret_cast<std::uintptr_t>(e.p) & 0x001fffffU) >> 1U; // this allows for 2^20 (roughly 1e6) unique nodes
        os << nodelabel << "[shape=circle, width=0.46, fixedsize=true, label=<";
        os << R"(<font point-size="6"><table border="0" cellspacing="0" cellpadding="0">)";
        os << R"(<tr><td colspan="2"><font point-size="18">q<sub><font point-size="10">)" << static_cast<std::size_t>(e.p->v) << R"(</font></sub></font></td></tr><tr>)";
        os << R"(<td port="0" tooltip=")" << e.p->e[0].w << R"(" href="javascript:;">)";
        if (e.p->e[0].w.approximatelyZero()) {
            os << R"(<font point-size="10">&nbsp;0 </font>)";
        } else {
            os << R"(<font color="white">&nbsp;0 </font>)";
        }
        os << "</td>";
        os << R"(<td port="1" tooltip=")" << e.p->e[1].w << R"(" href="javascript:;">)";
        if (e.p->e[1].w.approximatelyZero()) {
            os << R"(<font point-size="10">&nbsp;0 </font>)";
        } else {
            os << R"(<font color="white">&nbsp;0 </font>)";
        }
        os << "</td>";
        os << "</tr></table></font>>,tooltip=\"q" << static_cast<std::size_t>(e.p->v) << "\"]\n";
        return os;
    }
    template<class Edge>
    static std::ostream& memoryNode(const Edge& e, std::ostream& os) {
        constexpr std::size_t N         = std::tuple_size_v<decltype(e.p->e)>;
        auto                  nodelabel = (reinterpret_cast<std::uintptr_t>(e.p) & 0x001fffffU) >> 1U; // this allows for 2^20 (roughly 1e6) unique nodes
        os << nodelabel << "[label=<";
        os << R"(<font point-size="10"><table border="1" cellspacing="0" cellpadding="2" style="rounded">)";
        os << R"(<tr><td colspan=")" << N << R"(" border="1" sides="B">)" << std::hex << reinterpret_cast<std::uintptr_t>(e.p) << std::dec << " ref: " << e.p->ref << "</td></tr>";
        os << "<tr>";
        for (std::size_t i = 0; i < N; ++i) {
            os << "<td port=\"" << i << R"(" href="javascript:;" border="0" tooltip=")" << e.p->e[i].w.toString(false, 4) << "\">";
            if (e.p->e[i] == Edge::zero) {
                os << "&nbsp;0 "
                      "";
            } else {
                os << "<font color=\"white\">&nbsp;0 </font>";
            }
            os << "</td>";
        }
        os << "</tr>";
        os << "</table></font>>,tooltip=\"" << std::hex << reinterpret_cast<std::uintptr_t>(e.p) << "\"]\n"
           << std::dec;
        return os;
    }

    [[maybe_unused]] static std::ostream& bwEdge(const Package::mEdge& from, const Package::mEdge& to, short idx, std::ostream& os, bool edgeLabels = false, bool classic = false) {
        auto fromlabel = (reinterpret_cast<std::uintptr_t>(from.p) & 0x001fffffU) >> 1U;
        auto tolabel   = (reinterpret_cast<std::uintptr_t>(to.p) & 0x001fffffU) >> 1U;

        os << fromlabel << ":" << idx << ":";
        if (classic) {
            if (idx == 0)
                os << "sw";
            else if (idx == 1 || idx == 2)
                os << "s";
            else
                os << "se";
        } else {
            if (idx == 0)
                os << "sw";
            else if (idx == 1)
                os << "se";
            else
                os << 's';
        }
        os << "->";
        if (to.isTerminal()) {
            os << "t";
        } else {
            os << tolabel;
        }

        auto mag = thicknessFromMagnitude(to.w);
        os << "[penwidth=\"" << mag << "\",tooltip=\"" << to.w << "\"";
        if (!to.w.approximatelyOne()) {
            os << ",style=dashed";
        }
        if (edgeLabels) {
            os << ",label=<<font point-size=\"8\">&nbsp;" << to.w << "</font>>";
        }
        os << "]\n";

        return os;
    }
    [[maybe_unused]] static std::ostream& bwEdge(const Package::vEdge& from, const Package::vEdge& to, short idx, std::ostream& os, bool edgeLabels = false, [[maybe_unused]] bool classic = false) {
        auto fromlabel = (reinterpret_cast<std::uintptr_t>(from.p) & 0x001fffffU) >> 1U;
        auto tolabel   = (reinterpret_cast<std::uintptr_t>(to.p) & 0x001fffffU) >> 1U;

        os << fromlabel << ":" << idx << ":";
        os << (idx == 0 ? "sw" : "se") << "->";
        if (to.isTerminal()) {
            os << "t";
        } else {
            os << tolabel;
        }

        auto mag = thicknessFromMagnitude(to.w);
        os << "[penwidth=\"" << mag << "\",tooltip=\"" << to.w << "\"";
        if (!to.w.approximatelyOne()) {
            os << ",style=dashed";
        }
        if (edgeLabels) {
            os << ",label=<<font point-size=\"8\">&nbsp;" << to.w << "</font>>";
        }
        os << "]\n";

        return os;
    }
    [[maybe_unused]] static std::ostream& coloredEdge(const Package::mEdge& from, const Package::mEdge& to, short idx, std::ostream& os, bool edgeLabels = false, bool classic = false) {
        auto fromlabel = (reinterpret_cast<std::uintptr_t>(from.p) & 0x001fffffU) >> 1U;
        auto tolabel   = (reinterpret_cast<std::uintptr_t>(to.p) & 0x001fffffU) >> 1U;

        os << fromlabel << ":" << idx << ":";
        if (classic) {
            if (idx == 0)
                os << "sw";
            else if (idx == 1 || idx == 2)
                os << "s";
            else
                os << "se";
        } else {
            if (idx == 0)
                os << "sw";
            else if (idx == 1)
                os << "se";
            else
                os << 's';
        }
        os << "->";
        if (to.isTerminal()) {
            os << "t";
        } else {
            os << tolabel;
        }

        auto mag   = thicknessFromMagnitude(to.w);
        auto color = colorFromPhase(to.w);
        os << "[penwidth=\"" << mag << "\",tooltip=\"" << to.w << "\" color=\"" << color << "\"";
        if (edgeLabels) {
            os << ",label=<<font point-size=\"8\">&nbsp;" << to.w << "</font>>";
        }
        os << "]\n";

        return os;
    }
    [[maybe_unused]] static std::ostream& coloredEdge(const Package::vEdge& from, const Package::vEdge& to, short idx, std::ostream& os, bool edgeLabels = false, [[maybe_unused]] bool classic = false) {
        auto fromlabel = (reinterpret_cast<std::uintptr_t>(from.p) & 0x001fffffU) >> 1U;
        auto tolabel   = (reinterpret_cast<std::uintptr_t>(to.p) & 0x001fffffU) >> 1U;

        os << fromlabel << ":" << idx << ":";
        os << (idx == 0 ? "sw" : "se") << "->";
        if (to.isTerminal()) {
            os << "t";
        } else {
            os << tolabel;
        }

        auto mag   = thicknessFromMagnitude(to.w);
        auto color = colorFromPhase(to.w);
        os << "[penwidth=\"" << mag << "\",tooltip=\"" << to.w << "\" color=\"" << color << "\"";
        if (edgeLabels) {
            os << ",label=<<font point-size=\"8\">&nbsp;" << to.w << "</font>>";
        }
        os << "]\n";

        return os;
    }
    template<class Edge>
    static std::ostream& memoryEdge(const Edge& from, const Edge& to, short idx, std::ostream& os, bool edgeLabels = false) {
        auto fromlabel = (reinterpret_cast<std::uintptr_t>(from.p) & 0x001fffffU) >> 1U;
        auto tolabel   = (reinterpret_cast<std::uintptr_t>(to.p) & 0x001fffffU) >> 1U;

        os << fromlabel << ":" << idx << ":s->";
        if (to.isTerminal()) {
            os << "t";
        } else {
            os << tolabel;
        }

        auto mag   = thicknessFromMagnitude(to.w);
        auto color = colorFromPhase(to.w);
        os << "[penwidth=\"" << mag << "\",tooltip=\"" << to.w.toString(false, 4) << "\" color=\"" << color << "\"";
        if (edgeLabels) {
            os << ",label=<<font point-size=\"8\">&nbsp;[";
            if (to.w == Complex::one) {
                os << "1";
            } else {
                if (to.w.r == &ComplexTable<>::zero) {
                    os << "0";
                } else if (to.w.r == &ComplexTable<>::sqrt2_2) {
                    os << u8"\u221a\u00bd";
                } else if (to.w.r == &ComplexTable<>::one) {
                    os << "1";
                } else {
                    os << std::hex << reinterpret_cast<std::uintptr_t>(to.w.r) << std::dec;
                }
                os << " ";
                if (to.w.i == &ComplexTable<>::zero) {
                    os << "0";
                } else if (to.w.i == &ComplexTable<>::sqrt2_2) {
                    os << u8"\u221a\u00bd";
                } else if (to.w.i == &ComplexTable<>::one) {
                    os << "1";
                } else {
                    os << std::hex << reinterpret_cast<std::uintptr_t>(to.w.i) << std::dec;
                }
            }
            os << "]</font>>";
        }
        os << "]\n";

        return os;
    }

    template<class Edge>
    static void toDot(const Edge& e, std::ostream& os, bool colored = true, bool edgeLabels = false, bool classic = false, bool memory = false) {
        std::ostringstream oss{};
        // header, root and terminal declaration

        if (memory) {
            memoryHeader(e, oss, edgeLabels);
        } else if (colored) {
            coloredHeader(e, oss, edgeLabels);
        } else {
            header(e, oss, edgeLabels);
        }

        std::unordered_set<decltype(e.p)> nodes{};

        auto priocmp = [](const Edge* left, const Edge* right) { return left->p->v < right->p->v; };

        std::priority_queue<const Edge*, std::vector<const Edge*>, decltype(priocmp)> q(priocmp);
        q.push(&e);

        // bfs until finished
        while (!q.empty()) {
            auto node = q.top();
            q.pop();

            // base case
            if (node->isTerminal())
                continue;

            // check if node has already been processed
            auto ret = nodes.emplace(node->p);
            if (!ret.second) continue;

            // node definition as HTML-like label (href="javascript:;" is used as workaround to make tooltips work)
            if (memory) {
                memoryNode(*node, oss);
            } else if (classic) {
                classicNode(*node, oss);
            } else {
                modernNode(*node, oss);
            }

            // iterate over edges in reverse to guarantee correct proceossing order
            for (auto i = static_cast<Qubit>(node->p->e.size() - 1); i >= 0; --i) {
                auto& edge = node->p->e[i];
                if ((!memory && edge.w.approximatelyZero()) || edge.w == Complex::zero) {
                    // potentially add zero stubs here
                    continue;
                }

                // non-zero edge to be included
                q.push(&edge);

                if (memory) {
                    memoryEdge(*node, edge, i, oss, edgeLabels);
                } else if (colored) {
                    coloredEdge(*node, edge, i, oss, edgeLabels, classic);
                } else {
                    bwEdge(*node, edge, i, oss, edgeLabels, classic);
                }
            }
        }
        oss << "}\n";

        os << oss.str() << std::flush;
    }

    template<class Edge>
    [[maybe_unused]] static void export2Dot(Edge basic, const std::string& outputFilename, bool colored = true, bool edgeLabels = false, bool classic = false, bool memory = false, bool show = true) {
        std::ofstream init(outputFilename);
        toDot(basic, init, colored, edgeLabels, classic, memory);
        init.close();

        if (show) {
            std::ostringstream oss;
            oss << "dot -Tsvg " << outputFilename << " -o " << outputFilename.substr(0, outputFilename.find_last_of('.')) << ".svg";
            auto                  str = oss.str(); // required to avoid immediate deallocation of temporary
            [[maybe_unused]] auto rc  = std::system(str.c_str());
        }
    }

    ///
    /// Serialization
    /// Note: do not rely on the binary format being portable across different architectures/platforms
    ///

    [[maybe_unused]] static void serialize(const Package::vEdge& basic, std::ostream& os, bool writeBinary = false) {
        if (writeBinary) {
            os.write(reinterpret_cast<const char*>(&SERIALIZATION_VERSION), sizeof(decltype(SERIALIZATION_VERSION)));
            basic.w.writeBinary(os);
        } else {
            os << SERIALIZATION_VERSION << "\n";
            os << basic.w.toString(false, 16) << "\n";
        }
        std::int_least64_t                                      next_index = 0;
        std::unordered_map<Package::vNode*, std::int_least64_t> node_index{};

        // POST ORDER TRAVERSAL USING ONE STACK   https://www.geeksforgeeks.org/iterative-postorder-traversal-using-stack/
        std::stack<const Package::vEdge*> stack{};

        auto node = &basic;
        if (!node->isTerminal()) {
            do {
                while (node != nullptr && !node->isTerminal()) {
                    for (short i = RADIX - 1; i > 0; --i) {
                        auto& edge = node->p->e[i];
                        if (edge.isTerminal()) continue;
                        if (edge.w.approximatelyZero()) continue;
                        if (node_index.find(edge.p) != node_index.end()) continue;

                        // non-zero edge to be included
                        stack.push(&edge);
                    }
                    stack.push(node);
                    node = &node->p->e[0];
                }
                node = stack.top();
                stack.pop();

                bool hasChild = false;
                for (auto i = 1U; i < RADIX && !hasChild; ++i) {
                    auto& edge = node->p->e[i];
                    if (edge.w.approximatelyZero()) continue;
                    if (node_index.find(edge.p) != node_index.end()) continue;
                    if (!stack.empty())
                        hasChild = edge.p == stack.top()->p;
                }

                if (hasChild) {
                    const auto temp = stack.top();
                    stack.pop();
                    stack.push(node);
                    node = temp;
                } else {
                    if (node_index.find(node->p) != node_index.end()) {
                        node = nullptr;
                        continue;
                    }
                    node_index[node->p] = next_index;
                    next_index++;

                    if (writeBinary) {
                        os.write(reinterpret_cast<const char*>(&node_index[node->p]), sizeof(decltype(node_index[node->p])));
                        os.write(reinterpret_cast<const char*>(&node->p->v), sizeof(decltype(node->p->v)));

                        // iterate over edges in reverse to guarantee correct processing order
                        for (auto i = 0U; i < RADIX; ++i) {
                            auto&              edge     = node->p->e[i];
                            std::int_least64_t edge_idx = edge.isTerminal() ? -1 : node_index[edge.p];
                            os.write(reinterpret_cast<const char*>(&edge_idx), sizeof(decltype(edge_idx)));
                            edge.w.writeBinary(os);
                        }
                    } else {
                        os << node_index[node->p] << " " << static_cast<std::size_t>(node->p->v);

                        // iterate over edges in reverse to guarantee correct processing order
                        for (auto i = 0U; i < RADIX; ++i) {
                            os << " (";
                            auto& edge = node->p->e[i];
                            if (!edge.w.approximatelyZero()) {
                                std::int_least64_t edge_idx = edge.isTerminal() ? -1 : node_index[edge.p];
                                os << edge_idx << " " << edge.w.toString(false, 16);
                            }
                            os << ")";
                        }
                        os << "\n";
                    }
                    node = nullptr;
                }
            } while (!stack.empty());
        }
    }
    static void serializeMatrix(const Package::mEdge& basic, std::int_least64_t& idx, std::unordered_map<Package::mNode*, std::int_least64_t>& node_index, std::unordered_set<Package::mNode*>& visited, std::ostream& os, bool writeBinary = false) {
        if (!basic.isTerminal()) {
            for (auto& e: basic.p->e) {
                if (auto [iter, success] = visited.insert(e.p); success) {
                    serializeMatrix(e, idx, node_index, visited, os, writeBinary);
                }
            }

            if (node_index.find(basic.p) == node_index.end()) {
                node_index[basic.p] = idx;
                ++idx;
            }

            if (writeBinary) {
                os.write(reinterpret_cast<const char*>(&node_index[basic.p]), sizeof(decltype(node_index[basic.p])));
                os.write(reinterpret_cast<const char*>(&basic.p->v), sizeof(decltype(basic.p->v)));

                // iterate over edges in reverse to guarantee correct processing order
                for (auto& edge: basic.p->e) {
                    std::int_least64_t edge_idx = edge.isTerminal() ? -1 : node_index[edge.p];
                    os.write(reinterpret_cast<const char*>(&edge_idx), sizeof(decltype(edge_idx)));
                    edge.w.writeBinary(os);
                }
            } else {
                os << node_index[basic.p] << " " << static_cast<std::size_t>(basic.p->v);

                // iterate over edges in reverse to guarantee correct processing order
                for (auto& edge: basic.p->e) {
                    os << " (";
                    if (!edge.w.approximatelyZero()) {
                        std::int_least64_t edge_idx = edge.isTerminal() ? -1 : node_index[edge.p];
                        os << edge_idx << " " << edge.w.toString(false, 16);
                    }
                    os << ")";
                }
                os << "\n";
            }
        }
    }
    [[maybe_unused]] static void serialize(const Package::mEdge& basic, std::ostream& os, bool writeBinary = false) {
        if (writeBinary) {
            os.write(reinterpret_cast<const char*>(&SERIALIZATION_VERSION), sizeof(decltype(SERIALIZATION_VERSION)));
            basic.w.writeBinary(os);
        } else {
            os << SERIALIZATION_VERSION << "\n";
            os << basic.w.toString(false, 16) << "\n";
        }
        std::int_least64_t                                      idx = 0;
        std::unordered_map<Package::mNode*, std::int_least64_t> node_index{};
        std::unordered_set<Package::mNode*>                     visited{};
        serializeMatrix(basic, idx, node_index, visited, os, writeBinary);
    }
    template<class Edge>
    static void serialize(const Edge& basic, const std::string& outputFilename, bool writeBinary = false) {
        std::ofstream ofs = std::ofstream(outputFilename, std::ios::binary);

        if (!ofs.good()) {
            throw std::invalid_argument("Cannot open file: " + outputFilename);
        }

        serialize(basic, ofs, writeBinary);
    }

} // namespace dd

#endif //DDexport_H
