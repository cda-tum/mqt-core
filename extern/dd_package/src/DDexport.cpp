/*
 * This file is part of IIC-JKU DD package which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum_dd/ for more information.
 */

#include "DDexport.h"

namespace dd {
	std::ostream& header(const Edge& e, std::ostream& os) {
		os << "digraph \"DD\" {graph[];node[shape=plain];edge[arrowhead=none]\n";
		os << "root [label=\"\",shape=point,style=invis]\n";
		os << "t [label=<<font POINT-SIZE=\"10\">1</font>>,shape=box,tooltip=\"1\",width=0.3,height=0.3]\n";
		auto toplabel = ((uintptr_t)e.p & 0x001fffffu) >> 1u;
		auto mag = thicknessFromMagnitude(e.w);
		os << "root->" << toplabel << "[penwidth=\"" << mag <<  "\",tooltip=\"" << e.w << "\"";
		if (!CN::equalsOne(e.w)) {
			os << ",style=dashed";
		}
		os << "]\n";

		return os;
	}

	std::ostream& coloredHeader(const Edge& e, std::ostream& os) {
		os << "digraph \"DD\" {graph[];node[shape=plain];edge[arrowhead=none]\n";
		os << "root [label=\"\",shape=point,style=invis]\n";
		os << "t [label=<<font POINT-SIZE=\"10\">1</font>>,shape=box,tooltip=\"1\",width=0.3,height=0.3]\n";
		auto toplabel = ((uintptr_t)e.p & 0x001fffffu) >> 1u;
		auto mag = thicknessFromMagnitude(e.w);
		auto color = colorFromPhase(e.w);
		os << "root->" << toplabel << "[penwidth=\"" << mag << "\",tooltip=\"" << e.w << "\" color=\"#" << color << "\"]\n";
		return os;
	}

	std::ostream& matrixNodeMatrixAndXlabel(const Edge& e, std::ostream& os) {
		auto nodelabel = ((uintptr_t)e.p & 0x001fffffu) >> 1u; // this allows for 2^20 (roughly 1e6) unique nodes
		os << nodelabel << "[label=<";
		os << R"(<font point-size="6"><table border="1" cellspacing="0" style="rounded"><tr>)";
		os << R"(<td port="0" tooltip=")" << e.p->e[0].w << R"(" href=" " sides="RB">)" << (CN::equalsZero(e.p->e[0].w) ? "&nbsp;0 " : "<font color=\"white\">&nbsp;0 </font>") << "</td>";
		os << R"(<td port="1" tooltip=")" << e.p->e[1].w << R"(" href=" " sides="LB">)" << (CN::equalsZero(e.p->e[1].w) ? "&nbsp;0 " : "<font color=\"white\">&nbsp;0 </font>") << "</td></tr><tr>";
		os << R"(<td port="2" tooltip=")" << e.p->e[2].w << R"(" href=" " sides="RT">)" << (CN::equalsZero(e.p->e[2].w) ? "&nbsp;0 " : "<font color=\"white\">&nbsp;0 </font>") << "</td>";
		os << R"(<td port="3" tooltip=")" << e.p->e[3].w << R"(" href=" " sides="LT">)" << (CN::equalsZero(e.p->e[3].w) ? "&nbsp;0 " : "<font color=\"white\">&nbsp;0 </font>") << "</td>";
		os << "</tr></table></font>>,tooltip=\"q" << e.p->v << "\"" << R"(,xlabel=<<font point-size="8">q<sub><font point-size="6">)" << e.p->v << "</font></sub></font>>]\n";
		return os;
	}

	std::ostream& matrixNodeMiddleVar(const Edge& e, std::ostream& os) {
		auto nodelabel = ((uintptr_t)e.p & 0x001fffffu) >> 1u; // this allows for 2^20 (roughly 1e6) unique nodes
		os << nodelabel << "[label=<";
		os << R"(<font point-size="12"><table border="1" cellspacing="0" cellpadding="2" style="rounded">)";
		os << R"(<tr><td colspan="2" rowspan="2" port="0" href=" " border="0" tooltip=")" << e.p->e[0].w << "\">" << (CN::equalsZero(e.p->e[0].w) ? "&nbsp;0 " : "<font color=\"white\">&nbsp;0 </font>")
		<< R"(</td><td sides="R"></td><td sides="L"></td>)"
		<< R"(<td colspan="2" rowspan="2" port="1" href=" " border="0" tooltip=")" << e.p->e[1].w << "\">" << (CN::equalsZero(e.p->e[1].w) ? "&nbsp;0 " : "<font color=\"white\">&nbsp;0 </font>")<< R"(</td></tr>)";
		os << R"(<tr><td sides="R"></td><td sides="L"></td></tr>)";
		os << R"(<tr><td colspan="2" sides="B"></td><td colspan="2" rowspan="2" border="0"><font point-size="12">q<sub><font point-size="8">)" << e.p->v << R"(</font></sub></font></td><td colspan="2" sides="B"></td></tr>)";
		os << R"(<tr><td sides="T" colspan="2"></td><td sides="T" colspan="2"></td></tr>)";
		os << R"(<tr><td colspan="2" rowspan="2" port="2" href=" " border="0" tooltip=")" << e.p->e[2].w << "\">" << (CN::equalsZero(e.p->e[2].w) ? "&nbsp;0 " : "<font color=\"white\">&nbsp;0 </font>")
		   << R"(</td><td sides="R"></td><td sides="L"></td>)"
		   << R"(<td colspan="2" rowspan="2" port="3" href=" " border="0" tooltip=")" << e.p->e[3].w << "\">" << (CN::equalsZero(e.p->e[3].w) ? "&nbsp;0 " : "<font color=\"white\">&nbsp;0 </font>")<< "</td></tr>";
		os << R"(<tr><td sides="R"></td><td sides="L"></td></tr>)";
		os << "</table></font>>,tooltip=\"q" << e.p->v << "\"]\n";
		return os;
	}


	std::ostream& vectorNode(const Edge& e, std::ostream& os) {
		auto nodelabel = ((uintptr_t)e.p & 0x001fffffu) >> 1u; // this allows for 2^20 (roughly 1e6) unique nodes
		os << nodelabel << "[label=<";
		os << R"(<table border="1" cellspacing="0" cellpadding="0" style="rounded">)";
		os << R"(<tr><td colspan="2" border="0" cellpadding="2">q<sub><font point-size="8">)" << e.p->v << "</font></sub></td></tr><tr>";
		os << R"(<td port="0" tooltip=")" << e.p->e[0].w << R"(" href=" " sides="RT" height="8" width="12"><font point-size="5">&nbsp;0 </font></td>)";
		os << R"(<td port="2" tooltip=")" << e.p->e[2].w << R"(" href=" " sides="LT" height="8" width="12"><font point-size="5">&nbsp;1 </font></td>)";
		os << "</tr></table>>,tooltip=\"q" << e.p->v << "\"]\n";
		return os;
	}

	std::ostream& matrixEdge(const Edge& from, const Edge& to, short idx, std::ostream& os, bool edgeLabels) {
		auto fromlabel = ((uintptr_t)from.p & 0x001fffffu) >> 1u;
		auto tolabel = ((uintptr_t)to.p & 0x001fffffu) >> 1u;

		os << fromlabel << ":" << idx << ":";
		if (idx == 0) os << "sw";
		else if (idx == 1) os << "se";
		else os << 's';
		os << "->";
		if (Package::isTerminal(to)) {
			os << "t";
		} else {
			os << tolabel << ":n";
		}

		auto mag = thicknessFromMagnitude(to.w);
		os << "[penwidth=\"" << mag << "\",tooltip=\"" << to.w << "\"";
		if (!CN::equalsOne(to.w)) {
			os << ",style=dashed";
			if (edgeLabels) {
				os << ",label=<<font point-size=\"8\">&nbsp;" << to.w << "</font>>";
			}
		}
		os << "]\n";

		return os;
	}

	std::ostream& coloredMatrixEdge(const Edge& from, const Edge& to, short idx, std::ostream& os, bool edgeLabels) {
		auto fromlabel = ((uintptr_t)from.p & 0x001fffffu) >> 1u;
		auto tolabel = ((uintptr_t)to.p & 0x001fffffu) >> 1u;

		os << fromlabel << ":" << idx << ":";
		if (idx == 0) os << "sw";
		else if (idx == 1) os << "se";
		else os << 's';
		os << "->";
		if (Package::isTerminal(to)) {
			os << "t";
		} else {
			os << tolabel << ":n";
		}

		auto mag = thicknessFromMagnitude(to.w);
		auto color = colorFromPhase(to.w);
		os << "[penwidth=\"" << mag << "\",tooltip=\"" << to.w << "\" color=\"#" << color << "\"";
		if (!CN::equalsOne(to.w)) {
			if (edgeLabels) {
				os << ",label=<<font point-size=\"8\">&nbsp;" << to.w << "</font>>";
			}
		}
		os << "]\n";

		return os;
	}

	std::ostream& vectorEdge(const Edge& from, const Edge& to, short idx, std::ostream& os, bool edgeLabels) {
		auto fromlabel = ((uintptr_t)from.p & 0x001fffffu) >> 1u;
		auto tolabel = ((uintptr_t)to.p & 0x001fffffu) >> 1u;

		os << fromlabel << ":" << idx << ":";
		os << (idx == 0 ? "sw" : "se") << "->";
		if (Package::isTerminal(to)) {
			os << "t";
		} else {
			os << tolabel;
		}

		auto mag = thicknessFromMagnitude(to.w);
		os << "[penwidth=\"" << mag << "\",tooltip=\"" << to.w << "\"";
		if (!CN::equalsOne(to.w)) {
			os << ",style=dashed";
			if (edgeLabels) {
				os << ",label=<<font point-size=\"8\">&nbsp;" << to.w << "</font>>";
			}
		}
		os << "]\n";

		return os;
	}

	std::ostream& coloredVectorEdge(const Edge& from, const Edge& to, short idx, std::ostream& os, bool edgeLabels) {
		auto fromlabel = ((uintptr_t)from.p & 0x001fffffu) >> 1u;
		auto tolabel = ((uintptr_t)to.p & 0x001fffffu) >> 1u;

		os << fromlabel << ":" << idx << ":";
		os << (idx == 0 ? "sw" : "se") << "->";
		if (Package::isTerminal(to)) {
			os << "t";
		} else {
			os << tolabel;
		}

		auto mag = thicknessFromMagnitude(to.w);
		auto color = colorFromPhase(to.w);
		os << "[penwidth=\"" << mag << "\",tooltip=\"" << to.w << "\" color=\"#" << color << "\"";
		if (!CN::equalsOne(to.w)) {
			if (edgeLabels) {
				os << ",label=<<font point-size=\"8\">&nbsp;" << to.w << "</font>>";
			}
		}
		os << "]\n";

		return os;
	}

	void toDot(const Edge& e, std::ostream& os, bool isVector, bool colored, bool edgeLabels) {
		std::ostringstream oss{};
		// header, root and terminal declaration

		if (colored) {
			coloredHeader(e, oss);
		} else {
			header(e, oss);
		}

		std::unordered_set<NodePtr> nodes{};
		auto priocmp = [] (const dd::Edge* left, const dd::Edge* right) { return left->p->v < right->p->v; };
		std::priority_queue<const dd::Edge*, std::vector<const dd::Edge*>, decltype(priocmp)> q(priocmp);
		q.push(&e);

		// bfs until finished
		while (!q.empty()) {
			auto node = q.top();
			q.pop();

			// base case
			if (Package::isTerminal(*node))
				continue;

			// check if node has already been processed
			auto ret = nodes.emplace(node->p);
			if (!ret.second) continue;

			// node definition as HTML-like label (href=" " is used as workaround to make tooltips work)
			if (isVector) {
				vectorNode(*node, oss);
			} else {
				//matrixNodeMatrixAndXlabel(*node, oss);
				matrixNodeMiddleVar(*node, oss);
			}

			// iterate over edges in reverse to guarantee correct proceossing order
			for (short i=dd::NEDGE-1; i >= 0; --i) {
				if (isVector && i%2 != 0)
					continue;

				auto& edge = node->p->e[i];
				if (CN::equalsZero(edge.w))
					continue;

				// non-zero edge to be included
				q.push(&edge);

				if (isVector) {
					if (colored) {
						coloredVectorEdge(*node, edge, i, oss, edgeLabels);
					} else {
						vectorEdge(*node, edge, i, oss, edgeLabels);
					}
				} else {
					if (colored) {
						coloredMatrixEdge(*node, edge, i, oss, edgeLabels);
					} else {
						matrixEdge(*node, edge, i, oss, edgeLabels);
					}
				}
			}
		}
		oss << "}\n";

		os << oss.str() << std::flush;
	}

	void export2Dot(Edge basic, const std::string& outputFilename, bool isVector, bool colored, bool edgeLabels, bool show) {
		std::ofstream init(outputFilename);
		toDot(basic, init, isVector, colored, edgeLabels);
		init.close();

		if (show) {
			std::ostringstream oss;
			oss << "dot -Tsvg " << outputFilename << " -o " << outputFilename.substr(0, outputFilename.find_last_of('.')) << ".svg";
			auto str = oss.str(); // required to avoid immediate deallocation of temporary
			static_cast<void>(!std::system(str.c_str())); // cast and ! just to suppress the unused result warning
		}
	}

	fp hueToRGB(fp hue) {
		if (hue < 0) hue += 1.0;
		else if (hue > 1) hue -= 1.0;
		if (hue < 1./6) return 0.25 + 3*hue;
		if (hue < 1./2) return 0.75;
		if (hue < 2./3) return 0.25 + 3*(2./3 - hue);
		return 0.25;
	}

	RGB colorFromPhase(const Complex& a) {
		auto phase = CN::arg(a);
		auto twopi = 2*PI;
		phase = (phase + PI) / twopi;
		return {hueToRGB(phase+1./3), hueToRGB(phase), hueToRGB(phase-1./3)};
	}

	fp thicknessFromMagnitude (const Complex& a) {
		return 3.0*std::max(CN::mag(a), 0.10);
	}

}
