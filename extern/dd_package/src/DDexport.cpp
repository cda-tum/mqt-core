/*
 * This file is part of IIC-JKU DD package which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum_dd/ for more information.
 */

#include "DDexport.h"

namespace dd {
	std::ostream& header(const Edge& e, std::ostream& os, bool edgeLabels) {
		os << "digraph \"DD\" {graph[];node[shape=plain];edge[arrowhead=none]\n";
		os << "root [label=\"\",shape=point,style=invis]\n";
		os << "t [label=<<font point-size=\"20\">1</font>>,shape=box,tooltip=\"1\",width=0.3,height=0.3]\n";
		auto toplabel = ((uintptr_t)e.p & 0x001fffffu) >> 1u;
		auto mag = thicknessFromMagnitude(e.w);
		os << "root->";
		if (dd::Package::isTerminal(e)) {
			os << "t";
		} else {
			os << toplabel;
		}
		os<< "[penwidth=\"" << mag <<  "\",tooltip=\"" << e.w << "\"";
		if (!CN::equalsOne(e.w)) {
			os << ",style=dashed";
		}
		if (edgeLabels) {
		    os << ",label=<<font point-size=\"8\">&nbsp;" << e.w << "</font>>";
		}

		os << "]\n";

		return os;
	}

	std::ostream& coloredHeader(const Edge& e, std::ostream& os, bool edgeLabels) {
		os << "digraph \"DD\" {graph[];node[shape=plain];edge[arrowhead=none]\n";
		os << "root [label=\"\",shape=point,style=invis]\n";
		os << "t [label=<<font point-size=\"20\">1</font>>,shape=box,tooltip=\"1\",width=0.3,height=0.3]\n";

		auto toplabel = ((uintptr_t)e.p & 0x001fffffu) >> 1u;
		auto mag = thicknessFromMagnitude(e.w);
		auto color = colorFromPhase(e.w);
		os << "root->";
		if (dd::Package::isTerminal(e)) {
			os << "t";
		} else {
			os << toplabel;
		}
		os << "[penwidth=\"" << mag << "\",tooltip=\"" << e.w << "\",color=\"#" << color << "\"";
		if (edgeLabels) {
			os << ",label=<<font point-size=\"8\">&nbsp;" << e.w << "</font>>";
		}
		os << "]\n";
		return os;
	}

	std::ostream& matrixNodeMatrixAndXlabel(const Edge& e, std::ostream& os) {
		auto nodelabel = ((uintptr_t)e.p & 0x001fffffu) >> 1u; // this allows for 2^20 (roughly 1e6) unique nodes
		os << nodelabel << "[label=<";
		os << R"(<font point-size="6"><table border="1" cellspacing="0" style="rounded"><tr>)";
		os << R"(<td port="0" tooltip=")" << e.p->e[0].w << R"(" href="javascript:;" sides="RB">)" << (CN::equalsZero(e.p->e[0].w) ? "&nbsp;0 " : "<font color=\"white\">&nbsp;0 </font>") << "</td>";
		os << R"(<td port="1" tooltip=")" << e.p->e[1].w << R"(" href="javascript:;" sides="LB">)" << (CN::equalsZero(e.p->e[1].w) ? "&nbsp;0 " : "<font color=\"white\">&nbsp;0 </font>") << "</td></tr><tr>";
		os << R"(<td port="2" tooltip=")" << e.p->e[2].w << R"(" href="javascript:;" sides="RT">)" << (CN::equalsZero(e.p->e[2].w) ? "&nbsp;0 " : "<font color=\"white\">&nbsp;0 </font>") << "</td>";
		os << R"(<td port="3" tooltip=")" << e.p->e[3].w << R"(" href="javascript:;" sides="LT">)" << (CN::equalsZero(e.p->e[3].w) ? "&nbsp;0 " : "<font color=\"white\">&nbsp;0 </font>") << "</td>";
		os << "</tr></table></font>>,tooltip=\"q" << e.p->v << "\"" << R"(,xlabel=<<font point-size="8">q<sub><font point-size="6">)" << e.p->v << "</font></sub></font>>]\n";
		return os;
	}

	std::ostream& matrixNodeMiddleVar(const Edge& e, std::ostream& os) {
		auto nodelabel = ((uintptr_t)e.p & 0x001fffffu) >> 1u; // this allows for 2^20 (roughly 1e6) unique nodes
		os << nodelabel << "[label=<";
		os << R"(<font point-size="10"><table border="1" cellspacing="0" cellpadding="2" style="rounded">)";
		os << R"(<tr><td colspan="2" rowspan="2" port="0" href="javascript:;" border="0" tooltip=")" << e.p->e[0].w << "\">" << (CN::equalsZero(e.p->e[0].w) ? "&nbsp;0 " : "<font color=\"white\">&nbsp;0 </font>")
		<< R"(</td><td sides="R"></td><td sides="L"></td>)"
		<< R"(<td colspan="2" rowspan="2" port="1" href="javascript:;" border="0" tooltip=")" << e.p->e[1].w << "\">" << (CN::equalsZero(e.p->e[1].w) ? "&nbsp;0 " : "<font color=\"white\">&nbsp;0 </font>")<< R"(</td></tr>)";
		os << R"(<tr><td sides="R"></td><td sides="L"></td></tr>)";
		os << R"(<tr><td colspan="2" sides="B"></td><td colspan="2" rowspan="2" border="0"><font point-size="24">q<sub><font point-size="16">)" << e.p->v << R"(</font></sub></font></td><td colspan="2" sides="B"></td></tr>)";
		os << R"(<tr><td sides="T" colspan="2"></td><td sides="T" colspan="2"></td></tr>)";
		os << R"(<tr><td colspan="2" rowspan="2" port="2" href="javascript:;" border="0" tooltip=")" << e.p->e[2].w << "\">" << (CN::equalsZero(e.p->e[2].w) ? "&nbsp;0 " : "<font color=\"white\">&nbsp;0 </font>")
		   << R"(</td><td sides="R"></td><td sides="L"></td>)"
		   << R"(<td colspan="2" rowspan="2" port="3" href="javascript:;" border="0" tooltip=")" << e.p->e[3].w << "\">" << (CN::equalsZero(e.p->e[3].w) ? "&nbsp;0 " : "<font color=\"white\">&nbsp;0 </font>")<< "</td></tr>";
		os << R"(<tr><td sides="R"></td><td sides="L"></td></tr>)";
		os << "</table></font>>,tooltip=\"q" << e.p->v << "\"]\n";
		return os;
	}

	std::ostream& classicMatrixNode(const Edge& e, std::ostream& os) {
		auto nodelabel = ((uintptr_t)e.p & 0x001fffffu) >> 1u; // this allows for 2^20 (roughly 1e6) unique nodes
		os << nodelabel << "[shape=circle, width=0.53, fixedsize=true, label=<";
		os << R"(<font point-size="6"><table border="0" cellspacing="0" cellpadding="0">)";
		os << R"(<tr><td colspan="4"><font point-size="18">q<sub><font point-size="10">)" << e.p->v << R"(</font></sub></font></td></tr><tr>)";
		os << R"(<td port="0" tooltip=")" << e.p->e[0].w << R"(" href="javascript:;">)";
		if (CN::equalsZero(e.p->e[0].w)) {
			os << R"(<font point-size="8">&nbsp;0 </font>)";
		} else {
			os << R"(<font color="white">&nbsp;0 </font>)";
		}
		os << "</td>";
		os << "<td></td><td></td>";
		os << R"(<td port="3" tooltip=")" << e.p->e[3].w << R"(" href="javascript:;">)";
		if (CN::equalsZero(e.p->e[3].w)) {
			os << R"(<font point-size="8">&nbsp;0 </font>)";
		} else {
			os << R"(<font color="white">&nbsp;0 </font>)";
		}		os << "</td>";
		os << "</tr><tr><td></td>";
		os << R"(<td port="1" tooltip=")" << e.p->e[1].w << R"(" href="javascript:;">)";
		if (CN::equalsZero(e.p->e[1].w)) {
			os << R"(<font point-size="8">&nbsp;0 </font>)";
		} else {
			os << R"(<font color="white">&nbsp;0 </font>)";
		}		os << "</td>";
		os << R"(<td port="2" tooltip=")" << e.p->e[2].w << R"(" href="javascript:;">)";
		if (CN::equalsZero(e.p->e[2].w)) {
			os << R"(<font point-size="8">&nbsp;0 </font>)";
		} else {
			os << R"(<font color="white">&nbsp;0 </font>)";
		}		os << "</td>";
		os << "<td></td></tr></table></font>>,tooltip=\"q" << e.p->v << "\"]\n";
		return os;
	}

	std::ostream& vectorNode(const Edge& e, std::ostream& os) {
		auto nodelabel = ((uintptr_t)e.p & 0x001fffffu) >> 1u; // this allows for 2^20 (roughly 1e6) unique nodes
		os << nodelabel << "[label=<";
		os << R"(<font point-size="8"><table border="1" cellspacing="0" cellpadding="0" style="rounded">)";
		os << R"(<tr><td colspan="2" border="0" cellpadding="1"><font point-size="20">q<sub><font point-size="12">)" << e.p->v << R"(</font></sub></font></td></tr><tr>)";
		os << R"(<td height="6" width="14" port="0" tooltip=")" << e.p->e[0].w << R"(" href="javascript:;" sides="RT">)" << (CN::equalsZero(e.p->e[0].w) ? "&nbsp;0 " : R"(<font color="white">&nbsp;0 </font>)") << "</td>";
		os << R"(<td height="6" width="14" port="2" tooltip=")" << e.p->e[2].w << R"(" href="javascript:;" sides="LT">)" << (CN::equalsZero(e.p->e[2].w) ? "&nbsp;0 " : R"(<font color="white">&nbsp;0 </font>)") << "</td>";
		os << "</tr></table></font>>,tooltip=\"q" << e.p->v << "\"]\n";
		return os;
	}

	std::ostream& vectorNodeVectorLook(const Edge& e, std::ostream& os) {
		auto nodelabel = ((uintptr_t)e.p & 0x001fffffu) >> 1u; // this allows for 2^20 (roughly 1e6) unique nodes
		os << nodelabel << "[label=<";
		os << R"(<font point-size="10"><table border="1" cellspacing="0" cellpadding="2" style="rounded">)";
		os << R"(<tr><td rowspan="2" sides="R" cellpadding="2"><font point-size="18">q<sub><font point-size="12">)" << e.p->v << "</font></sub></font></td>";
		os << R"(<td port="0" tooltip=")" << e.p->e[0].w << R"(" href="javascript:;" sides="LB">)" << (CN::equalsZero(e.p->e[0].w) ? "&nbsp;0 " : R"(<font color="white">&nbsp;0 </font>)") << "</td></tr><tr>";
		os << R"(<td port="2" tooltip=")" << e.p->e[2].w << R"(" href="javascript:;" sides="LT">)" << (CN::equalsZero(e.p->e[2].w) ? "&nbsp;0 " : R"(<font color="white">&nbsp;0 </font>)") << "</td>";
		os << "</tr></table></font>>,tooltip=\"q" << e.p->v << "\"]\n";
		return os;
	}

	std::ostream& classicVectorNode(const Edge& e, std::ostream& os) {
		auto nodelabel = ((uintptr_t)e.p & 0x001fffffu) >> 1u; // this allows for 2^20 (roughly 1e6) unique nodes
		os << nodelabel << "[shape=circle, width=0.46, fixedsize=true, label=<";
		os << R"(<font point-size="6"><table border="0" cellspacing="0" cellpadding="0">)";
		os << R"(<tr><td colspan="2"><font point-size="18">q<sub><font point-size="10">)" << e.p->v << R"(</font></sub></font></td></tr><tr>)";
		os << R"(<td port="0" tooltip=")" << e.p->e[0].w << R"(" href="javascript:;">)";
		if (CN::equalsZero(e.p->e[0].w)) {
			os << R"(<font point-size="10">&nbsp;0 </font>)";
		} else {
			os << R"(<font color="white">&nbsp;0 </font>)";
		}
		os << "</td>";
		os << R"(<td port="2" tooltip=")" << e.p->e[2].w << R"(" href="javascript:;">)";
		if (CN::equalsZero(e.p->e[2].w)) {
			os << R"(<font point-size="10">&nbsp;0 </font>)";
		} else {
			os << R"(<font color="white">&nbsp;0 </font>)";
		}
		os << "</td>";
		os << "</tr></table></font>>,tooltip=\"q" << e.p->v << "\"]\n";
		return os;
	}

	std::ostream& matrixEdge(const Edge& from, const Edge& to, short idx, std::ostream& os, bool edgeLabels, bool classic) {
		auto fromlabel = ((uintptr_t)from.p & 0x001fffffu) >> 1u;
		auto tolabel = ((uintptr_t)to.p & 0x001fffffu) >> 1u;

		os << fromlabel << ":" << idx << ":";
		if (classic) {
			if (idx == 0) os << "sw";
			else if (idx == 1 || idx == 2) os << "s";
			else os << "se";
		} else {
			if (idx == 0) os << "sw";
			else if (idx == 1) os << "se";
			else os << 's';
		}
		os << "->";
		if (Package::isTerminal(to)) {
			os << "t";
		} else {
			os << tolabel;
			if (!classic)
				os << ":n";
		}

		auto mag = thicknessFromMagnitude(to.w);
		os << "[penwidth=\"" << mag << "\",tooltip=\"" << to.w << "\"";
		if (!CN::equalsOne(to.w)) {
			os << ",style=dashed";
		}
		if (edgeLabels) {
			os << ",label=<<font point-size=\"8\">&nbsp;" << to.w << "</font>>";
		}
		os << "]\n";

		return os;
	}

	std::ostream& coloredMatrixEdge(const Edge& from, const Edge& to, short idx, std::ostream& os, bool edgeLabels, bool classic) {
		auto fromlabel = ((uintptr_t)from.p & 0x001fffffu) >> 1u;
		auto tolabel = ((uintptr_t)to.p & 0x001fffffu) >> 1u;

		os << fromlabel << ":" << idx << ":";
		if (classic) {
			if (idx == 0) os << "sw";
			else if (idx == 1 || idx == 2) os << "s";
			else os << "se";
		} else {
			if (idx == 0) os << "sw";
			else if (idx == 1) os << "se";
			else os << 's';
		}
		os << "->";
		if (Package::isTerminal(to)) {
			os << "t";
		} else {
			os << tolabel;
			if (!classic)
				os << ":n";
		}

		auto mag = thicknessFromMagnitude(to.w);
		auto color = colorFromPhase(to.w);
		os << "[penwidth=\"" << mag << "\",tooltip=\"" << to.w << "\" color=\"#" << color << "\"";
		if (edgeLabels) {
			os << ",label=<<font point-size=\"8\">&nbsp;" << to.w << "</font>>";
		}
		os << "]\n";

		return os;
	}

	std::ostream& vectorEdge(const Edge& from, const Edge& to, short idx, std::ostream& os, bool edgeLabels, bool classic) {
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
		}
		if (edgeLabels) {
			os << ",label=<<font point-size=\"8\">&nbsp;" << to.w << "</font>>";
		}
		os << "]\n";

		return os;
	}

	std::ostream& coloredVectorEdge(const Edge& from, const Edge& to, short idx, std::ostream& os, bool edgeLabels, bool classic) {
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
		if (edgeLabels) {
			os << ",label=<<font point-size=\"8\">&nbsp;" << to.w << "</font>>";
		}
		os << "]\n";

		return os;
	}

	void toDot(const Edge& e, std::ostream& os, bool isVector, bool colored, bool edgeLabels, bool classic) {
		std::ostringstream oss{};
		// header, root and terminal declaration

		if (colored) {
			coloredHeader(e, oss, edgeLabels);
		} else {
			header(e, oss, edgeLabels);
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

			// node definition as HTML-like label (href="javascript:;" is used as workaround to make tooltips work)
			if (isVector) {
				if (classic)
					classicVectorNode(*node, oss);
				else
					vectorNode(*node, oss);
			} else {
				if (classic)
					classicMatrixNode(*node, oss);
				else
					matrixNodeMiddleVar(*node, oss);
			}

			// iterate over edges in reverse to guarantee correct proceossing order
			for (short i=dd::NEDGE-1; i >= 0; --i) {
				if (isVector && i%2 != 0)
					continue;

				auto& edge = node->p->e[i];
				if (CN::equalsZero(edge.w)) {
					if (classic) {
						// potentially add zero stubs here
//						auto nodelabel = ((uintptr_t)node->p & 0x001fffffu) >> 1u; // this allows for 2^20 (roughly 1e6) unique nodes
//						oss << nodelabel << "0" << i << "[label=<<font point-size=\"6\">0</font>>]\n";
//						oss << nodelabel << ":" << i << "->" << nodelabel << "0" << i << ":n\n";
						continue;
					} else {
						continue;
					}
				}

				// non-zero edge to be included
				q.push(&edge);

				if (isVector) {
					if (colored) {
						coloredVectorEdge(*node, edge, i, oss, edgeLabels, classic);
					} else {
						vectorEdge(*node, edge, i, oss, edgeLabels, classic);
					}
				} else {
					if (colored) {
						coloredMatrixEdge(*node, edge, i, oss, edgeLabels, classic);
					} else {
						matrixEdge(*node, edge, i, oss, edgeLabels, classic);
					}
				}
			}
		}
		oss << "}\n";

		os << oss.str() << std::flush;
	}

	void export2Dot(Edge basic, const std::string& outputFilename, bool isVector, bool colored, bool edgeLabels, bool classic, bool show) {
		std::ofstream init(outputFilename);
		toDot(basic, init, isVector, colored, edgeLabels, classic);
		init.close();

		if (show) {
			std::ostringstream oss;
			oss << "dot -Tsvg " << outputFilename << " -o " << outputFilename.substr(0, outputFilename.find_last_of('.')) << ".svg";
			auto str = oss.str(); // required to avoid immediate deallocation of temporary
			static_cast<void>(!std::system(str.c_str())); // cast and ! just to suppress the unused result warning
		}
	}

	RGB hlsToRGB(const fp& h, const fp& l, const fp& s) {
		if (s == 0.0) {
			return {l, l, l};
		}
		fp m2;
		if (l <= 0.5) {
			m2 = l * (1+s);
		} else {
			m2 = l+s-(l*s);
		}
		auto m1 = 2*l - m2;

		auto v = [] (const fp& m1, const fp& m2, fp hue) -> fp {
			while (hue < 0) hue += 1.0;
			while (hue > 1) hue -= 1.0;
			if (hue < 1./6)
				return m1 + (m2-m1)*hue*6.0;
			if (hue < 0.5)
				return m2;
			if (hue < 2./3)
				return m1 + (m2-m1)*(2./3-hue)*6.0;
			return m1;
		};

		return {v(m1, m2, h+1./3), v(m1, m2, h), v(m1, m2, h-1./3)};
	}

	RGB colorFromPhase(const Complex& a) {
		auto phase = CN::arg(a);
		auto twopi = 2*PI;
		phase = (phase) / twopi;
		return hlsToRGB(phase, 0.5, 0.5);
	}

	fp thicknessFromMagnitude (const Complex& a) {
		return 3.0*std::max(CN::mag(a), 0.10);
	}

}
