/*
 * This file is part of IIC-JKU DD package which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum_dd/ for more information.
 */

#include <stack>
#include <regex>
#include "DDexport.h"

namespace dd {
	std::ostream& header(const Edge& e, std::ostream& os, bool edgeLabels) {
		os << "digraph \"DD\" {graph[];node[shape=plain];edge[arrowhead=none]\n";
		os << "root [label=\"\",shape=point,style=invis]\n";
		os << "t [label=<<font point-size=\"20\">1</font>>,shape=box,tooltip=\"1\",width=0.3,height=0.3]\n";
		auto toplabel = (reinterpret_cast<uintptr_t>(e.p) & 0x001fffffU) >> 1U;
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

		auto toplabel = (reinterpret_cast<uintptr_t>(e.p) & 0x001fffffU) >> 1U;
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
		auto nodelabel = (reinterpret_cast<uintptr_t>(e.p) & 0x001fffffU) >> 1U; // this allows for 2^20 (roughly 1e6) unique nodes
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
		auto nodelabel = (reinterpret_cast<uintptr_t>(e.p) & 0x001fffffU) >> 1U; // this allows for 2^20 (roughly 1e6) unique nodes
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
		auto nodelabel = (reinterpret_cast<uintptr_t>(e.p) & 0x001fffffU) >> 1U; // this allows for 2^20 (roughly 1e6) unique nodes
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
		auto nodelabel = (reinterpret_cast<uintptr_t>(e.p) & 0x001fffffU) >> 1U; // this allows for 2^20 (roughly 1e6) unique nodes
		os << nodelabel << "[label=<";
		os << R"(<font point-size="8"><table border="1" cellspacing="0" cellpadding="0" style="rounded">)";
		os << R"(<tr><td colspan="2" border="0" cellpadding="1"><font point-size="20">q<sub><font point-size="12">)" << e.p->v << R"(</font></sub></font></td></tr><tr>)";
		os << R"(<td height="6" width="14" port="0" tooltip=")" << e.p->e[0].w << R"(" href="javascript:;" sides="RT">)" << (CN::equalsZero(e.p->e[0].w) ? "&nbsp;0 " : R"(<font color="white">&nbsp;0 </font>)") << "</td>";
		os << R"(<td height="6" width="14" port="2" tooltip=")" << e.p->e[2].w << R"(" href="javascript:;" sides="LT">)" << (CN::equalsZero(e.p->e[2].w) ? "&nbsp;0 " : R"(<font color="white">&nbsp;0 </font>)") << "</td>";
		os << "</tr></table></font>>,tooltip=\"q" << e.p->v << "\"]\n";
		return os;
	}

	std::ostream& vectorNodeVectorLook(const Edge& e, std::ostream& os) {
		auto nodelabel = (reinterpret_cast<uintptr_t>(e.p) & 0x001fffffU) >> 1U; // this allows for 2^20 (roughly 1e6) unique nodes
		os << nodelabel << "[label=<";
		os << R"(<font point-size="10"><table border="1" cellspacing="0" cellpadding="2" style="rounded">)";
		os << R"(<tr><td rowspan="2" sides="R" cellpadding="2"><font point-size="18">q<sub><font point-size="12">)" << e.p->v << "</font></sub></font></td>";
		os << R"(<td port="0" tooltip=")" << e.p->e[0].w << R"(" href="javascript:;" sides="LB">)" << (CN::equalsZero(e.p->e[0].w) ? "&nbsp;0 " : R"(<font color="white">&nbsp;0 </font>)") << "</td></tr><tr>";
		os << R"(<td port="2" tooltip=")" << e.p->e[2].w << R"(" href="javascript:;" sides="LT">)" << (CN::equalsZero(e.p->e[2].w) ? "&nbsp;0 " : R"(<font color="white">&nbsp;0 </font>)") << "</td>";
		os << "</tr></table></font>>,tooltip=\"q" << e.p->v << "\"]\n";
		return os;
	}

	std::ostream& classicVectorNode(const Edge& e, std::ostream& os) {
		auto nodelabel = (reinterpret_cast<uintptr_t>(e.p) & 0x001fffffU) >> 1U; // this allows for 2^20 (roughly 1e6) unique nodes
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
		auto fromlabel = (reinterpret_cast<uintptr_t>(from.p) & 0x001fffffU) >> 1U;
		auto tolabel = (reinterpret_cast<uintptr_t>(to.p) & 0x001fffffU) >> 1U;

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
		auto fromlabel = (reinterpret_cast<uintptr_t>(from.p) & 0x001fffffU) >> 1U;
		auto tolabel = (reinterpret_cast<uintptr_t>(to.p) & 0x001fffffU) >> 1U;

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

	std::ostream& vectorEdge(const Edge& from, const Edge& to, short idx, std::ostream& os, bool edgeLabels) {
		auto fromlabel = (reinterpret_cast<uintptr_t>(from.p) & 0x001fffffU) >> 1U;
		auto tolabel = (reinterpret_cast<uintptr_t>(to.p) & 0x001fffffU) >> 1U;

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

	std::ostream& coloredVectorEdge(const Edge& from, const Edge& to, short idx, std::ostream& os, bool edgeLabels) {
		auto fromlabel = (reinterpret_cast<uintptr_t>(from.p) & 0x001fffffU) >> 1U;
		auto tolabel = (reinterpret_cast<uintptr_t>(to.p) & 0x001fffffU) >> 1U;

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
					// potentially add zero stubs here
					continue;
				}

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

	void serialize(const Edge& basic, const std::string& outputFilename, bool isVector, bool writeBinary) {
		std::ofstream ofs(outputFilename);

		if(!ofs.good()) {
			std::cerr << "Cannot open file: " << outputFilename << std::endl;
			return;
		}

		serialize(basic, ofs, isVector, writeBinary);
	}

	void serialize(const Edge& basic, std::ostream& os, bool isVector, bool writeBinary) {
		if(writeBinary) {
			os.write(reinterpret_cast<const char*>(&SERIALIZATION_VERSION), sizeof(fp));
			writeBinaryAmplitude(os, basic.w);
		} else {
			os << SERIALIZATION_VERSION << "\n";
			os << CN::toString(basic.w, false, 16) << "\n";
		}

		if (isVector) {
			serializeVector(basic, os, writeBinary);
		} else {
			auto idx = 0;
			std::unordered_map<NodePtr, int> node_index{};
			std::unordered_set<NodePtr> visited{};
			serializeMatrix(basic, idx, node_index, visited, os, writeBinary);
		}
	}

	void writeBinaryAmplitude(std::ostream& os, const Complex& w) {
		fp temp = CN::val(w.r);
		os.write(reinterpret_cast<const char*>(&temp), sizeof(fp));
		temp = CN::val(w.i);
		os.write(reinterpret_cast<const char*>(&temp), sizeof(fp));
	}

	void writeBinaryAmplitude(std::ostream& os, const ComplexValue& w) {
		os.write(reinterpret_cast<const char*>(&w.r), sizeof(fp));
		os.write(reinterpret_cast<const char*>(&w.i), sizeof(fp));
	}

	void serializeVector(const Edge& basic, std::ostream& os, bool writeBinary) {
		int next_index = 0;
		std::unordered_map<NodePtr, int> node_index{};

		// POST ORDER TRAVERSAL USING ONE STACK   https://www.geeksforgeeks.org/iterative-postorder-traversal-using-stack/
		std::stack<const Edge*> stack{};

		const Edge *node = &basic;
		if(!Package::isTerminal(*node)) {
			do {
				while(node != nullptr && !Package::isTerminal(*node)) {
					for (short i=NEDGE-1; i > 0; --i) {
						if (i % 2 != 0) continue;
						auto& edge = node->p->e[i];
						if (Package::isTerminal(edge)) continue;
						if (CN::equalsZero(edge.w)) continue;
						if(node_index.find(edge.p) != node_index.end()) continue;

						// non-zero edge to be included
						stack.push(&edge);
					}
					stack.push(node);
					node = &node->p->e[0];
				}
				node = stack.top();
				stack.pop();

				bool hasChild = false;
				for (short i = 1; i < NEDGE && !hasChild; ++i) {
					if (i % 2 != 0) continue;
					auto& edge = node->p->e[i];
					if (CN::equalsZero(edge.w)) continue;
					if(node_index.find(edge.p) != node_index.end())	continue;
					if (!stack.empty())
						hasChild = edge.p == stack.top()->p;
				}

				if(hasChild) {
					const Edge* temp = stack.top();
					stack.pop();
					stack.push(node);
					node = temp;
				} else {
					if(node_index.find(node->p) != node_index.end()) {
						node = nullptr;
						continue;
					}
					node_index[node->p] = next_index;
					next_index++;

					if(writeBinary) {
						os.write(reinterpret_cast<const char*>(&node_index[node->p]), sizeof(int));
						os.write(reinterpret_cast<const char*>(&node->p->v), sizeof(short));

						// iterate over edges in reverse to guarantee correct processing order
						for (short i = 0; i < NEDGE; i += 2) {
							auto& edge = node->p->e[i];
							int edge_idx = Package::isTerminal(edge) ? -1 : node_index[edge.p];
							os.write(reinterpret_cast<const char*>(&edge_idx), sizeof(int));
							writeBinaryAmplitude(os, edge.w);
						}
					} else {
						os << node_index[node->p] << " " << node->p->v;

						// iterate over edges in reverse to guarantee correct processing order
						for (short i = 0; i < NEDGE; ++i) {
							os << " (";
							if (i % 2 == 0) {
								auto& edge = node->p->e[i];
								if (!CN::equalsZero(edge.w)) {
									int edge_idx = Package::isTerminal(edge) ? -1 : node_index[edge.p];
									os << edge_idx << " " << CN::toString(edge.w, false, 16);
								}
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

	void serializeMatrix(const Edge& basic, int& idx, std::unordered_map<NodePtr, int>& node_index, std::unordered_set<NodePtr>& visited, std::ostream& os, bool writeBinary) {
		if (!dd::Package::isTerminal(basic)) {
			for (auto& e : basic.p->e) {
				if (auto[iter, success] = visited.insert(e.p); success) {
					serializeMatrix(e, idx, node_index, visited, os, writeBinary);
				}
			}


			if (node_index.find(basic.p) == node_index.end()) {
				node_index[basic.p] = idx;
				++idx;
			}

			if (writeBinary) {
				os.write(reinterpret_cast<const char *>(&node_index[basic.p]), sizeof(int));
				os.write(reinterpret_cast<const char *>(&basic.p->v), sizeof(short));

				// iterate over edges in reverse to guarantee correct processing order
				for (auto& edge : basic.p->e) {
					int edge_idx = Package::isTerminal(edge) ? -1 : node_index[edge.p];
					os.write(reinterpret_cast<const char *>(&edge_idx), sizeof(int));
					writeBinaryAmplitude(os, edge.w);
				}
			} else {
				os << node_index[basic.p] << " " << basic.p->v;

				// iterate over edges in reverse to guarantee correct processing order
				for (auto& edge : basic.p->e) {
					os << " (";
					if (!CN::equalsZero(edge.w)) {
						int edge_idx = Package::isTerminal(edge) ? -1 : node_index[edge.p];
						os << edge_idx << " " << CN::toString(edge.w, false, 16);
					}
					os << ")";
				}
				os << "\n";
			}
		}
	}

	Edge create_deserialized_node(std::unique_ptr<Package>& dd, int index, short v, std::array<int, NEDGE>& edge_idx,
	                              std::array<ComplexValue, NEDGE>& edge_weight, std::unordered_map<int, NodePtr>& nodes) {
		if(index == -1) {
			return Package::DDzero;
		}

		std::array<Edge, NEDGE> edges{};
		for(int i = 0; i < NEDGE; i++) {
			if(edge_idx[i] == -2) {
				edges[i] = Package::DDzero;
			} else {
				edges[i].p = edge_idx[i] == -1 ? Package::DDone.p : nodes[edge_idx[i]];
				edges[i].w = dd->cn.lookup(edge_weight[i]);
			}
		}

		Edge newedge = dd->makeNonterminal(v, edges);
		nodes[index] = newedge.p;

		// reset
		edge_idx.fill(-2);

		return newedge;
	}

	ComplexValue toComplexValue(const std::string& real_str, std::string imag_str) {
		fp real = real_str.empty() ? 0. : std::stod(real_str);

		imag_str.erase(remove(imag_str.begin(), imag_str.end(), ' '), imag_str.end());
		imag_str.erase(remove(imag_str.begin(), imag_str.end(), 'i'), imag_str.end());
		if(imag_str == "+" || imag_str == "-") imag_str = imag_str + "1";
		fp imag = imag_str.empty() ? 0. : std::stod(imag_str);
		return ComplexValue{real, imag};
	}

	dd::Edge deserialize(std::unique_ptr<dd::Package>& dd, const std::string& inputFilename, bool isVector, bool readBinary) {
		auto ifs = std::ifstream(inputFilename);

		if(!ifs.good()) {
			std::cerr << "Cannot open serialized file: " << inputFilename << std::endl;
			return dd::Package::DDzero;
		}

		return deserialize(dd, ifs, isVector, readBinary);
	}

	ComplexValue readBinaryAmplitude(std::istream& is) {
		ComplexValue temp{};
		is.read(reinterpret_cast<char*>(&temp.r), sizeof(fp));
		is.read(reinterpret_cast<char*>(&temp.i), sizeof(fp));
		return temp;
	}

	dd::Edge deserialize(std::unique_ptr<dd::Package>& dd, std::istream& is, bool isVector, bool readBinary) {
		Edge result = Package::DDzero;
		ComplexValue rootweight{};

		std::unordered_map<int, NodePtr> nodes;
		int node_index;
		short v;
		std::array<int, NEDGE> edge_indices{};
		edge_indices.fill(-2);
		std::array<ComplexValue, NEDGE> edge_weights{};

		if(readBinary) {
			double version;
			is.read(reinterpret_cast<char*>(&version), sizeof(double));
			if(version != SERIALIZATION_VERSION) {
				std::cerr << "Wrong Version of serialization file version: " << version << std::endl;
				exit(1);
			}

			if(!is.eof()) {
				rootweight = readBinaryAmplitude(is);
			}


			while (is.read(reinterpret_cast<char*>(&node_index), sizeof(int))) {
				is.read(reinterpret_cast<char*>(&v), sizeof(short));
				for(int i = 0; i < NEDGE; i += isVector ? 2 : 1) {
					is.read(reinterpret_cast<char*>(&edge_indices[i]), sizeof(int));
					edge_weights[i] = readBinaryAmplitude(is);
				}
				result = create_deserialized_node(dd, node_index, v, edge_indices, edge_weights, nodes);
			}
		} else {
			std::string version;
			std::getline(is, version);
			// ifs >> version;
			if(std::stod(version) != SERIALIZATION_VERSION) {
				std::cerr << "Wrong Version of serialization file. version of file: " << version << "; current version: " << SERIALIZATION_VERSION << std::endl;
				exit(1);
			}

			std::string line;
			std::string complex_real_regex = R"(([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?(?![ \d\.]*(?:[eE][+-])?\d*[iI]))?)";
			std::string complex_imag_regex = R"(( ?[+-]? ?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)?[iI])?)";
			std::string edge_regex = " \\(((-?\\d+) (" + complex_real_regex + complex_imag_regex + "))?\\)";
			std::regex complex_weight_regex (complex_real_regex + complex_imag_regex);
			std::regex line_regex ("(\\d+) (\\d+)(?:" + edge_regex + ")(?:" + edge_regex + ")(?:" + edge_regex + ")(?:" + edge_regex + ") *(?:#.*)?");
			// std::regex e ("(\\d+) (\\d+)(?:" + edge_regex + "){4} *#.*"); // TODO {4} overwrites groups
			std::smatch m;

			if(std::getline(is, line)) {
				if(!std::regex_match(line, m, complex_weight_regex)) {
					std::cerr << "Regex did not match second line: " << line << std::endl;
					exit(1);
				}
				rootweight = toComplexValue(m.str(1), m.str(2));
			}
			// std::cout << "rootweight = " << rootweight << std::endl;

			while (std::getline(is, line)) {
				if (line.empty() || line.size() == 1) continue;

				if(!std::regex_match(line, m, line_regex)) {
					std::cerr << "Regex did not match line: " << line << std::endl;
					exit(1);
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
				v          = static_cast<short>(std::stoi(m.str(2)));

				// std::cout << "nidx: " << node_index << " v: " << v << std::endl;

				for(int edge_idx = 3, i = 0; i < NEDGE; i++, edge_idx += 5) {
					if(m.str(edge_idx).empty()) {
						// std::cout << "index " << i << " is empty " << std::endl;
						continue;
					}

					edge_indices[i] = std::stoi(m.str(edge_idx + 1));
					edge_weights[i] = toComplexValue(m.str(edge_idx + 3), m.str(edge_idx + 4));
				}

				result = create_deserialized_node(dd, node_index, v, edge_indices, edge_weights, nodes);
			}
		}

		Complex w = dd->cn.getCachedComplex(rootweight.r, rootweight.i);
		CN::mul(w, result.w, w);
		result.w = dd->cn.lookup(w);
		dd->cn.releaseCached(w);

		return result;
	}
}
