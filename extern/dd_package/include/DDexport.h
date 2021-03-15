/*
 * This file is part of IIC-JKU DD package which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum_dd/ for more information.
 */

#ifndef DDexport_H
#define DDexport_H

#include <iomanip>
#include <unordered_map>
#include "DDpackage.h"

namespace dd {
	struct RGB {
		fp R=0., G=0., B=0.;
		RGB(fp R, fp G, fp B): R(R), G(G), B(B) {};

		std::ostream& printHex(std::ostream& os) const {
			std::ostringstream oss{};
			oss.flags(std::ios_base::hex);
			oss.fill('0');
			oss << std::setw(2) << static_cast<short>(R*255.) << std::setw(2) << static_cast<short>(G*255.) << std::setw(2) << static_cast<short>(B*255.);
			os << oss.str();
			return os;
		}

		friend std::ostream& operator<<(std::ostream& os, const RGB& rgb) {
			return rgb.printHex(os);
		}
	};

	fp hueToRGB(fp hue);
	RGB hlsToRGB(const fp& h, const fp& l, const fp& s);

	RGB colorFromPhase(const Complex& a);
	fp thicknessFromMagnitude (const Complex& a);

	std::ostream& header(const Edge& e, std::ostream& os, bool edgeLabels);
	std::ostream& coloredHeader(const Edge& e, std::ostream& os, bool edgeLabels);

	std::ostream& matrixNodeMatrixAndXlabel(const Edge& e, std::ostream& os);
	std::ostream& matrixNodeMiddleVar(const Edge& e, std::ostream& os);
	std::ostream& classicMatrixNode(const Edge& e, std::ostream& os);

	std::ostream& vectorNode(const Edge& e, std::ostream& os);
	std::ostream& vectorNodeVectorLook(const Edge& e, std::ostream& os);
	std::ostream& classicVectorNode(const Edge& e, std::ostream& os);

	std::ostream& matrixEdge(const Edge& from, const Edge& to, short idx, std::ostream& os, bool edgeLabels=false, bool classic=false);
	std::ostream& coloredMatrixEdge(const Edge& from, const Edge& to, short idx, std::ostream& os, bool edgeLabels=false, bool classic=false);

	std::ostream& vectorEdge(const Edge& from, const Edge& to, short idx, std::ostream& os, bool edgeLabels = false);
	std::ostream& coloredVectorEdge(const Edge& from, const Edge& to, short idx, std::ostream& os, bool edgeLabels = false);

	void toDot(const Edge& e, std::ostream& os, bool isVector = false, bool colored=true, bool edgeLabels=false, bool classic=false);
	void export2Dot(Edge basic, const std::string& outputFilename, bool isVector = false, bool colored=true, bool edgeLabels=false, bool classic=false, bool show = true);

	ComplexValue toComplexValue(const std::string& real_str, std::string imag_str);
	ComplexValue readBinaryAmplitude(std::istream& is);
	void writeBinaryAmplitude(std::ostream& os, const Complex& w);
	void writeBinaryAmplitude(std::ostream& os, const ComplexValue& w);

	void serialize(const Edge& basic, std::ostream& os, bool isVector = false, bool writeBinary = false);
	void serialize(const Edge& basic, const std::string& outputFilename, bool isVector = false, bool writeBinary = false);
	void serializeVector(const Edge& basic, std::ostream& os, bool writeBinary = false);
	void serializeMatrix(const Edge& basic, int& idx, std::unordered_map<NodePtr, int>& node_index, std::unordered_set<NodePtr>& visited, std::ostream& os, bool writeBinary = false);
	Edge create_deserialized_node(std::unique_ptr<Package>& dd, int index, short v, std::array<int, NEDGE>& edge_idx,
	                              std::array<ComplexValue, NEDGE>& edge_weight, std::unordered_map<int, NodePtr>& nodes);
	// isVector only used if readBinary is true
	dd::Edge deserialize(std::unique_ptr<dd::Package>& dd, const std::string& inputFilename, bool isVector = false, bool readBinary = false);
	dd::Edge deserialize(std::unique_ptr<dd::Package>& dd, std::istream& is, bool isVector = false, bool readBinary = false);
}


#endif //DDexport_H
