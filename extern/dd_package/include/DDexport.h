/*
 * This file is part of IIC-JKU DD package which is released under the MIT license.
 * See file README.md or go to http://iic.jku.at/eda/research/quantum_dd/ for more information.
 */

#ifndef DDexport_H
#define DDexport_H

#include <iomanip>
#include "DDpackage.h"

namespace dd {
	struct RGB {
		fp R=0., G=0., B=0.;
		RGB(fp R, fp G, fp B): R(R), G(G), B(B) {};

		std::ostream& printHex(std::ostream& os) const {
			std::ostringstream oss{};
			oss.flags(std::ios_base::hex);
			oss.fill('0');
			oss << std::setw(2) << short(R*255) << std::setw(2) << short(G*255) << std::setw(2) << short(B*255);
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

	std::ostream& vectorEdge(const Edge& from, const Edge& to, short idx, std::ostream& os, bool edgeLabels=false, bool classic=false);
	std::ostream& coloredVectorEdge(const Edge& from, const Edge& to, short idx, std::ostream& os, bool edgeLabels=false, bool classic=false);

	void toDot(const Edge& e, std::ostream& os, bool isVector = false, bool colored=true, bool edgeLabels=false, bool classic=false);
	void export2Dot(Edge basic, const std::string& outputFilename, bool isVector = false, bool colored=true, bool edgeLabels=false, bool classic=false, bool show = true);
}


#endif //DDexport_H
