#ifndef ZX_INCLUDE_RULES_HPP_
#define ZX_INCLUDE_RULES_HPP_

#include "ZXDiagram.hpp"

namespace zx {
bool checkIdSimp(ZXDiagram &diag, Vertex v);

void removeId(ZXDiagram &diag, Vertex v);

bool checkSpiderFusion(ZXDiagram &diag, Vertex v0, Vertex v1);

void fuseSpiders(ZXDiagram &diag, Vertex v0, Vertex v1);

bool checkLocalComp(ZXDiagram &diag, Vertex v);

void localComp(ZXDiagram &diag, Vertex v);

bool checkPivotPauli(ZXDiagram &diag, Vertex v0, Vertex v1);

void pivotPauli(ZXDiagram &diag, Vertex v0, Vertex v1);

bool checkPivot(ZXDiagram &diag, Vertex v0, Vertex v1);

void pivot(ZXDiagram &diag, Vertex v0, Vertex v1);

bool checkPivotGadget(ZXDiagram &diag, Vertex v0, Vertex v1);

void pivotGadget(ZXDiagram &diag, Vertex v0, Vertex v1);

bool checkAndFuseGadget(ZXDiagram &diag, Vertex v);
} // namespace zx

#endif /* JKQZX_INCLUDE_RULES_HPP_ */
