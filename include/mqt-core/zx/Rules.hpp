#pragma once

#include "zx/ZXDefinitions.hpp"

namespace zx {
class ZXDiagram;

bool checkIdSimp(const ZXDiagram& diag, Vertex v);

void removeId(ZXDiagram& diag, Vertex v);

bool checkSpiderFusion(const ZXDiagram& diag, Vertex v0, Vertex v1);

void fuseSpiders(ZXDiagram& diag, Vertex v0, Vertex v1);

bool checkLocalComp(const ZXDiagram& diag, Vertex v);

void localComp(ZXDiagram& diag, Vertex v);

bool checkPivotPauli(const ZXDiagram& diag, Vertex v0, Vertex v1);

void pivotPauli(ZXDiagram& diag, Vertex v0, Vertex v1);

bool checkPivot(const ZXDiagram& diag, Vertex v0, Vertex v1);

void pivot(ZXDiagram& diag, Vertex v0, Vertex v1);

bool checkPivotGadget(const ZXDiagram& diag, Vertex v0, Vertex v1);

void pivotGadget(ZXDiagram& diag, Vertex v0, Vertex v1);

bool checkAndFuseGadget(ZXDiagram& diag, Vertex v);
} // namespace zx
