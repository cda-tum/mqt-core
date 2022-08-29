#pragma once

#include "Definitions.hpp"

namespace zx {
    class ZXDiagram;

    bool checkIdSimp(ZXDiagram& diag, Vertex v);

    void removeId(ZXDiagram& diag, Vertex v);

    bool checkSpiderFusion(ZXDiagram& diag, Vertex v0, Vertex v1);

    void fuseSpiders(ZXDiagram& diag, Vertex v0, Vertex v1);

    bool checkLocalComp(ZXDiagram& diag, Vertex v);

    void localComp(ZXDiagram& diag, Vertex v);

    bool checkPivotPauli(ZXDiagram& diag, Vertex v0, Vertex v1);

    void pivotPauli(ZXDiagram& diag, Vertex v0, Vertex v1);

    bool checkPivot(ZXDiagram& diag, Vertex v0, Vertex v1);

    void pivot(ZXDiagram& diag, Vertex v0, Vertex v1);

    bool checkPivotGadget(ZXDiagram& diag, Vertex v0, Vertex v1);

    void pivotGadget(ZXDiagram& diag, Vertex v0, Vertex v1);

    bool checkAndFuseGadget(ZXDiagram& diag, Vertex v);
} // namespace zx
