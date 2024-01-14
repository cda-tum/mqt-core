#pragma once

#include "mqt_core_export.h"
#include "zx/ZXDefinitions.hpp"

namespace zx {
class ZXDiagram;

MQT_CORE_EXPORT bool checkIdSimp(const ZXDiagram& diag, Vertex v);

MQT_CORE_EXPORT void removeId(ZXDiagram& diag, Vertex v);

MQT_CORE_EXPORT bool checkSpiderFusion(const ZXDiagram& diag, Vertex v0,
                                       Vertex v1);

MQT_CORE_EXPORT void fuseSpiders(ZXDiagram& diag, Vertex v0, Vertex v1);

MQT_CORE_EXPORT bool checkLocalComp(const ZXDiagram& diag, Vertex v);

MQT_CORE_EXPORT void localComp(ZXDiagram& diag, Vertex v);

MQT_CORE_EXPORT bool checkPivotPauli(const ZXDiagram& diag, Vertex v0,
                                     Vertex v1);

MQT_CORE_EXPORT void pivotPauli(ZXDiagram& diag, Vertex v0, Vertex v1);

MQT_CORE_EXPORT bool checkPivot(const ZXDiagram& diag, Vertex v0, Vertex v1);

MQT_CORE_EXPORT void pivot(ZXDiagram& diag, Vertex v0, Vertex v1);

MQT_CORE_EXPORT bool checkPivotGadget(const ZXDiagram& diag, Vertex v0,
                                      Vertex v1);

MQT_CORE_EXPORT void pivotGadget(ZXDiagram& diag, Vertex v0, Vertex v1);

MQT_CORE_EXPORT bool checkAndFuseGadget(ZXDiagram& diag, Vertex v);
} // namespace zx
