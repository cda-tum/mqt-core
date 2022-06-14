#ifndef ZX_INCLUDE_SIMPLIFY_HPP_
#define ZX_INCLUDE_SIMPLIFY_HPP_

#include "Rules.hpp"
#include "ZXDiagram.hpp"

namespace zx {
using VertexCheckFun = decltype(checkIdSimp);
using VertexRuleFun = decltype(removeId);
using EdgeCheckFun = decltype(checkSpiderFusion);
using EdgeRuleFun = decltype(fuseSpiders);

int32_t simplifyVertices(ZXDiagram &diag, VertexCheckFun check,
                          VertexRuleFun rule);

int32_t simplifyEdges(ZXDiagram &diag, EdgeCheckFun check, EdgeRuleFun rule);

int32_t gadgetSimp(ZXDiagram &diag);

int32_t idSimp(ZXDiagram &diag);

int32_t spiderSimp(ZXDiagram &diag);

int32_t localCompSimp(ZXDiagram &diag);

int32_t pivotPauliSimp(ZXDiagram &diag);

int32_t pivotSimp(ZXDiagram &diag);

int32_t interiorCliffordSimp(ZXDiagram &diag);

int32_t cliffordSimp(ZXDiagram &diag);

int32_t pivotgadgetSimp(ZXDiagram &diag);

int32_t fullReduce(ZXDiagram &diag);

} // namespace zx

#endif /* JKQZX_INCLUDE_SIMPLIFY_HPP_ */
