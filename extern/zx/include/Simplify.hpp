#pragma once

#include "Definitions.hpp"
#include "Rules.hpp"

#include <cstddef>

namespace zx {
    class ZXDiagram;

    using VertexCheckFun = decltype(checkIdSimp);
    using VertexRuleFun  = decltype(removeId);
    using EdgeCheckFun   = decltype(checkSpiderFusion);
    using EdgeRuleFun    = decltype(fuseSpiders);

    std::size_t simplifyVertices(ZXDiagram& diag, VertexCheckFun check,
                                 VertexRuleFun rule);

    std::size_t simplifyEdges(ZXDiagram& diag, EdgeCheckFun check,
                              EdgeRuleFun rule);

    std::size_t gadgetSimp(ZXDiagram& diag);

    std::size_t idSimp(ZXDiagram& diag);

    std::size_t spiderSimp(ZXDiagram& diag);

    std::size_t localCompSimp(ZXDiagram& diag);

    std::size_t pivotPauliSimp(ZXDiagram& diag);

    std::size_t pivotSimp(ZXDiagram& diag);

    std::size_t interiorCliffordSimp(ZXDiagram& diag);

    std::size_t cliffordSimp(ZXDiagram& diag);

    std::size_t pivotgadgetSimp(ZXDiagram& diag);

    std::size_t fullReduce(ZXDiagram& diag);
    std::size_t fullReduceApproximate(ZXDiagram& diag, fp tolerance);

} // namespace zx
