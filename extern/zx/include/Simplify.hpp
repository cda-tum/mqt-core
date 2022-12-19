#pragma once

#include "Definitions.hpp"
#include "Rules.hpp"
#include "ZXDiagram.hpp"

#include <cstddef>

namespace zx {

template <class VertexCheckFun, class VertexRuleFun>
std::size_t simplifyVertices(ZXDiagram& diag, VertexCheckFun check,
                             VertexRuleFun rule) {
  std::size_t nSimplifications = 0;
  bool        newMatches       = true;

  while (newMatches) {
    newMatches = false;
    for (const auto [v, _] : diag.getVertices()) {
      if (check(diag, v)) {
        rule(diag, v);
        newMatches = true;
        nSimplifications++;
      }
    }
  }

  return nSimplifications;
}

template <class EdgeCheckFun, class EdgeRuleFun>
std::size_t simplifyEdges(ZXDiagram& diag, EdgeCheckFun check,
                          EdgeRuleFun rule) {
  std::size_t nSimplifications = 0;
  bool        newMatches       = true;

  while (newMatches) {
    newMatches = false;
    for (const auto& [v0, v1] : diag.getEdges()) {
      if (diag.isDeleted(v0) || diag.isDeleted(v1) || !check(diag, v0, v1)) {
        continue;
      }
      rule(diag, v0, v1);
      newMatches = true;
      nSimplifications++;
    }
  }

  return nSimplifications;
}

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
