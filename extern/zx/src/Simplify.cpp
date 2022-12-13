#include "Simplify.hpp"

#include "Definitions.hpp"
#include "Rules.hpp"
#include "ZXDiagram.hpp"

#include <utility>

namespace zx {

std::size_t gadgetSimp(ZXDiagram& diag) {
  std::size_t nSimplifications = 0;
  bool        newMatches       = true;

  while (newMatches) {
    newMatches = false;
    for (auto [v, _] : diag.getVertices()) {
      if (diag.isDeleted(v)) {
        continue;
      }

      if (checkAndFuseGadget(diag, v)) {
        newMatches = true;
        nSimplifications++;
      }
    }
  }
  return nSimplifications;
}

std::size_t idSimp(ZXDiagram& diag) {
  return simplifyVertices(diag, checkIdSimp, removeId);
}

std::size_t spiderSimp(ZXDiagram& diag) {
  return simplifyEdges(diag, checkSpiderFusion, fuseSpiders);
}

std::size_t localCompSimp(ZXDiagram& diag) {
  return simplifyVertices(diag, checkLocalComp, localComp);
}

std::size_t pivotPauliSimp(ZXDiagram& diag) {
  return simplifyEdges(diag, checkPivotPauli, pivotPauli);
}

std::size_t pivotSimp(ZXDiagram& diag) {
  return simplifyEdges(diag, checkPivot, pivot);
}

std::size_t interiorCliffordSimp(ZXDiagram& diag) {
  spiderSimp(diag);

  bool        newMatches       = true;
  std::size_t nSimplifications = 0;
  while (newMatches) {
    newMatches            = false;
    const auto nId        = idSimp(diag);
    const auto nSpider    = spiderSimp(diag);
    const auto nPivot     = pivotPauliSimp(diag);
    const auto nLocalComp = localCompSimp(diag);

    if ((nId + nSpider + nPivot + nLocalComp) != 0) {
      newMatches = true;
      nSimplifications++;
    }
  }
  return nSimplifications;
}

std::size_t cliffordSimp(ZXDiagram& diag) {
  bool        newMatches       = true;
  std::size_t nSimplifications = 0;
  while (newMatches) {
    newMatches           = false;
    const auto nClifford = interiorCliffordSimp(diag);
    const auto nPivot    = pivotSimp(diag);
    if ((nClifford + nPivot) != 0) {
      newMatches = true;
      nSimplifications++;
    }
  }
  return nSimplifications;
}

std::size_t pivotgadgetSimp(ZXDiagram& diag) {
  return simplifyEdges(diag, checkPivotGadget, pivotGadget);
}

std::size_t fullReduce(ZXDiagram& diag) {
  diag.toGraphlike();
  interiorCliffordSimp(diag);

  std::size_t nSimplifications = 0;
  while (true) {
    cliffordSimp(diag);
    const auto nGadget = gadgetSimp(diag);
    interiorCliffordSimp(diag);
    const auto nPivot = pivotgadgetSimp(diag);
    if ((nGadget + nPivot) == 0) {
      break;
    }
    nSimplifications += nGadget + nPivot;
  }
  diag.removeDisconnectedSpiders();

  return nSimplifications;
}

std::size_t fullReduceApproximate(ZXDiagram& diag, const fp tolerance) {
  auto        nSimplifications = fullReduce(diag);
  std::size_t newSimps         = 0;
  do {
    diag.approximateCliffords(tolerance);
    newSimps = fullReduce(diag);
    nSimplifications += newSimps;
  } while (newSimps != 0U);
  return nSimplifications;
}
} // namespace zx
