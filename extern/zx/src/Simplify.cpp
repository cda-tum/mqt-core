#include "Simplify.hpp"
#include "Definitions.hpp"
#include "Rules.hpp"
#include <iostream>

namespace zx {
int32_t simplifyVertices(ZXDiagram &diag, VertexCheckFun check,
                          VertexRuleFun rule) {
  int32_t n_simplifications = 0;
  bool new_matches = true;

  while (new_matches) {
    new_matches = false;
    for (auto [v, _] : diag.getVertices()) {
      if (check(diag, v)) {
        rule(diag, v);
        new_matches = true;
        n_simplifications++;
      }
    }
  }

  return n_simplifications;
}

int32_t simplifyEdges(ZXDiagram &diag, EdgeCheckFun check, EdgeRuleFun rule) {
  int32_t n_simplifications = 0;
  bool new_matches = true;

  while (new_matches) {
    new_matches = false;
    for (auto [v0, v1] : diag.getEdges()) {
      if (diag.isDeleted(v0) || diag.isDeleted(v1) || !check(diag, v0, v1)) {
        continue;
      }
      rule(diag, v0, v1);
      new_matches = true;
      n_simplifications++;
    }
  }

  return n_simplifications;
}

int32_t gadgetSimp(ZXDiagram &diag) {
  int32_t n_simplifications = 0;
  bool new_matches = true;

  while (new_matches) {
    new_matches = false;
    for (auto [v, _] : diag.getVertices()) {

      if (diag.isDeleted(v))
        continue;

      if (checkAndFuseGadget(diag, v)) {

        new_matches = true;
        n_simplifications++;
      }
    }
  }
  return n_simplifications;
}

int32_t idSimp(ZXDiagram &diag) {
  return simplifyVertices(diag, checkIdSimp, removeId);
}

int32_t spiderSimp(ZXDiagram &diag) {
  return simplifyEdges(diag, checkSpiderFusion, fuseSpiders);
}

int32_t localCompSimp(ZXDiagram &diag) {
  return simplifyVertices(diag, checkLocalComp, localComp);
}

int32_t pivotPauliSimp(ZXDiagram &diag) {
  return simplifyEdges(diag, checkPivotPauli, pivotPauli);
}

int32_t pivotSimp(ZXDiagram &diag) {
  return simplifyEdges(diag, checkPivot, pivot);
}

int32_t interiorCliffordSimp(ZXDiagram &diag) {
  spiderSimp(diag);

  bool new_matches = true;
  int32_t n_simplifications = 0;
  int32_t n_id, n_spider, n_pivot, n_localComp;
  while (new_matches) {
    new_matches = false;
    n_id = idSimp(diag);
    n_spider = spiderSimp(diag);
    n_pivot = pivotPauliSimp(diag);
    n_localComp = localCompSimp(diag);

    if (n_id + n_spider + n_pivot + n_localComp != 0) {
      new_matches = true;
      n_simplifications++;
    }
    // std::cout << "ID " << n_id << "\n";
    // std::cout << "SPIDER " << n_spider << "\n";
    // std::cout << "PIVOT PAULI" << n_pivot << "\n";
    // std::cout << "LOCALCOMP " << n_localComp << "\n";
  }
  return n_simplifications;
}

int32_t cliffordSimp(ZXDiagram &diag) {
  bool new_matches = true;
  int32_t n_simplifications = 0;
  int32_t n_clifford, n_pivot;
  while (new_matches) {
    new_matches = false;
    n_clifford = interiorCliffordSimp(diag);
    n_pivot = pivotSimp(diag);
    if (n_clifford + n_pivot != 0) {
      new_matches = true;
      n_simplifications++;
    }
  }
  return n_simplifications;
}

int32_t pivotgadgetSimp(ZXDiagram &diag) {
  return simplifyEdges(diag, checkPivotGadget, pivotGadget);
}

int32_t fullReduce(ZXDiagram &diag) {
  diag.toGraphlike();
  interiorCliffordSimp(diag);

  // pivotgadgetSimp(diag);

  int32_t n_gadget, n_pivot;
  int32_t n_simplifications = 0;
  while (true) {
    cliffordSimp(diag);
    n_gadget = gadgetSimp(diag);
    interiorCliffordSimp(diag);
    n_pivot = pivotgadgetSimp(diag);
    if (n_gadget + n_pivot == 0)
      break;
    n_simplifications += n_gadget + n_pivot;
    // std::cout <<"Gadget: " << n_gadget << "\n";
    // std::cout << "Pivot: " << n_pivot << "\n";
    // std::cout << "\nSimps: " << n_simplifications << "\n\n";
  }
  // cliffordSimp(diag);
  return n_simplifications;
}
} // namespace zx
