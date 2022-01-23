#include "Simplify.hpp"
#include "Definitions.hpp"
#include "Rules.hpp"
#include <iostream>

namespace zx {
int32_t simplify_vertices(ZXDiagram &diag, VertexCheckFun check,
                          VertexRuleFun rule) {
  int32_t n_simplifications = 0;
  bool new_matches = true;

  while (new_matches) {
    new_matches = false;
    for (auto [v, _] : diag.get_vertices()) {
      if (check(diag, v)) {
        rule(diag, v);
        new_matches = true;
        n_simplifications++;
      }
    }
  }

  return n_simplifications;
}

int32_t simplify_edges(ZXDiagram &diag, EdgeCheckFun check, EdgeRuleFun rule) {
  int32_t n_simplifications = 0;
  bool new_matches = true;

  while (new_matches) {
    new_matches = false;
    for (auto [v0, v1] : diag.get_edges()) {
      if (diag.is_deleted(v0) || diag.is_deleted(v1) || !check(diag, v0, v1)) {
        continue;
      }
      rule(diag, v0, v1);
      new_matches = true;
      n_simplifications++;
    }
  }

  return n_simplifications;
}

int32_t gadget_simp(ZXDiagram &diag) {
  int32_t n_simplifications = 0;
  bool new_matches = true;

  while (new_matches) {
    new_matches = false;
    for (auto [v, _] : diag.get_vertices()) {

      if (diag.is_deleted(v))
        continue;

      if (check_and_fuse_gadget(diag, v)) {

        new_matches = true;
        n_simplifications++;
      }
    }
  }
  return n_simplifications;
}

int32_t id_simp(ZXDiagram &diag) {
  return simplify_vertices(diag, check_id_simp, remove_id);
}

int32_t spider_simp(ZXDiagram &diag) {
  return simplify_edges(diag, check_spider_fusion, fuse_spiders);
}

int32_t local_comp_simp(ZXDiagram &diag) {
  return simplify_vertices(diag, check_local_comp, local_comp);
}

int32_t pivot_pauli_simp(ZXDiagram &diag) {
  return simplify_edges(diag, check_pivot_pauli, pivot_pauli);
}

int32_t pivot_simp(ZXDiagram &diag) {
  return simplify_edges(diag, check_pivot, pivot);
}

int32_t interior_clifford_simp(ZXDiagram &diag) {
  spider_simp(diag);

  bool new_matches = true;
  int32_t n_simplifications = 0;
  int32_t n_id, n_spider, n_pivot, n_local_comp;
  while (new_matches) {
    new_matches = false;
    n_id = id_simp(diag);
    n_spider = spider_simp(diag);
    n_pivot = pivot_pauli_simp(diag);
    n_local_comp = local_comp_simp(diag);

    if (n_id + n_spider + n_pivot + n_local_comp != 0) {
      new_matches = true;
      n_simplifications++;
    }
  }
  return n_simplifications;
}

int32_t clifford_simp(ZXDiagram &diag) {
  bool new_matches = true;
  int32_t n_simplifications = 0;
  int32_t n_clifford, n_pivot;
  while (new_matches) {
    new_matches = false;
    n_clifford = interior_clifford_simp(diag);
    n_pivot = pivot_simp(diag);
    if (n_clifford + n_pivot != 0) {
      new_matches = true;
      n_simplifications++;
    }
  }
  return n_simplifications;
}

int32_t pivot_gadget_simp(ZXDiagram &diag) {
  return simplify_edges(diag, check_pivot_gadget, pivot_gadget);
}

int32_t full_reduce(ZXDiagram &diag) {
  diag.to_graph_like();


  interior_clifford_simp(diag);
  pivot_gadget_simp(diag);

  int32_t n_gadget, n_pivot;
  int32_t n_simplifications = 0;
  while (true) {
    clifford_simp(diag);
    n_gadget = gadget_simp(diag);
    interior_clifford_simp(diag);
    n_pivot = pivot_gadget_simp(diag);
    if (n_gadget + n_pivot == 0)
      break;
    n_simplifications += n_gadget + n_pivot;
  }
  // clifford_simp(diag);
  return n_simplifications;
}
} // namespace zx
