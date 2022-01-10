#ifndef JKQZX_INCLUDE_SIMPLIFY_HPP_
#define JKQZX_INCLUDE_SIMPLIFY_HPP_

#include "Rules.hpp"
#include "ZXDiagram.hpp"

namespace zx {
using VertexCheckFun = decltype(check_id_simp);
using VertexRuleFun = decltype(remove_id);
using EdgeCheckFun = decltype(check_spider_fusion);
using EdgeRuleFun = decltype(fuse_spiders);

int32_t simplify_vertices(ZXDiagram &diag, VertexCheckFun check,
                          VertexRuleFun rule);

int32_t simplify_edges(ZXDiagram &diag, EdgeCheckFun check, EdgeRuleFun rule);

int32_t id_simp(ZXDiagram &diag);

int32_t spider_simp(ZXDiagram &diag);

int32_t local_comp_simp(ZXDiagram &diag);

int32_t pivot_pauli_simp(ZXDiagram &diag);

int32_t pivot_simp(ZXDiagram &diag);

int32_t interior_clifford_simp(ZXDiagram &diag);

int32_t clifford_simp(ZXDiagram &diag);

int32_t full_reduce(ZXDiagram &diag);

} // namespace zx

#endif /* JKQZX_INCLUDE_SIMPLIFY_HPP_ */
