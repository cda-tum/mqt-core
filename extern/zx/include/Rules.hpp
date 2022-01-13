#ifndef JKQZX_INCLUDE_RULES_HPP_
#define JKQZX_INCLUDE_RULES_HPP_

#include "ZXDiagram.hpp"

namespace zx {
bool check_id_simp(ZXDiagram &diag, Vertex v);

void remove_id(ZXDiagram &diag, Vertex v);

bool check_spider_fusion(ZXDiagram &diag, Vertex v0, Vertex v1);

void fuse_spiders(ZXDiagram &diag, Vertex v0, Vertex v1);

bool check_local_comp(ZXDiagram &diag, Vertex v);

void local_comp(ZXDiagram &diag, Vertex v);

bool check_pivot_pauli(ZXDiagram &diag, Vertex v0, Vertex v1);

void pivot_pauli(ZXDiagram &diag, Vertex v0, Vertex v1);

bool check_pivot(ZXDiagram &diag, Vertex v0, Vertex v1);

void pivot(ZXDiagram &diag, Vertex v0, Vertex v1);

bool check_pivot_gadget(ZXDiagram &diag, Vertex v0, Vertex v1);

void pivot_gadget(ZXDiagram &diag, Vertex v0, Vertex v1);

bool check_and_fuse_gadget(ZXDiagram &diag, Vertex v);
} // namespace zx

#endif /* JKQZX_INCLUDE_RULES_HPP_ */
