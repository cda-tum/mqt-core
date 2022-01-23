#include "Rules.hpp"
#include "Definitions.hpp"
#include "Rational.hpp"
#include "ZXDiagram.hpp"
#include <algorithm>
#include <iostream>

namespace zx {
bool check_id_simp(ZXDiagram &diag, Vertex v) {
  return diag.degree(v) == 2 && diag.phase(v) == 0 &&
         !diag.is_boundary_vertex(v);
}

void remove_id(ZXDiagram &diag, Vertex v) {
  auto edges = diag.incident_edges(v);
  Vertex v0 = edges[0].to;
  Vertex v1 = edges[1].to;

  EdgeType type = EdgeType::Simple;
  if (edges[0].type != edges[1].type) {
    type = EdgeType::Hadamard;
  }
  //  diag.add_edge(v0, v1,type);
  diag.add_edge_parallel_aware(v0, v1, type);
  diag.remove_vertex(v);
}

bool check_spider_fusion(ZXDiagram &diag, Vertex v0, Vertex v1) {
  auto edge_opt = diag.get_edge(v0, v1);
  return v0 != v1 && diag.type(v0) == diag.type(v1) &&
         edge_opt.value_or(Edge{0, EdgeType::Hadamard}).type ==
             EdgeType::Simple &&
         diag.type(v0) != VertexType::Boundary;
}

void fuse_spiders(ZXDiagram &diag, Vertex v0, Vertex v1) {
  diag.add_phase(v0, diag.phase(v1));
  for (auto &[to, type] : diag.incident_edges(v1)) {
    if (v0 != to)
      diag.add_edge_parallel_aware(v0, to, type);
  }
  diag.remove_vertex(v1);
}

bool check_local_comp(ZXDiagram &diag, Vertex v) {
  auto v_data = diag.get_vdata(v).value_or(
      VertexData{0, 0, Rational(0, 1), VertexType::X});
  if (v_data.type != VertexType::Z || v_data.phase.denom != 2)
    return false;

  auto &edges = diag.incident_edges(v);
  return std::all_of(edges.begin(), edges.end(), [&](auto &edge) {
    return edge.type == EdgeType::Hadamard &&
           diag.type(edge.to) == VertexType::Z;
  });
}

void local_comp(ZXDiagram &diag, Vertex v) { // TODO:scalars
  auto phase = -diag.phase(v);
  auto &edges = diag.incident_edges(v);

  for (auto i = 0; i < edges.size(); i++) {
    auto &[n0, _] = edges[i];
    diag.add_phase(n0, phase);
    for (auto j = i + 1; j < edges.size(); j++) {
      auto &[n1, _] = edges[j];
      diag.add_edge_parallel_aware(n0, n1, EdgeType::Hadamard);
    }
  }
  diag.remove_vertex(v);
}

static bool is_pauli(ZXDiagram &diag, Vertex v) {
  return diag.phase(v).is_integer();
}

bool check_pivot_pauli(ZXDiagram &diag, Vertex v0, Vertex v1) {
  auto v0_data = diag.get_vdata(v0).value_or(
      VertexData{0, 0, Rational(0, 1), VertexType::X});
  auto v1_data = diag.get_vdata(v0).value_or(
      VertexData{0, 0, Rational(0, 1), VertexType::X});

  if (v0_data.type != VertexType::Z || // maybe problem if there is a self-loop?
      v1_data.type != VertexType::Z || !is_pauli(diag, v0) ||
      !is_pauli(diag, v1)) {
    return false;
  }

  auto edge_opt = diag.get_edge(v0, v1);

  if (!edge_opt.has_value() || edge_opt.value().type != EdgeType::Hadamard) {
    return false;
  }

  auto &edges_v0 = diag.incident_edges(v0);
  auto is_valid_edge = [&](const Edge &e) {
    return diag.type(e.to) == VertexType::Z && e.type == EdgeType::Hadamard;
  };

  if (!std::all_of(edges_v0.begin(), edges_v0.end(), is_valid_edge))
    return false;

  auto &edges_v1 = diag.incident_edges(v1);

  return std::all_of(edges_v1.begin(), edges_v1.end(), is_valid_edge);
}

void pivot_pauli(ZXDiagram &diag, Vertex v0, Vertex v1) { // TODO: phases

  auto phase_v0 = diag.phase(v0);
  auto phase_v1 = diag.phase(v1);

  auto &edges_v0 = diag.incident_edges(v0);
  auto &edges_v1 = diag.incident_edges(v1);

  for (auto &[neighbor_v0, _] : edges_v0) {
    if (neighbor_v0 == v1) {
      continue;
    }

    diag.add_phase(neighbor_v0, phase_v1);
    for (auto &[neighbor_v1, _] : edges_v1) {
      if (neighbor_v1 != v0)
        diag.add_edge_parallel_aware(neighbor_v0, neighbor_v1,
                                     EdgeType::Hadamard);
    }
  }

  for (auto &[neighbor_v1, _] : edges_v1) {
    diag.add_phase(neighbor_v1, phase_v0);
  }

  diag.remove_vertex(v0);
  diag.remove_vertex(v1);
}

bool is_interior(ZXDiagram &diag, Vertex v) {
  auto &edges = diag.incident_edges(v);
  return std::all_of(edges.begin(), edges.end(), [&](auto &edge) {
    return diag.degree(edge.to) > 1 && diag.type(edge.to) == VertexType::Z;
  });
}

bool check_pivot(ZXDiagram &diag, Vertex v0, Vertex v1) {
  auto v0_type = diag.type(v0);
  auto v1_type = diag.type(v1);

  if (v0 == v1 || v0_type != VertexType::Z || v1_type != VertexType::Z) {
    return false;
  }

  auto edge_opt = diag.get_edge(v0, v1);
  if (!edge_opt.has_value() || edge_opt.value().type != EdgeType::Hadamard) {
    return false;
  }

  auto &edges_v0 = diag.incident_edges(v0);
  auto is_invalid_edge = [&](const Edge &e) {
    auto to_type = diag.type(e.to);
    return !((to_type == VertexType::Z && e.type == EdgeType::Hadamard) ||
             to_type == VertexType::Boundary);
  };

  if (std::any_of(edges_v0.begin(), edges_v0.end(), is_invalid_edge))
    return false;

  auto &edges_v1 = diag.incident_edges(v1);

  if (std::any_of(edges_v1.begin(), edges_v1.end(), is_invalid_edge))
    return false;

  // auto is_interior = [&](Vertex v) {
  //   auto &edges = diag.incident_edges(v);
  //   return std::all_of(edges.begin(), edges.end(), [&](auto &edge) {
  //     return diag.degree(edge.to) > 1 && diag.type(edge.to) == VertexType::Z;
  //   });
  // };

  auto is_interior_pauli = [&](Vertex v) {
    return is_interior(diag, v) && is_pauli(diag, v);
  };

  return (is_interior_pauli(v0) || is_interior_pauli(v1));
}

static void extract_gadget(ZXDiagram &diag, Vertex v) {
  auto v_data = diag.get_vdata(v).value();
  Vertex phase_vert = diag.add_vertex(v_data.qubit, v_data.col, v_data.phase);
  Vertex id_vert = diag.add_vertex(v_data.col, v_data.qubit);
  diag.set_phase(v, Rational(0, 1));
  diag.add_hadamard_edge(v, id_vert);
  diag.add_hadamard_edge(id_vert, phase_vert);
}

static void extract_pauli_gadget(ZXDiagram &diag, Vertex v) {
  if (diag.phase(v).is_integer())
    return;

  extract_gadget(diag, v);
}

static void ensure_interior(ZXDiagram &diag, Vertex v) {
  // auto &edges = diag.incident_edges(v);
  // auto v_data = diag.get_vdata(v).value();
  // for (auto &[to, type] : edges) {
  //   if (diag.is_boundary_vertex(to)) {
  //     Vertex new_v = diag.add_vertex(v_data.qubit, v_data.col, Rational(0,
  //     1));

  //     auto other_dir = diag.get_edge(to, v);
  //     auto boundary_edge_type = type == zx::EdgeType::Simple
  //                                   ? zx::EdgeType::Hadamard
  //                                   : zx::EdgeType::Simple;

  //     auto& new_edges = diag.incident_edges(new_v);
  //     new_edges.emplace_back(v, EdgeType::Hadamard);
  //     new_edges.emplace_back(to, boundary_edge_type);

  //     to = new_v;
  //     type = zx::EdgeType::Hadamard;

  //     other_dir.value().to = new_v;
  //     other_dir.value().type = boundary_edge_type;
  //   }
  // }

  auto edges = diag.incident_edges(v);
  auto v_data = diag.get_vdata(v).value();

  for (auto &[to, type] : edges) {
    if (!diag.is_boundary_vertex(to))
      continue;

    Vertex new_v = diag.add_vertex(v_data.qubit, v_data.col, Rational(0, 1));
    auto boundary_edge_type = type == zx::EdgeType::Simple
                                  ? zx::EdgeType::Hadamard
                                  : zx::EdgeType::Simple;

    diag.add_edge(v, new_v, EdgeType::Hadamard);
    diag.add_edge(to, new_v, boundary_edge_type);
    diag.remove_edge(v, to);
  }
}

static void ensure_pauli_vertex(ZXDiagram &diag, Vertex v) {
  extract_pauli_gadget(diag, v);
  ensure_interior(diag, v);
}

void pivot(ZXDiagram &diag, Vertex v0, Vertex v1) {

  ensure_pauli_vertex(diag, v0);
  ensure_pauli_vertex(diag, v1);

  pivot_pauli(diag, v0, v1);
}

bool check_pivot_gadget(ZXDiagram &diag, Vertex v0, Vertex v1) {
  auto p0 = diag.phase(v0);
  auto p1 = diag.phase(v1);
  if (!p0.is_integer()) {
    if (!p1.is_integer()) {
      return false;
    }
  } else if (p1.is_integer()) {
    return false;
  }
  if (!is_interior(diag, v0) || !is_interior(diag, v1))
    return false;

  return check_pivot(diag, v0, v1);
}

void pivot_gadget(ZXDiagram &diag, Vertex v0, Vertex v1) {
  if (is_pauli(diag, v0)) {
    extract_gadget(diag, v1);
  } else {
    extract_gadget(diag, v0);
  }
  pivot_pauli(diag, v0, v1);
}

bool check_and_fuse_gadget(ZXDiagram &diag, Vertex v) {
  if (diag.degree(v) != 1 || diag.is_boundary_vertex(v))
    return false;

  auto [id0, id0_etype] = diag.incident_edges(v)[0];
  if (diag.phase(id0) != 0 || diag.degree(id0) < 2 ||
      id0_etype != zx::EdgeType::Hadamard)
    return false;

  if (diag.degree(id0) == 2) {
    auto &[v0, _] = diag.incident_edges(id0)[0].to == v
                        ? diag.incident_edges(id0)[1]
                        : diag.incident_edges(id0)[0];
    diag.add_phase(v0, diag.phase(v));
    diag.remove_vertex(v);
    diag.remove_vertex(id0);
    return true;
  }

  Vertex n0;
  EdgeType n0_etype;
  for (auto &[n, etype] : diag.incident_edges(id0)) {
    if (n == v)
      continue;

    if (etype != zx::EdgeType::Hadamard)
      return false;
    n0 = n;
    n0_etype = etype;
  }

  Vertex id1;
  Vertex phase_spider = -1;

  bool found_gadget = false;
  for (auto &[n, etype] : diag.incident_edges(n0)) {
    if (n == id0)
      continue;

    if (etype != zx::EdgeType::Hadamard || diag.is_deleted(n) ||
        diag.phase(n) != 0 || diag.degree(n) != diag.degree(id0) ||
        diag.connected(n, id0)) {
      continue;
    }

    found_gadget = true;
    id1 = n;

    for (auto &[nn, nn_etype] :
         diag.incident_edges(id1)) { // Todo: maybe problem with parallel edge?
                                     // There shouldnt be any
      if (nn_etype != zx::EdgeType::Hadamard || diag.is_deleted(nn)) {
        found_gadget = false;
        break; // not a phase gadget
      }

      if (diag.degree(nn) == 1 && !diag.is_boundary_vertex(nn)) {
        found_gadget = true;
        phase_spider = nn;
        continue;
      }

      if (std::find_if(diag.incident_edges(nn).begin(),
                       diag.incident_edges(nn).end(), [&](Edge e) {
                         return e.to == id0;
                       }) == diag.incident_edges(nn).end()) {
        found_gadget = false;
        break;
      }
    }

    if (found_gadget)
      break;
  }

  if (!found_gadget || phase_spider < 0)
    return false;

  diag.add_phase(v, diag.phase(phase_spider));
  diag.remove_vertex(phase_spider);
  diag.remove_vertex(id1);
  return true;
}

} // namespace zx
