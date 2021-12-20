#include "ZXDiagram.hpp"
#include "Definitions.hpp"
#include "QuantumComputation.hpp"
#include "Rational.hpp"
#include "dd/Definitions.hpp"
#include "operations/OpType.hpp"
#include <algorithm>
#include <vector>

namespace zx {

ZXDiagram::ZXDiagram(std::string filename) {
  qc::QuantumComputation qc(filename);
  auto qubit_vertices = init_graph(qc.getNqubits());

  for (const auto &op : qc) {
    if (op->getNcontrols() == 0 && op->getNtargets() == 1) {
      auto target = op->getTargets()[0];

      switch (op->getType()) {
      case qc::OpType::Z: {
        add_z_spider(target, qubit_vertices, Rational(1, 1));
        break;
      }
      case qc::OpType::RZ: {
        add_z_spider(target, qubit_vertices, op->getParameter()[0]);
        break;
      }
      case qc::OpType::X: {
        add_x_spider(target, qubit_vertices, Rational(1, 1));
        break;
      }
      case qc::OpType::RX: {
        add_x_spider(target, qubit_vertices, op->getParameter()[0]);
        break;
      }
      case qc::OpType::T: {
        add_z_spider(target, qubit_vertices, Rational(1, 4));
        break;
      }
      case qc::OpType::Tdag: {
        add_z_spider(target, qubit_vertices, Rational(-1, 4));
        break;
      }
      case qc::OpType::S: {
        add_z_spider(target, qubit_vertices, Rational(1, 2));
        break;
      }
      case qc::OpType::Sdag: {
        add_z_spider(target, qubit_vertices, Rational(-1, 2));
        break;
      }
      case qc::OpType::U2: {
        add_z_spider(target, qubit_vertices,
                     op->getParameter()[1] - Rational(1, 2));
        add_x_spider(target, qubit_vertices, Rational(1, 2));
        add_z_spider(target, qubit_vertices,
                     op->getParameter()[0] + Rational(1, 2));
        break;
      }
      case qc::OpType::U3: {
        add_z_spider(target, qubit_vertices, op->getParameter()[2]);
        add_x_spider(target, qubit_vertices, Rational(1, 2));
        add_z_spider(target, qubit_vertices,
                     op->getParameter()[0] + Rational(1, 1));
        add_x_spider(target, qubit_vertices, Rational(1, 2));
        add_z_spider(target, qubit_vertices,
                     op->getParameter()[1] + Rational(3, 1));
        break;
      }
      case qc::OpType::SWAP: {
        auto target2 = op->getTargets()[0];
        add_cnot(target, target2, qubit_vertices);
        add_cnot(target2, target, qubit_vertices);
        add_cnot(target, target2, qubit_vertices);
        break;
      }
      case qc::OpType::H: {
        add_z_spider(target, qubit_vertices, 0, EdgeType::Hadamard);
        break;
      }
      default: {
        throw ZXException("Unsupported Operation: " +
                          qc::toString(op->getType()));
      }
      }
    } else if (op->getNcontrols() == 1 && op->getNtargets() == 1) {
      auto target = op->getTargets()[0];
      auto ctrl = (*op->getControls().begin()).qubit;
      switch (op->getType()) { // TODO: any gate can be controlled
      case qc::OpType::X: {
        add_cnot(ctrl, target, qubit_vertices);
        break;
      }
      case qc::OpType::Z: {
        auto phase = Rational(op->getParameter()[0]);
        add_z_spider(target, qubit_vertices, phase / 2);
        add_cnot(ctrl, target, qubit_vertices);
        add_z_spider(target, qubit_vertices, -phase / 2);
        add_cnot(ctrl, target, qubit_vertices);
        break;
      }
      default: {
        throw ZXException("Unsupported Controlled Operation: " +
                          qc::toString(op->getType()));
      }
      }
    }
  }
  close_graph(qubit_vertices);
}

void ZXDiagram::add_edge(Vertex from, Vertex to, EdgeType type) {
  edges[from].push_back({to, type});
  edges[to].push_back({from, type});
  nedges++;
}

void ZXDiagram::remove_edge(Vertex from, Vertex to) {
  remove_half_edge(from, to);
  remove_half_edge(to, from);
  nedges--;
}

  void ZXDiagram::remove_half_edge(Vertex from, Vertex to) {
      auto incident = incident_edges(from);
  std::remove_if(incident.begin(), incident.end(),
                 [&](auto &edge) { return edge.to == to; });
  }
  
Vertex ZXDiagram::add_vertex(const VertexData &data) {
  vertices.push_back(data);
  edges.emplace_back();
  return nvertices++;
}

  void ZXDiagram::remove_vertex(Vertex to_remove) {
    deleted.push_back(to_remove);
    vertices[to_remove].reset();
    nvertices--;
    
    for(auto [to, _]: incident_edges(to_remove)) {
      remove_half_edge(to, to_remove);
      nedges--;
    }
  }

[[nodiscard]] std::optional<Edge> ZXDiagram::get_edge(Vertex from,
                                                      Vertex to) const {
  std::optional<Edge> ret;
  auto incident = incident_edges(from);
  auto edge = std::find_if(incident.begin(), incident.end(),
                           [&](auto &edge) { return edge.to == to; });
  if (edge != incident.end())
    ret = *edge;
  return ret;
}

void ZXDiagram::add_z_spider(dd::Qubit qubit,
                             std::vector<Vertex> &qubit_vertices,
                             Rational phase, EdgeType type) {
  auto new_vertex =
      add_vertex({vertices[qubit].value().col, qubit, phase, VertexType::Z});

  add_edge(qubit, new_vertex, type);
  qubit_vertices[qubit] = new_vertex;
}

void ZXDiagram::add_x_spider(dd::Qubit qubit,
                             std::vector<Vertex> &qubit_vertices,
                             Rational phase, EdgeType type) {
  auto new_vertex = add_vertex(
      {vertices[qubit].value().col + 1, qubit, phase, VertexType::X});
  add_edge(qubit, new_vertex, type);
  qubit_vertices[qubit] = new_vertex;
}

void ZXDiagram::add_cnot(dd::Qubit ctrl, dd::Qubit target,
                         std::vector<Vertex> &qubit_vertices) {
  add_z_spider(ctrl, qubit_vertices);
  add_x_spider(target, qubit_vertices);
  add_edge(qubit_vertices[ctrl], qubit_vertices[target]);
}

std::vector<Vertex> ZXDiagram::init_graph(int nqubits) {

  std::vector<Vertex> qubit_vertices(nqubits);
  for (size_t i = 0; i < qubit_vertices.size(); i++) {
    auto v = add_vertex(
        {1, static_cast<dd::Qubit>(i), Rational(0, 1), VertexType::Boundary});
    qubit_vertices[i] = v;
    inputs.push_back(v);
  }
  return qubit_vertices;
}

void ZXDiagram::close_graph(std::vector<Vertex> &qubit_vertices) {
  for (Vertex v : qubit_vertices) {
    VertexData v_data = vertices[v].value();
    Vertex new_v =
        add_vertex({v_data.col + 1, v_data.qubit, 0, VertexType::Boundary});
    add_edge(v, new_v);
    outputs.push_back(new_v);
  }
}
} // namespace zx
