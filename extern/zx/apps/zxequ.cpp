#include "Definitions.hpp"
#include "QuantumComputation.hpp"
#include "Simplify.hpp"
#include "ZXDiagram.hpp"

#include <iostream>

int main(int argc, char **argv) {
  qc::QuantumComputation c0(argv[1]);
  qc::QuantumComputation c1(argv[2]);
  zx::ZXDiagram d0(c0);
  zx::ZXDiagram d1(c1);
  d0.invert();
  d0.concat(d1);
  zx::full_reduce(d0);

  // d0 = d1;
  // std::cout << d0.get_nedges() << "\n";
  // std::cout << d0.get_nvertices() << "\n";

  // for (auto [from, to] : d0.get_edges()) {
  //   std::cout << from
  //             << (d0.get_edge(from, to).value().type ==
  //             zx::EdgeType::Hadamard
  //                     ? "- -"
  //                     : "---")
  //             << to << "\n";
  // }
  // std::cout << ""
  //           << "\n";

  // for (int i = 0; i < d0.get_inputs().size(); i++) {
  //   std::cout << d0.get_inputs()[i] << "--" << d0.get_outputs()[i] << "\n";
  // }
  // std::cout << ""
  //           << "\n";

  // for (auto [v, data] : d0.get_vertices())
  //   std::cout << v << " p: " << data.phase <<", q:" << ((int)data.qubit) <<"\n";
  // std::cout << ""
  //           << "\n";
  // for (auto [v, data] : d0.get_vertices()) {
  //   std::cout << v << " p:" << data.phase << " boundary "
  //             << (data.type == zx::VertexType::Boundary ? "True" : "False")
  //             << " type " << (d0.type(v) == zx::VertexType::Z ? "Z" : "X")
  //             << "\n";
  // }


  
  std::cout << static_cast<int>(c0.getNqubits()) << ", " << c0.getNops() << ", " << c1.getNops() << ", ";
  
  if(d0.is_identity())
    std::cout << "IDENTITY " << "\n";

  if (d0.get_inputs().size() == d0.get_nedges()) {
    std::cout << "TRUE";
  } else {
    std::cout << "FALSE";
  }
  return 0;
}
