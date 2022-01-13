#include "Definitions.hpp"
#include "Simplify.hpp"
#include "ZXDiagram.hpp"

#include <iostream>

int main(int argc, char **argv) {
  zx::ZXDiagram d0(argv[1]);
  zx::ZXDiagram d1(argv[2]);
    d0.invert();
  d0.concat(d1);
   zx::full_reduce(d0);

  // d0 = d1;
  std::cout << d0.get_nedges() << "\n";
  std::cout << d0.get_nvertices() << "\n";

  for (auto [from, to] : d0.get_edges()) {
    std::cout << from
              << (d0.get_edge(from, to).value().type == zx::EdgeType::Hadamard
                      ? "- -"
                      : "---")
              << to << "\n";
  }
  std::cout << ""
            << "\n";

  for (int i = 0; i < d0.get_inputs().size(); i++) {
    std::cout << d0.get_inputs()[i] << "--" << d0.get_outputs()[i] << "\n";
  }
  std::cout << ""
            << "\n";

  for (auto [v, data] : d0.get_vertices())
    std::cout << v << " p: " << data.phase << "\n";
  std::cout << ""
            << "\n";
  for (auto [v, data] : d0.get_vertices()) {
    std::cout << v << " p:" << data.phase << " boundary "
              << (data.type == zx::VertexType::Boundary ? "True" : "False")
              << " type " << (d0.type(v) == zx::VertexType::Z ? "Z" : "X")
              << "\n";
  }

  if (d0.get_inputs().size() == d0.get_nedges()) {
    std::cout << "EQUIVALENT"
              << "\n";
  } else {
    std::cout << "NOT EQUIVALENT"
              << "\n";
  }
  return 0;
}
