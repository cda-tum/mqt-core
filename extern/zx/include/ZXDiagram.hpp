#ifndef ZX_INCLUDE_GRAPH_HPP_
#define ZX_INCLUDE_GRAPH_HPP_

#include <algorithm>
#include <cstddef>
#include <memory>
#include <numeric>
#include <optional>
#include <stdint.h>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "Definitions.hpp"
#include "Rational.hpp"
#include "Utils.hpp"
#include "Expression.hpp"

    namespace zx {

<<<<<<< HEAD
  class ZXDiagram {
  public:
    ZXDiagram() = default;
    ZXDiagram(int32_t nqubits); // create n_qubit identity_diagram
    explicit ZXDiagram(std::string filename);
    // explicit ZXDiagram(const qc::QuantumComputation &circuit);
=======
class ZXDiagram {
public:
  ZXDiagram() = default;
  ZXDiagram(int32_t nqubits); // create n_qubit identity_diagram
  explicit ZXDiagram(std::string filename);
  // explicit ZXDiagram(const qc::QuantumComputation &circuit);
>>>>>>> ea18283 (Removed QFR as dependency)

    void add_edge(Vertex from, Vertex to, EdgeType type = EdgeType::Simple);
    void add_hadamard_edge(Vertex from, Vertex to) {
      add_edge(from, to, EdgeType::Hadamard);
    };
    void add_edge_parallel_aware(Vertex from, Vertex to,
                                 EdgeType type = EdgeType::Simple);
    void remove_edge(Vertex from, Vertex to);

<<<<<<< HEAD
    Vertex add_vertex(const VertexData &data);
    Vertex add_vertex(Qubit qubit, Col col = 0,
                      const Expression &phase = Expression(),
                      VertexType type = VertexType::Z);
    void remove_vertex(Vertex to_remove);
=======
  Vertex add_vertex(const VertexData &data);
  Vertex add_vertex(Qubit qubit, Col col = 0,
                    const Expression& phase = Expression(),
                    VertexType type = VertexType::Z);
  void remove_vertex(Vertex to_remove);
>>>>>>> ea18283 (Removed QFR as dependency)

    int32_t get_ndeleted() const { return deleted.size(); }
    [[nodiscard]] int32_t get_nvertices() const { return nvertices; }
    [[nodiscard]] int32_t get_nedges() const { return nedges; }
    [[nodiscard]] int32_t get_nqubits() const { return inputs.size(); }

    [[nodiscard]] bool connected(Vertex from, Vertex to) const;
    [[nodiscard]] std::optional<Edge> get_edge(Vertex from, Vertex to) const;
    [[nodiscard]] std::vector<Edge> &incident_edges(Vertex v) {
      return edges[v];
    }

    [[nodiscard]] int32_t degree(Vertex v) const { return edges[v].size(); }

    [[nodiscard]] const Expression &phase(Vertex v) const {
      return vertices[v].value().phase;
    }

<<<<<<< HEAD
    [[nodiscard]] Qubit qubit(Vertex v) const {
      return vertices[v].value().qubit;
    }
=======
  [[nodiscard]] Qubit qubit(Vertex v) const {
    return vertices[v].value().qubit;
  }
>>>>>>> ea18283 (Removed QFR as dependency)

    [[nodiscard]] VertexType type(Vertex v) const {
      return vertices[v].value().type;
    }

    [[nodiscard]] std::optional<VertexData> get_vdata(Vertex v) const {
      return vertices[v];
    }

    [[nodiscard]] std::vector<std::pair<Vertex, VertexData &>> get_vertices();
    [[nodiscard]] std::vector<std::pair<Vertex, Vertex>> get_edges();

    [[nodiscard]] const std::vector<Vertex> &get_inputs() const {
      return inputs;
    }

    [[nodiscard]] const std::vector<Vertex> &get_outputs() const {
      return outputs;
    }

    [[nodiscard]] bool is_deleted(Vertex v) const {
      return !vertices[v].has_value();
    }

    [[nodiscard]] bool is_boundary_vertex(Vertex v) const {
      return vertices[v].value().type == VertexType::Boundary;
    }

    [[nodiscard]] bool is_input(Vertex v) const;
    [[nodiscard]] bool is_output(Vertex v) const;

    void add_phase(Vertex v, const Expression &phase) {
      vertices[v].value().phase += phase;
    }

    void set_phase(Vertex v, const Expression &phase) {
      vertices[v].value().phase = phase;
    }

    void set_type(Vertex v, VertexType type) {
      vertices[v].value().type = type;
    }

    void to_graph_like();

<<<<<<< HEAD
    [[nodiscard]] bool is_identity() const;
    // [[nodiscard]] bool is_identity(const qc::Permutation &perm) const;
=======
  [[nodiscard]] bool is_identity() const;
  // [[nodiscard]] bool is_identity(const qc::Permutation &perm) const;
>>>>>>> ea18283 (Removed QFR as dependency)

    [[nodiscard]] ZXDiagram adjoint() const;

    ZXDiagram &invert();

    ZXDiagram &concat(const ZXDiagram &rhs);

    // What about Swaps?

<<<<<<< HEAD
    void make_ancilla(Qubit qubit);

  private:
    std::vector<std::vector<Edge>> edges;
    std::vector<std::optional<VertexData>> vertices;
    std::vector<Vertex> deleted;
    std::vector<Vertex> inputs;
    std::vector<Vertex> outputs;
    int32_t nvertices = 0;
    int32_t nedges = 0;
    // std::optional<qc::Permutation> initial_layout;
    // std::optional<qc::Permutation> output_permutation;

    void add_z_spider(Qubit qubit, std::vector<Vertex> &qubit_vertices,
                      const Expression &phase = Expression(),
                      EdgeType type = EdgeType::Simple);
    void add_x_spider(Qubit qubit, std::vector<Vertex> &qubit_vertices,

                      const Expression &phase = Expression(),
                      EdgeType type = EdgeType::Simple);
    void add_cnot(Qubit ctrl, Qubit target,
                  std::vector<Vertex> &qubit_vertices);
=======
  void make_ancilla(Qubit qubit);

private:
  std::vector<std::vector<Edge>> edges;
  std::vector<std::optional<VertexData>> vertices;
  std::vector<Vertex> deleted;
  std::vector<Vertex> inputs;
  std::vector<Vertex> outputs;
  int32_t nvertices = 0;
  int32_t nedges = 0;
  // std::optional<qc::Permutation> initial_layout;
  // std::optional<qc::Permutation> output_permutation;

  void add_z_spider(Qubit qubit, std::vector<Vertex> &qubit_vertices,
                    const Expression& phase = Expression(),
                    EdgeType type = EdgeType::Simple);
  void add_x_spider(Qubit qubit, std::vector<Vertex> &qubit_vertices,
const Expression& phase = Expression(),
                    EdgeType type = EdgeType::Simple);
  void add_cnot(Qubit ctrl, Qubit target,
                std::vector<Vertex> &qubit_vertices);
  void add_cphase(PiRational phase, Qubit ctrl, Qubit target,
                  std::vector<Vertex> &qubit_vertices);
  void add_swap(Qubit ctrl, Qubit target,
                std::vector<Vertex> &qubit_vertices);
  void add_ccx(Qubit ctrl_0, Qubit ctrl_1, Qubit target,
               std::vector<Vertex> &qubit_vertices);
>>>>>>> ea18283 (Removed QFR as dependency)

    void add_cphase(PiRational phase, Qubit ctrl, Qubit target,
                    std::vector<Vertex> &qubit_vertices);
    void add_swap(Qubit ctrl, Qubit target,
                  std::vector<Vertex> &qubit_vertices);
    void add_ccx(Qubit ctrl_0, Qubit ctrl_1, Qubit target,
                 std::vector<Vertex> &qubit_vertices);

    std::vector<Vertex> init_graph(int nqubits);
    void close_graph(std::vector<Vertex> &qubit_vertices);

    void remove_half_edge(Vertex from, Vertex to);

    std::vector<Edge>::iterator get_edge_ptr(Vertex from, Vertex to);

    // using op_it =
    //     decltype(std::begin(std::vector<std::unique_ptr<qc::Operation>>()));
    // op_it parse_op(op_it it, op_it end,
    //                       std::vector<Vertex> &qubit_vertices);
  };
} // namespace zx
#endif /* ZX_INCLUDE_GRAPH_HPP_ */
