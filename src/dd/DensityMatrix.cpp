#include "dd/Matrix.hpp"
#include "dd/Node.hpp"

namespace dd {

void getDensityMatrixFromDD(Edge<dNode>& e, const std::complex<fp>& amp,
                            const std::size_t i, const std::size_t j,
                            CMat& mat) {
  // calculate new accumulated amplitude
  const auto c = amp * static_cast<std::complex<fp>>(e.w);
  if (e.isTerminal()) {
    mat.at(i).at(j) = c;
    return;
  }

  const std::size_t x = i | (1ULL << e.p->v);
  const std::size_t y = j | (1ULL << e.p->v);

  // recursive case
  if (!e.p->e[0].w.exactlyZero()) {
    dEdge::applyDmChangesToEdge(e.p->e[0]);
    getDensityMatrixFromDD(e.p->e[0], c, i, j, mat);
    dd::dEdge::revertDmChangesToEdge(e.p->e[0]);
  }
  if (!e.p->e[1].w.exactlyZero()) {
    dEdge::applyDmChangesToEdge(e.p->e[1]);
    getDensityMatrixFromDD(e.p->e[1], c, i, y, mat);
    dd::dEdge::revertDmChangesToEdge(e.p->e[1]);
  }
  if (!e.p->e[2].w.exactlyZero()) {
    dEdge::applyDmChangesToEdge(e.p->e[2]);
    getDensityMatrixFromDD(e.p->e[2], c, x, j, mat);
    dd::dEdge::revertDmChangesToEdge(e.p->e[2]);
  }
  if (!e.p->e[3].w.exactlyZero()) {
    dEdge::applyDmChangesToEdge(e.p->e[3]);
    getDensityMatrixFromDD(e.p->e[3], c, x, y, mat);
    dd::dEdge::revertDmChangesToEdge(e.p->e[3]);
  }
}

CMat getDensityMatrixFromDD(Edge<dNode>& e) {
  dEdge::alignDensityEdge(e);
  const std::size_t dim = 2ULL << e.p->v;
  // allocate resulting matrix
  auto mat = CMat(dim, CVec(dim, {0.0, 0.0}));
  dEdge::applyDmChangesToEdge(e);
  getDensityMatrixFromDD(e, 1, 0ULL, 0ULL, mat);
  dd::dEdge::revertDmChangesToEdge(e);
  return mat;
}

} // namespace dd
