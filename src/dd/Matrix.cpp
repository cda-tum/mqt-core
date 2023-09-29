#include "dd/Matrix.hpp"

#include "dd/Node.hpp"

#include <iomanip>

namespace dd {

void getMatrixFromDD(const Edge<mNode>& e, const std::complex<fp>& amp,
                     const std::size_t i, const std::size_t j, CMat& mat) {
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
    getMatrixFromDD(e.p->e[0], c, i, j, mat);
  }
  if (!e.p->e[1].w.exactlyZero()) {
    getMatrixFromDD(e.p->e[1], c, i, y, mat);
  }
  if (!e.p->e[2].w.exactlyZero()) {
    getMatrixFromDD(e.p->e[2], c, x, j, mat);
  }
  if (!e.p->e[3].w.exactlyZero()) {
    getMatrixFromDD(e.p->e[3], c, x, y, mat);
  }
}

CMat getMatrixFromDD(const Edge<mNode>& e) {
  const std::size_t dim = 2ULL << e.p->v;
  // allocate resulting matrix
  auto mat = CMat(dim, CVec(dim, {0.0, 0.0}));
  getMatrixFromDD(e, 1, 0ULL, 0ULL, mat);
  return mat;
}

std::complex<fp> getValueByIndex(const Edge<mNode>& e, const std::size_t i,
                                 const std::size_t j) {
  if (e.isTerminal()) {
    return static_cast<std::complex<fp>>(e.w);
  }
  return getValueByIndex(e, 1, i, j);
}

std::complex<fp> getValueByIndex(const Edge<mNode>& e,
                                 const std::complex<fp>& amp,
                                 const std::size_t i, const std::size_t j) {
  const auto c = amp * static_cast<std::complex<fp>>(e.w);
  if (e.isTerminal()) {
    return c;
  }

  const bool row = (i & (1ULL << e.p->v)) != 0U;
  const bool col = (j & (1ULL << e.p->v)) != 0U;

  std::complex<fp> r{0.0, 0.0};
  if (!row && !col && !e.p->e[0].w.exactlyZero()) {
    r = getValueByIndex(e.p->e[0], c, i, j);
  } else if (!row && col && !e.p->e[1].w.exactlyZero()) {
    r = getValueByIndex(e.p->e[1], c, i, j);
  } else if (row && !col && !e.p->e[2].w.exactlyZero()) {
    r = getValueByIndex(e.p->e[2], c, i, j);
  } else if (row && col && !e.p->e[3].w.exactlyZero()) {
    r = getValueByIndex(e.p->e[3], c, i, j);
  }
  return r;
}

void printMatrix(const Edge<mNode>& e) {
  const std::size_t element = 2ULL << e.p->v;
  for (auto i = 0ULL; i < element; i++) {
    for (auto j = 0ULL; j < element; j++) {
      const auto amplitude = getValueByIndex(e, i, j);
      constexpr auto precision = 3;
      // set fixed width to maximum of a printed number
      // (-) 0.precision plus/minus 0.precision i
      constexpr auto width = 1 + 2 + precision + 1 + 2 + precision + 1;
      std::cout << std::setw(width)
                << ComplexValue::toString(amplitude.real(), amplitude.imag(),
                                          false, precision)
                << " ";
    }
    std::cout << "\n";
  }
  std::cout << std::flush;
}

} // namespace dd
