#include "dd/Vector.hpp"

#include "dd/Node.hpp"

#include <cassert>
#include <iomanip>
#include <numeric>

namespace dd {

void getVectorFromDD(const vEdge& e, const std::complex<fp>& amp,
                     const std::size_t i, std::complex<dd::fp>* vec) {
  // calculate new accumulated amplitude
  const auto c = amp * static_cast<std::complex<fp>>(e.w);

  // base case
  if (e.isTerminal()) {
    vec[i] = c; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    return;
  }

  // recursive case
  if (!e.p->e[0].w.exactlyZero()) {
    getVectorFromDD(e.p->e[0], c, i, vec);
  }
  if (!e.p->e[1].w.exactlyZero()) {
    getVectorFromDD(e.p->e[1], c, i | (1ULL << e.p->v), vec);
  }
}

CVec getVectorFromDD(const Edge<vNode>& root) {
  assert(!root.isTerminal() && "Root must not be a terminal");
  const std::size_t dim = 2ULL << root.p->v;
  auto vec = CVec(dim, {0.0, 0.0});
  getVectorFromDD(root, 1, 0ULL, vec.data());
  return vec;
}

fp getVectorNorm(const CVec& vec) {
  return std::accumulate(
      vec.begin(), vec.end(), 0.0,
      [](const auto& sum, const auto& val) { return sum + std::norm(val); });
}

void printVector(const Edge<vNode>& e) {
  const std::size_t element = 2ULL << e.p->v;
  for (auto i = 0ULL; i < element; i++) {
    const auto amplitude = getValueByIndex(e, i);
    const auto n = static_cast<std::size_t>(e.p->v) + 1U;
    for (auto j = n; j > 0; --j) {
      std::cout << ((i >> (j - 1)) & 1ULL);
    }
    constexpr auto precision = 3;
    // set fixed width to maximum of a printed number
    // (-) 0.precision plus/minus 0.precision i
    constexpr auto width = 1 + 2 + precision + 1 + 2 + precision + 1;
    std::cout << ": " << std::setw(width)
              << ComplexValue::toString(amplitude.real(), amplitude.imag(),
                                        false, precision)
              << "\n";
  }
  std::cout << std::flush;
}

std::complex<fp> getValueByIndex(const Edge<vNode>& e, const std::size_t i) {
  if (e.isTerminal()) {
    return static_cast<std::complex<fp>>(e.w);
  }
  return getValueByIndex(e, 1, i);
}

std::complex<fp> getValueByIndex(const Edge<vNode>& e,
                                 const std::complex<fp>& amp,
                                 const std::size_t i) {
  const auto c = amp * static_cast<std::complex<fp>>(e.w);
  if (e.isTerminal()) {
    return c;
  }

  const bool zero = (i & (1ULL << e.p->v)) == 0U;
  if (zero) {
    if (!e.p->e[0].w.exactlyZero()) {
      return getValueByIndex(e.p->e[0], c, i);
    }
  } else {
    if (!e.p->e[1].w.exactlyZero()) {
      return getValueByIndex(e.p->e[1], c, i);
    }
  }
  return 0.0;
}

} // namespace dd
