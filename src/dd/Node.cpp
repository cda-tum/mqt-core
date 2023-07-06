#include "dd/Node.hpp"

#include "dd/RealNumber.hpp"

namespace dd {
// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables,
// cppcoreguidelines-interfaces-global-init)
vNode vNode::terminal{
    {{{nullptr, Complex::zero}, {nullptr, Complex::zero}}}, nullptr, 0U, -1};

mNode mNode::terminal{{{{nullptr, Complex::zero},
                        {nullptr, Complex::zero},
                        {nullptr, Complex::zero},
                        {nullptr, Complex::zero}}},
                      nullptr,
                      0U,
                      -1,
                      32 + 16};

dNode dNode::terminal{{{{nullptr, Complex::zero},
                        {nullptr, Complex::zero},
                        {nullptr, Complex::zero},
                        {nullptr, Complex::zero}}},
                      nullptr,
                      0,
                      -1,
                      0};
// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables,
// cppcoreguidelines-interfaces-global-init)

void dNode::setDensityMatrixNodeFlag(const bool densityMatrix) {
  if (densityMatrix) {
    flags = (flags | static_cast<std::uint8_t>(8U));
  } else {
    flags = (flags & static_cast<std::uint8_t>(~8U));
  }
}

std::uint8_t dNode::alignDensityNodeNode(dNode*& p) {
  const auto flags = static_cast<std::uint8_t>(getDensityMatrixTempFlags(p));
  alignDensityNode(p);

  if (p == nullptr || p->v <= -1) {
    return 0;
  }

  if (isNonReduceTempFlagSet(flags) && !isConjugateTempFlagSet(flags)) {
    // first edge paths are not modified and the property is inherited by all
    // child paths
    return flags;
  }
  if (!isConjugateTempFlagSet(flags)) {
    // Conjugate the second edge (i.e. negate the complex part of the second
    // edge)
    p->e[2].w.i = RealNumber::flipPointerSign(p->e[2].w.i);
    setConjugateTempFlagTrue(p->e[2].p);
    // Mark the first edge
    setNonReduceTempFlagTrue(p->e[1].p);

    for (auto& edge : p->e) {
      setDensityMatTempFlagTrue(edge.p);
    }

  } else {
    std::swap(p->e[2], p->e[1]);
    for (auto& edge : p->e) {
      // Conjugate all edges
      edge.w.i = RealNumber::flipPointerSign(edge.w.i);
      setConjugateTempFlagTrue(edge.p);
      setDensityMatTempFlagTrue(edge.p);
    }
  }
  return flags;
}

void dNode::getAlignedNodeRevertModificationsOnSubEdges(dNode* p) {
  // Before I do anything else, I must align the pointer
  alignDensityNode(p);

  for (auto& edge : p->e) {
    // remove the set properties from the node pointers of edge.p->e
    alignDensityNode(edge.p);
  }

  if (isNonReduceTempFlagSet(p->flags) && !isConjugateTempFlagSet(p->flags)) {
    // first edge paths are not modified I only have to remove the first edge
    // property
    return;
  }

  if (!isConjugateTempFlagSet(p->flags)) {
    // Conjugate the second edge (i.e. negate the complex part of the second
    // edge)
    p->e[2].w.i = RealNumber::flipPointerSign(p->e[2].w.i);
    return;
  }
  for (auto& edge : p->e) {
    // Align all nodes and conjugate the weights
    edge.w.i = RealNumber::flipPointerSign(edge.w.i);
  }
  std::swap(p->e[2], p->e[1]);
}

void dNode::applyDmChangesToNode(dNode*& p) {
  // Align the node pointers
  if (isDensityMatrixTempFlagSet(p)) {
    auto tmp = alignDensityNodeNode(p);
    assert(getDensityMatrixTempFlags(p->flags) == 0);
    p->flags = p->flags | tmp;
  }
}

void dNode::revertDmChangesToNode(dNode*& p) {
  // Align the node pointers
  if (isDensityMatrixTempFlagSet(p->flags)) {
    getAlignedNodeRevertModificationsOnSubEdges(p);
    p->unsetTempDensityMatrixFlags();
  }
}

} // namespace dd
