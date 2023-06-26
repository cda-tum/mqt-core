#include "dd/Node.hpp"

namespace dd {
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables
vNode vNode::terminal{
    {{{nullptr, Complex::zero}, {nullptr, Complex::zero}}}, nullptr, 0U, -1};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
mNode mNode::terminal{{{{nullptr, Complex::zero},
                        {nullptr, Complex::zero},
                        {nullptr, Complex::zero},
                        {nullptr, Complex::zero}}},
                      nullptr,
                      0U,
                      -1,
                      32 + 16};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
dNode dNode::terminal{{{{nullptr, Complex::zero},
                        {nullptr, Complex::zero},
                        {nullptr, Complex::zero},
                        {nullptr, Complex::zero}}},
                      nullptr,
                      0,
                      -1,
                      0};
} // namespace dd
