#pragma once

#include "dd/Complex.hpp"
#include "dd/ComplexNumbers.hpp"
#include "dd/DDDefinitions.hpp"
#include "dd/Edge.hpp"
#include "dd/Node.hpp"
#include "dd/RealNumber.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <queue>
#include <sstream>
#include <stack>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace dd {

inline std::string colorFromPhase(const Complex& a) {
  auto phase = dd::ComplexNumbers::arg(a) / (2 * dd::PI);
  if (phase < 0) {
    phase += 1.;
  }
  std::ostringstream oss{};
  oss << std::fixed << std::setprecision(3) << phase << " " << 0.667 << " "
      << 0.75;
  return oss.str();
}

inline fp thicknessFromMagnitude(const Complex& a) {
  return 3.0 * std::max(dd::ComplexNumbers::mag(a), 0.10);
}

static void printPhaseFormatted(std::ostream& os, fp r) {
  const auto tol = dd::RealNumber::eps;

  r /= dd::PI;
  // special case treatment for +-i
  os << "ℯ(" << (std::signbit(r) ? "-" : "") << "iπ";

  const auto absr = std::abs(r);
  auto fraction = ComplexValue::getLowestFraction(absr);
  auto approx =
      static_cast<fp>(fraction.first) / static_cast<fp>(fraction.second);
  auto error = std::abs(absr - approx);

  if (error < tol) { // suitable fraction a/b found
    if (fraction.first == 1U && fraction.second == 1U) {
      os << ")";
    } else if (fraction.second == 1U) {
      os << " " << fraction.first << ")";
    } else if (fraction.first == 1U) {
      os << "/" << fraction.second << ")";
    } else {
      os << " " << fraction.first << "/" << fraction.second << ")";
    }
    return;
  }

  auto abssqrt = absr / SQRT2_2;
  fraction = ComplexValue::getLowestFraction(abssqrt);
  approx = static_cast<fp>(fraction.first) / static_cast<fp>(fraction.second);
  error = std::abs(abssqrt - approx);

  if (error < tol) { // suitable fraction a/(b * sqrt(2)) found
    if (fraction.first == 1U && fraction.second == 1U) {
      os << "/√2)";
    } else if (fraction.second == 1U) {
      os << " " << fraction.first << "/√2)";
    } else if (fraction.first == 1U) {
      os << "/(" << fraction.second << "√2))";
    } else {
      os << " " << fraction.first << "/(" << fraction.second << "√2))";
    }
    return;
  }

  auto abspi = absr / PI;
  fraction = ComplexValue::getLowestFraction(abspi);
  approx = static_cast<fp>(fraction.first) / static_cast<fp>(fraction.second);
  error = std::abs(abspi - approx);

  if (error < tol) { // suitable fraction a/b π found
    if (fraction.first == 1U && fraction.second == 1U) {
      os << " π)";
    } else if (fraction.second == 1U) {
      os << " " << fraction.first << "π)";
    } else if (fraction.first == 1U) {
      os << " π/" << fraction.second << ")";
    } else {
      os << " " << fraction.first << "π/" << fraction.second << ")";
    }
    return;
  }

  // default
  os << " " << absr << ")";
}

inline std::string conditionalFormat(const Complex& a,
                                     bool formatAsPolar = true) {
  if (!formatAsPolar) {
    return a.toString();
  }

  const auto mag = ComplexNumbers::mag(a);
  const auto phase = ComplexNumbers::arg(a);

  if (RealNumber::approximatelyZero(mag)) {
    return "0";
  }

  std::ostringstream ss{};
  // magnitude is (almost) 1
  if (RealNumber::approximatelyOne(mag)) {
    if (RealNumber::approximatelyZero(phase)) {
      return "1";
    }
    if (RealNumber::approximatelyEquals(phase, dd::PI_2)) {
      return "i";
    }
    if (RealNumber::approximatelyEquals(phase, -dd::PI_2)) {
      return "-i";
    }
    if (RealNumber::approximatelyEquals(phase, dd::PI)) {
      return "-1";
    }
    printPhaseFormatted(ss, phase);
    return ss.str();
  }

  if (RealNumber::approximatelyEquals(std::abs(phase), dd::PI)) {
    ss << "-";
    dd::ComplexValue::printFormatted(ss, mag);
    return ss.str();
  }
  if (RealNumber::approximatelyZero(phase)) {
    dd::ComplexValue::printFormatted(ss, mag);
    return ss.str();
  }

  dd::ComplexValue::printFormatted(ss, mag);
  ss << " ";
  printPhaseFormatted(ss, phase);

  return ss.str();
}

template <class Node>
static std::ostream& header(const Edge<Node>& e, std::ostream& os,
                            bool edgeLabels, bool formatAsPolar = true) {
  os << "digraph \"DD\" {graph[];node[shape=plain];edge[arrowhead=none]\n";
  os << "root [label=\"\",shape=point,style=invis]\n";
  os << "t [label=<<font "
        "point-size=\"20\">1</"
        "font>>,shape=box,tooltip=\"1\",width=0.3,height=0.3]\n";
  auto toplabel = (reinterpret_cast<std::uintptr_t>(e.p) & 0x001fffffU) >> 1U;
  os << "root->";
  if (e.isTerminal()) {
    os << "t";
  } else {
    os << toplabel;
  }
  os << "[penwidth=\"" << thicknessFromMagnitude(e.w) << "\",tooltip=\""
     << conditionalFormat(e.w, formatAsPolar) << "\"";
  if (!e.w.approximatelyOne()) {
    os << ",style=dashed";
  }
  if (edgeLabels) {
    os << ",label=<<font point-size=\"8\">&nbsp;"
       << conditionalFormat(e.w, formatAsPolar) << "</font>>";
  }

  os << "]\n";

  return os;
}
template <class Node>
static std::ostream& coloredHeader(const Edge<Node>& e, std::ostream& os,
                                   bool edgeLabels, bool formatAsPolar = true) {
  os << "digraph \"DD\" {graph[];node[shape=plain];edge[arrowhead=none]\n";
  os << "root [label=\"\",shape=point,style=invis]\n";
  os << "t [label=<<font "
        "point-size=\"20\">1</"
        "font>>,shape=box,tooltip=\"1\",width=0.3,height=0.3]\n";

  auto toplabel = (reinterpret_cast<std::uintptr_t>(e.p) & 0x001fffffU) >> 1U;
  auto mag = thicknessFromMagnitude(e.w);
  auto color = colorFromPhase(e.w);
  os << "root->";
  if (e.isTerminal()) {
    os << "t";
  } else {
    os << toplabel;
  }
  os << "[penwidth=\"" << mag << "\",tooltip=\""
     << conditionalFormat(e.w, formatAsPolar) << "\",color=\"" << color << "\"";
  if (edgeLabels) {
    os << ",label=<<font point-size=\"8\">&nbsp;"
       << conditionalFormat(e.w, formatAsPolar) << "</font>>";
  }
  os << "]\n";
  return os;
}
template <class Node>
static std::ostream& memoryHeader(const Edge<Node>& e, std::ostream& os,
                                  bool edgeLabels) {
  os << "digraph \"DD\" {graph[];node[shape=plain];edge[arrowhead=none]\n";
  os << "root [label=\"\",shape=point,style=invis]\n";
  os << "t [label=<<font "
        "point-size=\"20\">1</"
        "font>>,shape=box,tooltip=\"1\",width=0.3,height=0.3]\n";

  auto toplabel = (reinterpret_cast<std::uintptr_t>(e.p) & 0x001fffffU) >> 1U;
  auto mag = thicknessFromMagnitude(e.w);
  auto color = colorFromPhase(e.w);
  os << "root->";
  if (e.isTerminal()) {
    os << "t";
  } else {
    os << toplabel;
  }
  os << "[penwidth=\"" << mag << "\",tooltip=\"" << e.w.toString(false, 4)
     << "\" color=\"" << color << "\"";
  if (edgeLabels) {
    os << ",label=<<font point-size=\"8\">&nbsp;[";
    if (e.w.exactlyZero()) {
      os << "0";
    } else if (e.w.exactlyOne()) {
      os << "1";
    } else {
      if (RealNumber::exactlyZero(e.w.r)) {
        os << "0";
      } else if (RealNumber::exactlySqrt2over2(e.w.r)) {
        os << "\xe2\x88\x9a\xc2\xbd";
      } else if (RealNumber::exactlyOne(e.w.r)) {
        os << "1";
      } else {
        os << std::hex << reinterpret_cast<std::uintptr_t>(e.w.r) << std::dec;
      }
      os << " ";
      if (RealNumber::exactlyZero(e.w.i)) {
        os << "0";
      } else if (RealNumber::exactlySqrt2over2(e.w.i)) {
        os << "\xe2\x88\x9a\xc2\xbd";
      } else if (RealNumber::exactlyOne(e.w.i)) {
        os << "1";
      } else {
        os << std::hex << reinterpret_cast<std::uintptr_t>(e.w.i) << std::dec;
      }
    }
    os << "]</font>>";
  }
  os << "]\n";
  return os;
}

[[maybe_unused]] static std::ostream&
modernNode(const mEdge& e, std::ostream& os, bool formatAsPolar = true) {
  auto nodelabel = (reinterpret_cast<std::uintptr_t>(e.p) & 0x001fffffU) >>
                   1U; // this allows for 2^20 (roughly 1e6) unique nodes
  os << nodelabel << "[label=<";
  os << R"(<font point-size="10"><table border="1" cellspacing="0" cellpadding="2" style="rounded">)";
  os << R"(<tr><td colspan="2" rowspan="2" port="0" href="javascript:;" border="0" tooltip=")"
     << conditionalFormat(e.p->e[0].w, formatAsPolar) << "\">"
     << (e.p->e[0].w.approximatelyZero()
             ? "&nbsp;0 "
             : "<font color=\"white\">&nbsp;0 </font>")
     << R"(</td><td sides="R"></td><td sides="L"></td>)"
     << R"(<td colspan="2" rowspan="2" port="1" href="javascript:;" border="0" tooltip=")"
     << conditionalFormat(e.p->e[1].w, formatAsPolar) << "\">"
     << (e.p->e[1].w.approximatelyZero()
             ? "&nbsp;0 "
             : "<font color=\"white\">&nbsp;0 </font>")
     << R"(</td></tr>)";
  os << R"(<tr><td sides="R"></td><td sides="L"></td></tr>)";
  os << R"(<tr><td colspan="2" sides="B"></td><td colspan="2" rowspan="2" border="0"><font point-size="24">q<sub><font point-size="16">)"
     << static_cast<std::size_t>(e.p->v)
     << R"(</font></sub></font></td><td colspan="2" sides="B"></td></tr>)";
  os << R"(<tr><td sides="T" colspan="2"></td><td sides="T" colspan="2"></td></tr>)";
  os << R"(<tr><td colspan="2" rowspan="2" port="2" href="javascript:;" border="0" tooltip=")"
     << conditionalFormat(e.p->e[2].w, formatAsPolar) << "\">"
     << (e.p->e[2].w.approximatelyZero()
             ? "&nbsp;0 "
             : "<font color=\"white\">&nbsp;0 </font>")
     << R"(</td><td sides="R"></td><td sides="L"></td>)"
     << R"(<td colspan="2" rowspan="2" port="3" href="javascript:;" border="0" tooltip=")"
     << conditionalFormat(e.p->e[3].w, formatAsPolar) << "\">"
     << (e.p->e[3].w.approximatelyZero()
             ? "&nbsp;0 "
             : "<font color=\"white\">&nbsp;0 </font>")
     << "</td></tr>";
  os << R"(<tr><td sides="R"></td><td sides="L"></td></tr>)";
  os << "</table></font>>,tooltip=\"q" << static_cast<std::size_t>(e.p->v)
     << "\"]\n";
  return os;
}
[[maybe_unused]] static std::ostream&
modernNode(const vEdge& e, std::ostream& os, bool formatAsPolar = true) {
  auto nodelabel = (reinterpret_cast<std::uintptr_t>(e.p) & 0x001fffffU) >>
                   1U; // this allows for 2^20 (roughly 1e6) unique nodes
  os << nodelabel << "[label=<";
  os << R"(<font point-size="8"><table border="1" cellspacing="0" cellpadding="0" style="rounded">)";
  os << R"(<tr><td colspan="2" border="0" cellpadding="1"><font point-size="20">q<sub><font point-size="12">)"
     << static_cast<std::size_t>(e.p->v)
     << R"(</font></sub></font></td></tr><tr>)";
  os << R"(<td height="6" width="14" port="0" tooltip=")"
     << conditionalFormat(e.p->e[0].w, formatAsPolar)
     << R"(" href="javascript:;" sides="RT">)"
     << (e.p->e[0].w.approximatelyZero()
             ? "&nbsp;0 "
             : R"(<font color="white">&nbsp;0 </font>)")
     << "</td>";
  os << R"(<td height="6" width="14" port="1" tooltip=")"
     << conditionalFormat(e.p->e[1].w, formatAsPolar)
     << R"(" href="javascript:;" sides="LT">)"
     << (e.p->e[1].w.approximatelyZero()
             ? "&nbsp;0 "
             : R"(<font color="white">&nbsp;0 </font>)")
     << "</td>";
  os << "</tr></table></font>>,tooltip=\"q" << static_cast<std::size_t>(e.p->v)
     << "\"]\n";
  return os;
}
[[maybe_unused]] static std::ostream&
classicNode(const mEdge& e, std::ostream& os, bool formatAsPolar = true) {
  auto nodelabel = (reinterpret_cast<std::uintptr_t>(e.p) & 0x001fffffU) >>
                   1U; // this allows for 2^20 (roughly 1e6) unique nodes
  os << nodelabel << "[shape=circle, width=0.53, fixedsize=true, label=<";
  os << R"(<font point-size="6"><table border="0" cellspacing="0" cellpadding="0">)";
  os << R"(<tr><td colspan="4"><font point-size="18">q<sub><font point-size="10">)"
     << static_cast<std::size_t>(e.p->v)
     << R"(</font></sub></font></td></tr><tr>)";
  os << R"(<td port="0" tooltip=")"
     << conditionalFormat(e.p->e[0].w, formatAsPolar)
     << R"(" href="javascript:;">)";
  if (e.p->e[0].w.approximatelyZero()) {
    os << R"(<font point-size="8">&nbsp;0 </font>)";
  } else {
    os << R"(<font color="white">&nbsp;0 </font>)";
  }
  os << "</td>";
  os << "<td></td><td></td>";
  os << R"(<td port="3" tooltip=")"
     << conditionalFormat(e.p->e[3].w, formatAsPolar)
     << R"(" href="javascript:;">)";
  if (e.p->e[3].w.approximatelyZero()) {
    os << R"(<font point-size="8">&nbsp;0 </font>)";
  } else {
    os << R"(<font color="white">&nbsp;0 </font>)";
  }
  os << "</td>";
  os << "</tr><tr><td></td>";
  os << R"(<td port="1" tooltip=")"
     << conditionalFormat(e.p->e[1].w, formatAsPolar)
     << R"(" href="javascript:;">)";
  if (e.p->e[1].w.approximatelyZero()) {
    os << R"(<font point-size="8">&nbsp;0 </font>)";
  } else {
    os << R"(<font color="white">&nbsp;0 </font>)";
  }
  os << "</td>";
  os << R"(<td port="2" tooltip=")"
     << conditionalFormat(e.p->e[2].w, formatAsPolar)
     << R"(" href="javascript:;">)";
  if (e.p->e[2].w.approximatelyZero()) {
    os << R"(<font point-size="8">&nbsp;0 </font>)";
  } else {
    os << R"(<font color="white">&nbsp;0 </font>)";
  }
  os << "</td>";
  os << "<td></td></tr></table></font>>,tooltip=\"q"
     << static_cast<std::size_t>(e.p->v) << "\"]\n";
  return os;
}
[[maybe_unused]] static std::ostream&
classicNode(const vEdge& e, std::ostream& os, bool formatAsPolar = true) {
  auto nodelabel = (reinterpret_cast<std::uintptr_t>(e.p) & 0x001fffffU) >>
                   1U; // this allows for 2^20 (roughly 1e6) unique nodes
  os << nodelabel << "[shape=circle, width=0.46, fixedsize=true, label=<";
  os << R"(<font point-size="6"><table border="0" cellspacing="0" cellpadding="0">)";
  os << R"(<tr><td colspan="2"><font point-size="18">q<sub><font point-size="10">)"
     << static_cast<std::size_t>(e.p->v)
     << R"(</font></sub></font></td></tr><tr>)";
  os << R"(<td port="0" tooltip=")"
     << conditionalFormat(e.p->e[0].w, formatAsPolar)
     << R"(" href="javascript:;">)";
  if (e.p->e[0].w.approximatelyZero()) {
    os << R"(<font point-size="10">&nbsp;0 </font>)";
  } else {
    os << R"(<font color="white">&nbsp;0 </font>)";
  }
  os << "</td>";
  os << R"(<td port="1" tooltip=")"
     << conditionalFormat(e.p->e[1].w, formatAsPolar)
     << R"(" href="javascript:;">)";
  if (e.p->e[1].w.approximatelyZero()) {
    os << R"(<font point-size="10">&nbsp;0 </font>)";
  } else {
    os << R"(<font color="white">&nbsp;0 </font>)";
  }
  os << "</td>";
  os << "</tr></table></font>>,tooltip=\"q" << static_cast<std::size_t>(e.p->v)
     << "\"]\n";
  return os;
}
template <class Node>
static std::ostream& memoryNode(const Edge<Node>& e, std::ostream& os) {
  constexpr std::size_t n = std::tuple_size_v<decltype(e.p->e)>;
  auto nodelabel = (reinterpret_cast<std::uintptr_t>(e.p) & 0x001fffffU) >>
                   1U; // this allows for 2^20 (roughly 1e6) unique nodes
  os << nodelabel << "[label=<";
  os << R"(<font point-size="10"><table border="1" cellspacing="0" cellpadding="2" style="rounded">)";
  os << R"(<tr><td colspan=")" << n << R"(" border="1" sides="B">)" << std::hex
     << reinterpret_cast<std::uintptr_t>(e.p) << std::dec
     << " ref: " << e.p->ref << "</td></tr>";
  os << "<tr>";
  for (std::size_t i = 0; i < n; ++i) {
    os << "<td port=\"" << i << R"(" href="javascript:;" border="0" tooltip=")"
       << e.p->e[i].w.toString(false, 4) << "\">";
    if (e.p->e[i].isZeroTerminal()) {
      os << "&nbsp;0 "
            "";
    } else {
      os << "<font color=\"white\">&nbsp;0 </font>";
    }
    os << "</td>";
  }
  os << "</tr>";
  os << "</table></font>>,tooltip=\"" << std::hex
     << reinterpret_cast<std::uintptr_t>(e.p) << "\"]\n"
     << std::dec;
  return os;
}

[[maybe_unused]] static std::ostream&
bwEdge(const mEdge& from, const mEdge& to, std::uint16_t idx, std::ostream& os,
       bool edgeLabels = false, bool classic = false,
       bool formatAsPolar = true) {
  auto fromlabel =
      (reinterpret_cast<std::uintptr_t>(from.p) & 0x001fffffU) >> 1U;
  auto tolabel = (reinterpret_cast<std::uintptr_t>(to.p) & 0x001fffffU) >> 1U;

  os << fromlabel << ":" << idx << ":";
  if (classic) {
    if (idx == 0) {
      os << "sw";
    } else if (idx == 1 || idx == 2) {
      os << "s";
    } else {
      os << "se";
    }
  } else {
    if (idx == 0) {
      os << "sw";
    } else if (idx == 1) {
      os << "se";
    } else {
      os << 's';
    }
  }
  os << "->";
  if (to.isTerminal()) {
    os << "t";
  } else {
    os << tolabel;
  }

  auto mag = thicknessFromMagnitude(to.w);
  os << "[penwidth=\"" << mag << "\",tooltip=\""
     << conditionalFormat(to.w, formatAsPolar) << "\"";
  if (!to.w.approximatelyOne()) {
    os << ",style=dashed";
  }
  if (edgeLabels) {
    os << ",label=<<font point-size=\"8\">&nbsp;"
       << conditionalFormat(to.w, formatAsPolar) << "</font>>";
  }
  os << "]\n";

  return os;
}
[[maybe_unused]] static std::ostream&
bwEdge(const vEdge& from, const vEdge& to, std::uint16_t idx, std::ostream& os,
       bool edgeLabels = false, [[maybe_unused]] bool classic = false,
       bool formatAsPolar = true) {
  auto fromlabel =
      (reinterpret_cast<std::uintptr_t>(from.p) & 0x001fffffU) >> 1U;
  auto tolabel = (reinterpret_cast<std::uintptr_t>(to.p) & 0x001fffffU) >> 1U;

  os << fromlabel << ":" << idx << ":";
  os << (idx == 0 ? "sw" : "se") << "->";
  if (to.isTerminal()) {
    os << "t";
  } else {
    os << tolabel;
  }

  auto mag = thicknessFromMagnitude(to.w);
  os << "[penwidth=\"" << mag << "\",tooltip=\""
     << conditionalFormat(to.w, formatAsPolar) << "\"";
  if (!to.w.approximatelyOne()) {
    os << ",style=dashed";
  }
  if (edgeLabels) {
    os << ",label=<<font point-size=\"8\">&nbsp;"
       << conditionalFormat(to.w, formatAsPolar) << "</font>>";
  }
  os << "]\n";

  return os;
}
[[maybe_unused]] static std::ostream&
coloredEdge(const mEdge& from, const mEdge& to, std::uint16_t idx,
            std::ostream& os, bool edgeLabels = false, bool classic = false,
            bool formatAsPolar = true) {
  auto fromlabel =
      (reinterpret_cast<std::uintptr_t>(from.p) & 0x001fffffU) >> 1U;
  auto tolabel = (reinterpret_cast<std::uintptr_t>(to.p) & 0x001fffffU) >> 1U;

  os << fromlabel << ":" << idx << ":";
  if (classic) {
    if (idx == 0) {
      os << "sw";
    } else if (idx == 1 || idx == 2) {
      os << "s";
    } else {
      os << "se";
    }
  } else {
    if (idx == 0) {
      os << "sw";
    } else if (idx == 1) {
      os << "se";
    } else {
      os << 's';
    }
  }
  os << "->";
  if (to.isTerminal()) {
    os << "t";
  } else {
    os << tolabel;
  }

  auto mag = thicknessFromMagnitude(to.w);
  auto color = colorFromPhase(to.w);
  os << "[penwidth=\"" << mag << "\",tooltip=\""
     << conditionalFormat(to.w, formatAsPolar) << "\" color=\"" << color
     << "\"";
  if (edgeLabels) {
    os << ",label=<<font point-size=\"8\">&nbsp;"
       << conditionalFormat(to.w, formatAsPolar) << "</font>>";
  }
  os << "]\n";

  return os;
}
[[maybe_unused]] static std::ostream&
coloredEdge(const vEdge& from, const vEdge& to, std::uint16_t idx,
            std::ostream& os, bool edgeLabels = false,
            [[maybe_unused]] bool classic = false, bool formatAsPolar = true) {
  auto fromlabel =
      (reinterpret_cast<std::uintptr_t>(from.p) & 0x001fffffU) >> 1U;
  auto tolabel = (reinterpret_cast<std::uintptr_t>(to.p) & 0x001fffffU) >> 1U;

  os << fromlabel << ":" << idx << ":";
  os << (idx == 0 ? "sw" : "se") << "->";
  if (to.isTerminal()) {
    os << "t";
  } else {
    os << tolabel;
  }

  auto mag = thicknessFromMagnitude(to.w);
  auto color = colorFromPhase(to.w);
  os << "[penwidth=\"" << mag << "\",tooltip=\""
     << conditionalFormat(to.w, formatAsPolar) << "\" color=\"" << color
     << "\"";
  if (edgeLabels) {
    os << ",label=<<font point-size=\"8\">&nbsp;"
       << conditionalFormat(to.w, formatAsPolar) << "</font>>";
  }
  os << "]\n";

  return os;
}
template <class Node>
static std::ostream& memoryEdge(const Edge<Node>& from, const Edge<Node>& to,
                                std::uint16_t idx, std::ostream& os,
                                bool edgeLabels = false) {
  auto fromlabel =
      (reinterpret_cast<std::uintptr_t>(from.p) & 0x001fffffU) >> 1U;
  auto tolabel = (reinterpret_cast<std::uintptr_t>(to.p) & 0x001fffffU) >> 1U;

  os << fromlabel << ":" << idx << ":s->";
  if (to.isTerminal()) {
    os << "t";
  } else {
    os << tolabel;
  }

  auto mag = thicknessFromMagnitude(to.w);
  auto color = colorFromPhase(to.w);
  os << "[penwidth=\"" << mag << "\",tooltip=\"" << to.w.toString(false, 4)
     << "\" color=\"" << color << "\"";
  if (edgeLabels) {
    os << ",label=<<font point-size=\"8\">&nbsp;[";
    if (to.w.exactlyOne()) {
      os << "1";
    } else {
      if (RealNumber::exactlyZero(to.w.r)) {
        os << "0";
      } else if (RealNumber::exactlySqrt2over2(to.w.r)) {
        os << "\xe2\x88\x9a\xc2\xbd";
      } else if (RealNumber::exactlyOne(to.w.r)) {
        os << "1";
      } else {
        os << std::hex << reinterpret_cast<std::uintptr_t>(to.w.r) << std::dec;
      }
      os << " ";
      if (RealNumber::exactlyZero(to.w.i)) {
        os << "0";
      } else if (RealNumber::exactlySqrt2over2(to.w.i)) {
        os << "\xe2\x88\x9a\xc2\xbd";
      } else if (RealNumber::exactlyOne(to.w.i)) {
        os << "1";
      } else {
        os << std::hex << reinterpret_cast<std::uintptr_t>(to.w.i) << std::dec;
      }
    }
    os << "]</font>>";
  }
  os << "]\n";

  return os;
}

template <class Node>
static void toDot(const Edge<Node>& e, std::ostream& os, bool colored = true,
                  bool edgeLabels = false, bool classic = false,
                  bool memory = false, bool formatAsPolar = true) {
  std::ostringstream oss{};
  // header, root and terminal declaration

  if (memory) {
    memoryHeader(e, oss, edgeLabels);
  } else if (colored) {
    coloredHeader(e, oss, edgeLabels, formatAsPolar);
  } else {
    header(e, oss, edgeLabels, formatAsPolar);
  }

  std::unordered_set<decltype(e.p)> nodes{};

  auto priocmp = [](const Edge<Node>* left, const Edge<Node>* right) {
    if (left->p == nullptr) {
      return right->p != nullptr;
    }
    if (right->p == nullptr) {
      return false;
    }
    return left->p->v < right->p->v;
  };

  std::priority_queue<const Edge<Node>*, std::vector<const Edge<Node>*>,
                      decltype(priocmp)>
      q(priocmp);
  q.push(&e);

  // bfs until finished
  while (!q.empty()) {
    auto node = q.top();
    q.pop();

    // base case
    if (node->isTerminal()) {
      continue;
    }

    // check if node has already been processed
    auto ret = nodes.emplace(node->p);
    if (!ret.second) {
      continue;
    }

    // node definition as HTML-like label (href="javascript:;" is used as
    // workaround to make tooltips work)
    if (memory) {
      memoryNode(*node, oss);
    } else if (classic) {
      classicNode(*node, oss, formatAsPolar);
    } else {
      modernNode(*node, oss, formatAsPolar);
    }

    // iterate over edges in reverse to guarantee correct processing order
    for (auto i = static_cast<std::int16_t>(node->p->e.size() - 1); i >= 0;
         --i) {
      auto& edge = node->p->e[static_cast<std::size_t>(i)];
      if ((!memory && edge.w.approximatelyZero()) || edge.w.exactlyZero()) {
        // potentially add zero stubs here
        continue;
      }

      // non-zero edge to be included
      q.push(&edge);

      if (memory) {
        memoryEdge(*node, edge, static_cast<std::uint16_t>(i), oss, edgeLabels);
      } else if (colored) {
        coloredEdge(*node, edge, static_cast<std::uint16_t>(i), oss, edgeLabels,
                    classic, formatAsPolar);
      } else {
        bwEdge(*node, edge, static_cast<std::uint16_t>(i), oss, edgeLabels,
               classic, formatAsPolar);
      }
    }
  }
  oss << "}\n";

  os << oss.str() << std::flush;
}

template <class Node>
[[maybe_unused]] static void
export2Dot(Edge<Node> basic, const std::string& outputFilename,
           bool colored = true, bool edgeLabels = false, bool classic = false,
           bool memory = false, bool show = true, bool formatAsPolar = true) {
  std::ofstream init(outputFilename);
  toDot(basic, init, colored, edgeLabels, classic, memory, formatAsPolar);
  init.close();

  if (show) {
    std::ostringstream oss;
    oss << "dot -Tsvg " << outputFilename << " -o "
        << outputFilename.substr(0, outputFilename.find_last_of('.')) << ".svg";
    const auto str =
        oss.str(); // required to avoid immediate deallocation of temporary
    const auto exitCode = std::system(str.c_str());
    if (exitCode != 0) {
      std::cerr << "Error: dot returned with exit code " << exitCode << "\n";
    }
  }
}

///
/// Serialization
/// Note: do not rely on the binary format being portable across different
/// architectures/platforms
///

[[maybe_unused]] static void serialize(const vEdge& basic, std::ostream& os,
                                       bool writeBinary = false) {
  if (writeBinary) {
    os.write(reinterpret_cast<const char*>(&SERIALIZATION_VERSION),
             sizeof(decltype(SERIALIZATION_VERSION)));
    basic.w.writeBinary(os);
  } else {
    os << SERIALIZATION_VERSION << "\n";
    os << basic.w.toString(false, 16) << "\n";
  }
  std::int64_t nextIndex = 0;
  std::unordered_map<vNode*, std::int64_t> nodeIndex{};

  // POST ORDER TRAVERSAL USING ONE STACK
  // https://www.geeksforgeeks.org/iterative-postorder-traversal-using-stack/
  std::stack<const vEdge*> stack{};

  const auto* node = &basic;
  if (!node->isTerminal()) {
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-do-while)
    do {
      while (node != nullptr && !node->isTerminal()) {
        for (auto i = static_cast<std::size_t>(RADIX - 1); i > 0; --i) {
          auto& edge = node->p->e.at(i);
          if (edge.isTerminal()) {
            continue;
          }
          if (edge.w.approximatelyZero()) {
            continue;
          }
          if (nodeIndex.find(edge.p) != nodeIndex.end()) {
            continue;
          }

          // non-zero edge to be included
          stack.push(&edge);
        }
        stack.push(node);
        node = &node->p->e[0]; // NOLINT(readability-container-data-pointer)
      }
      node = stack.top();
      stack.pop();

      bool hasChild = false;
      for (auto i = 1U; i < RADIX && !hasChild; ++i) {
        auto& edge = node->p->e.at(i);
        if (edge.w.approximatelyZero()) {
          continue;
        }
        if (nodeIndex.find(edge.p) != nodeIndex.end()) {
          continue;
        }
        if (!stack.empty()) {
          hasChild = edge.p == stack.top()->p;
        }
      }

      if (hasChild) {
        const auto* const temp = stack.top();
        stack.pop();
        stack.push(node);
        node = temp;
      } else {
        if (nodeIndex.find(node->p) != nodeIndex.end()) {
          node = nullptr;
          continue;
        }
        nodeIndex[node->p] = nextIndex;
        nextIndex++;

        if (writeBinary) {
          os.write(reinterpret_cast<const char*>(&nodeIndex[node->p]),
                   sizeof(decltype(nodeIndex[node->p])));
          os.write(reinterpret_cast<const char*>(&node->p->v),
                   sizeof(decltype(node->p->v)));

          // iterate over edges in reverse to guarantee correct processing order
          for (auto i = 0U; i < RADIX; ++i) {
            auto& edge = node->p->e.at(i);
            std::int64_t edgeIdx = edge.isTerminal() ? -1 : nodeIndex[edge.p];
            os.write(reinterpret_cast<const char*>(&edgeIdx),
                     sizeof(decltype(edgeIdx)));
            edge.w.writeBinary(os);
          }
        } else {
          os << nodeIndex[node->p] << " "
             << static_cast<std::size_t>(node->p->v);

          // iterate over edges in reverse to guarantee correct processing order
          for (auto i = 0U; i < RADIX; ++i) {
            os << " (";
            auto& edge = node->p->e.at(i);
            if (!edge.w.approximatelyZero()) {
              const std::int64_t edgeIdx =
                  edge.isTerminal() ? -1 : nodeIndex[edge.p];
              os << edgeIdx << " " << edge.w.toString(false, 16);
            }
            os << ")";
          }
          os << "\n";
        }
        node = nullptr;
      }
    } while (!stack.empty());
  }
}

static void serializeMatrix(const mEdge& basic, std::int64_t& idx,
                            std::unordered_map<mNode*, std::int64_t>& nodeIndex,
                            std::unordered_set<mNode*>& visited,
                            std::ostream& os, bool writeBinary = false) {
  if (!basic.isTerminal()) {
    for (auto& e : basic.p->e) {
      if (visited.insert(e.p).second) {
        serializeMatrix(e, idx, nodeIndex, visited, os, writeBinary);
      }
    }

    if (nodeIndex.find(basic.p) == nodeIndex.end()) {
      nodeIndex[basic.p] = idx;
      ++idx;
    }

    if (writeBinary) {
      os.write(reinterpret_cast<const char*>(&nodeIndex[basic.p]),
               sizeof(decltype(nodeIndex[basic.p])));
      os.write(reinterpret_cast<const char*>(&basic.p->v),
               sizeof(decltype(basic.p->v)));

      // iterate over edges in reverse to guarantee correct processing order
      for (auto& edge : basic.p->e) {
        std::int64_t edgeIdx = edge.isTerminal() ? -1 : nodeIndex[edge.p];
        os.write(reinterpret_cast<const char*>(&edgeIdx),
                 sizeof(decltype(edgeIdx)));
        edge.w.writeBinary(os);
      }
    } else {
      os << nodeIndex[basic.p] << " " << static_cast<std::size_t>(basic.p->v);

      // iterate over edges in reverse to guarantee correct processing order
      for (auto& edge : basic.p->e) {
        os << " (";
        if (!edge.w.approximatelyZero()) {
          const std::int64_t edgeIdx =
              edge.isTerminal() ? -1 : nodeIndex[edge.p];
          os << edgeIdx << " " << edge.w.toString(false, 16);
        }
        os << ")";
      }
      os << "\n";
    }
  }
}
[[maybe_unused]] static void serialize(const mEdge& basic, std::ostream& os,
                                       bool writeBinary = false) {
  if (writeBinary) {
    os.write(reinterpret_cast<const char*>(&SERIALIZATION_VERSION),
             sizeof(decltype(SERIALIZATION_VERSION)));
    basic.w.writeBinary(os);
  } else {
    os << SERIALIZATION_VERSION << "\n";
    os << basic.w.toString(false, std::numeric_limits<dd::fp>::max_digits10)
       << "\n";
  }
  std::int64_t idx = 0;
  std::unordered_map<mNode*, std::int64_t> nodeIndex{};
  std::unordered_set<mNode*> visited{};
  serializeMatrix(basic, idx, nodeIndex, visited, os, writeBinary);
}
template <class Node>
static void serialize(const Edge<Node>& basic,
                      const std::string& outputFilename,
                      bool writeBinary = false) {
  std::ofstream ofs = std::ofstream(outputFilename, std::ios::binary);

  if (!ofs.good()) {
    throw std::invalid_argument("Cannot open file: " + outputFilename);
  }

  serialize(basic, ofs, writeBinary);
}

template <typename Node>
static void exportEdgeWeights(const Edge<Node>& edge, std::ostream& stream) {
  struct Priocmp {
    bool operator()(const Edge<Node>* left, const Edge<Node>* right) {
      if (left->p == nullptr) {
        return right->p != nullptr;
      }
      if (right->p == nullptr) {
        return false;
      }
      return left->p->v < right->p->v;
    }
  };
  stream << std::showpos << RealNumber::val(edge.w.r)
         << RealNumber::val(edge.w.i) << std::noshowpos << "i\n";

  std::unordered_set<decltype(edge.p)> nodes{};

  std::priority_queue<const Edge<Node>*, std::vector<const Edge<Node>*>,
                      Priocmp>
      q;
  q.push(&edge);

  // bfs until finished
  while (!q.empty()) {
    auto edgePtr = q.top();
    q.pop();

    // base case
    if (edgePtr->isTerminal()) {
      continue;
    }

    // check if edgePtr has already been processed
    if (auto ret = nodes.emplace(edgePtr->p); !ret.second) {
      continue;
    }

    // iterate over edges in reverse to guarantee correct processing order
    for (auto i = static_cast<std::int16_t>(edgePtr->p->e.size() - 1); i >= 0;
         --i) {
      auto& child = edgePtr->p->e[static_cast<std::size_t>(i)];
      if (child.w.approximatelyZero()) {
        // potentially add zero stubs here
        continue;
      }

      // non-zero child to be included
      q.push(&child);
      stream << std::showpos << RealNumber::val(child.w.r)
             << RealNumber::val(child.w.i) << std::noshowpos << "i\n";
    }
  }
}

} // namespace dd
