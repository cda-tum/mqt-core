//
// This file is part of the MQT QMAP library released under the MIT license.
// See README.md or go to https://github.com/cda-tum/qmap for more information.
//

#include "na/ShuttlingOperation.hpp"

namespace na {
[[nodiscard]] bool
  ShuttlingOperation::isCompatible(const std::vector<SingleAtomMove>& moves,
               const SingleAtomMove& move,
               std::map<qc::Qubit, Point>& mapping) {
    Point const moveStart = mapping.at(move.getStart());
    Point const moveEnd = mapping.at(move.getEnd());
    return std::all_of(moves.begin(), moves.end(), [&](const auto& m) {
      Point const mStart = mapping.at(m.getStart());
      Point const mEnd = mapping.at(m.getEnd());
      // the topological order of the moves is not changed
      // Note that (A ==> B) == (!A || B)
      return (moveStart.x > mStart.x || moveEnd.x <= mEnd.x) &&
             (moveStart.x < mStart.x || moveEnd.x >= mEnd.x) &&
             (moveStart.y > mStart.y || moveEnd.y <= mEnd.y) &&
             (moveStart.y < mStart.y || moveEnd.y >= mEnd.y);
    });
  }
} // namespace na