//
// This file is part of the MQT QMAP library released under the MIT license.
// See README.md or go to https://github.com/cda-tum/qmap for more information.
//

#pragma once

#include "../Definitions.hpp"
#include "na/Definitions.hpp"
#include "operations/OpType.hpp"
#include "operations/Operation.hpp"

#include <iterator>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace na {
class ShuttlingOperation : public qc::Operation {
public:
  class SingleAtomMove {
  private:
    qc::Qubit start;
    qc::Qubit end;

  public:
    SingleAtomMove(const qc::Qubit& s, const qc::Qubit& e) : start(s), end(e) {}
    [[nodiscard]] qc::Qubit getStart() const { return start; }
    [[nodiscard]] qc::Qubit getEnd() const { return end; }
  };

private:
  std::vector<SingleAtomMove> moves;
  std::map<qc::Qubit, Point> mapping;

public:
  explicit ShuttlingOperation(const qc::Targets& starts,
                              const qc::Targets& ends,
                              std::map<qc::Qubit, Point>& m)
      : mapping(m) {
    if (starts.size() != ends.size()) {
      throw std::logic_error("Shuttling operation must have the same number of "
                             "start and end qubits.");
    }
    nqubits = starts.size();
    name = "Shuttling operation";
    type = qc::Shuttle;
    std::vector<SingleAtomMove> tmpMoves = {};
    for (std::size_t i = 0; i < nqubits; ++i) {
      auto move = SingleAtomMove(starts[i], ends[i]);
      if (isCompatible(tmpMoves, move, mapping)) {
        tmpMoves.emplace_back(move);
      } else {
        throw std::logic_error(
            "Shuttling operation are not compatible with each other.");
      }
    }
    moves = tmpMoves;
  }

  void addControl([[maybe_unused]] const qc::Control c) override {
    throw std::logic_error("shuttling does not support controls.");
  }

  void clearControls() override {
    throw std::logic_error("shuttling does not support controls.");
  }

  void removeControl([[maybe_unused]] const qc::Control c) override {
    throw std::logic_error("shuttling does not support controls.");
  }

  qc::Controls::iterator
  removeControl([[maybe_unused]] const qc::Controls::iterator it) override {
    throw std::logic_error("shuttling does not support controls.");
  }

  [[nodiscard]] std::set<qc::Qubit> getUsedQubits() const override {
    std::set<qc::Qubit> usedQubits;
    for (const auto& move : moves) {
      usedQubits.insert(move.getStart());
    }
    return usedQubits;
  }

  [[nodiscard]] qc::Qubit movesTo(const qc::Qubit& q) const {
    for (const auto& move : moves) {
      if (move.getStart() == q) {
        return move.getEnd();
      }
    }
    std::stringstream ss;
    ss << "Qubit " << q << " is not moved by this shuttling operation.";
    throw std::invalid_argument(ss.str());
  }

  /**
   * @brief Emplaces a new shuttling operation if it is compatible with the
   * existing moves.
   * @param move the new move
   * @throw std::logic_error if the new move is not compatible with the existing
   */
  void emplace(const SingleAtomMove& move) {
    if (isCompatible(moves, move, mapping)) {
      moves.emplace_back(move);
    } else {
      throw std::logic_error(
          "Shuttling operation is not compatible with the existing moves.");
    }
  }

  /**
   * @brief Try to emplace a new shuttling operation if it is compatible with
   * the existing moves.
   *
   * @param move the new move
   * @return true if the move is compatible
   * @return false if the move is not compatible
   */
  bool tryEmplace(const SingleAtomMove& move) noexcept {
    if (isCompatible(moves, move, mapping)) {
      moves.emplace_back(move);
      return true;
    }
    return false;
  }

  /**
   * @brief Checks whether the single new move is compatible with the existing
   * moves.
   * @details A shuttling operation is compatible with another shuttling
   * operation if the topological order of the moved atoms does not changed.
   *
   * @param moves
   * @param move
   * @param mapping
   * @return true
   * @return false
   */
  [[nodiscard]] static bool
  isCompatible(const std::vector<SingleAtomMove>& moves,
               const SingleAtomMove& move,
               std::map<qc::Qubit, Point>& mapping);

  [[nodiscard]] bool
  isCompatible(const SingleAtomMove& move) {
    return isCompatible(moves, move, mapping);
  }
};
} // namespace na