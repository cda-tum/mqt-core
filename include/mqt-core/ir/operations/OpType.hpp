/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>

namespace qc {

enum OpTypeFlags : uint8_t {
  OpTypeNone = 0b00,
  OpTypeInv = 0b01,
  OpTypeDiag = 0b10,
};

constexpr static unsigned NUM_OP_TYPE_FLAG_BITS = 2;

// Natively supported operations of the QFR library
enum OpType : std::uint8_t {
#define HANDLE_OP_TYPE(N, id, flags, repr)                                     \
  id = ((N) << (NUM_OP_TYPE_FLAG_BITS)) | (flags),
#define LAST_OP_TYPE(N) OpTypeEnd = (N) << (NUM_OP_TYPE_FLAG_BITS),
#include "OpType.inc"
#undef HANDLE_OP_TYPE
#undef LAST_OP_TYPE
};

inline std::string toString(const OpType& opType) {
  static const std::unordered_map<OpType, std::string_view> OP_NAMES{
#define HANDLE_OP_TYPE(N, id, flags, repr) {id, {repr}},
#define LAST_OP_TYPE(N)
#include "OpType.inc"
#undef HANDLE_OP_TYPE
#undef LAST_OP_TYPE
  };

  if (const auto it = OP_NAMES.find(opType); it != OP_NAMES.end()) {
    return std::string(it->second);
  }
  throw std::invalid_argument("Invalid OpType!");
}

/**
 * @brief Gives a short name for the given OpType (at most 3 characters)
 * @param opType OpType to get the short name for
 * @return Short name for the given OpType
 */
inline std::string shortName(const OpType& opType) {
  switch (opType) {
  case GPhase:
    return "GPh";
  case SXdg:
    return "sxd";
  case SWAP:
    return "sw";
  case iSWAP:
    return "isw";
  case iSWAPdg:
    return "isd";
  case Peres:
    return "pr";
  case Peresdg:
    return "prd";
  case XXminusYY:
    return "x-y";
  case XXplusYY:
    return "x+y";
  case Barrier:
    return "====";
  case Measure:
    return "msr";
  case Reset:
    return "rst";
  case Teleportation:
    return "tel";
  default:
    return toString(opType);
  }
}

[[nodiscard]] constexpr bool isTwoQubitGate(const OpType& opType) {
  switch (opType) {
  case SWAP:
  case iSWAP:
  case iSWAPdg:
  case Peres:
  case Peresdg:
  case DCX:
  case ECR:
  case RXX:
  case RYY:
  case RZZ:
  case RZX:
  case XXminusYY:
  case XXplusYY:
    return true;
  default:
    return false;
  }
}

[[nodiscard]] constexpr bool isSingleQubitGate(const qc::OpType& type) {
  switch (type) {
  case I:
  case U:
  case U2:
  case P:
  case X:
  case Y:
  case Z:
  case H:
  case S:
  case Sdg:
  case T:
  case SX:
  case SXdg:
  case Tdg:
  case V:
  case Vdg:
  case RX:
  case RY:
  case RZ:
    return true;
  default:
    return false;
  }
}

inline std::ostream& operator<<(std::ostream& out, const OpType& opType) {
  return out << toString(opType);
}

const inline static std::unordered_map<std::string, qc::OpType>
    OP_NAME_TO_TYPE = {
        {"none", OpType::None},
        {"gphase", OpType::GPhase},
        {"i", OpType::I},
        {"id", OpType::I},
        {"h", OpType::H},
        {"ch", OpType::H},
        {"x", OpType::X},
        {"cnot", OpType::X},
        {"cx", OpType::X},
        {"mcx", OpType::X},
        {"y", OpType::Y},
        {"cy", OpType::Y},
        {"z", OpType::Z},
        {"cz", OpType::Z},
        {"s", OpType::S},
        {"cs", OpType::S},
        {"sdg", OpType::Sdg},
        {"csdg", OpType::Sdg},
        {"t", OpType::T},
        {"ct", OpType::T},
        {"tdg", OpType::Tdg},
        {"ctdg", OpType::Tdg},
        {"v", OpType::V},
        {"vdg", OpType::Vdg},
        {"u", OpType::U},
        {"cu", OpType::U},
        {"u3", OpType::U},
        {"cu3", OpType::U},
        {"u2", OpType::U2},
        {"cu2", OpType::U2},
        {"p", OpType::P},
        {"cp", OpType::P},
        {"mcp", OpType::P},
        {"phase", OpType::P},
        {"cphase", OpType::P},
        {"mcphase", OpType::P},
        {"u1", OpType::P},
        {"cu1", OpType::P},
        {"sx", OpType::SX},
        {"csx", OpType::SX},
        {"sxdg", OpType::SXdg},
        {"csxdg", OpType::SXdg},
        {"rx", OpType::RX},
        {"crx", OpType::RX},
        {"ry", OpType::RY},
        {"cry", OpType::RY},
        {"rz", OpType::RZ},
        {"crz", OpType::RZ},
        {"swap", OpType::SWAP},
        {"cswap", OpType::SWAP},
        {"iswap", OpType::iSWAP},
        {"iswapdg", OpType::iSWAPdg},
        {"peres", OpType::Peres},
        {"peresdg", OpType::Peresdg},
        {"dcx", OpType::DCX},
        {"ecr", OpType::ECR},
        {"rxx", OpType::RXX},
        {"ryy", OpType::RYY},
        {"rzz", OpType::RZZ},
        {"rzx", OpType::RZX},
        {"xx_minus_yy", OpType::XXminusYY},
        {"xx_plus_yy", OpType::XXplusYY},
        {"measure", OpType::Measure},
        {"reset", OpType::Reset},
        {"barrier", OpType::Barrier},
        {"teleportation", OpType::Teleportation},
        {"classic_controlled", OpType::ClassicControlled},
        {"compound", OpType::Compound},
        {"move", OpType::Move},
        {"aod_activate", OpType::AodActivate},
        {"aod_deactivate", OpType::AodDeactivate},
        {"aod_move", OpType::AodMove},
};

[[nodiscard]] inline OpType opTypeFromString(const std::string& opType) {
  // try to find the operation type in the map of known operation types and
  // return it if found or throw an exception otherwise.
  if (const auto it = OP_NAME_TO_TYPE.find(opType);
      it != OP_NAME_TO_TYPE.end()) {
    return OP_NAME_TO_TYPE.at(opType);
  }
  throw std::invalid_argument("Unsupported operation type: " + opType);
}

inline std::istream& operator>>(std::istream& in, OpType& opType) {
  std::string opTypeStr;
  in >> opTypeStr;

  if (opTypeStr.empty()) {
    in.setstate(std::istream::failbit);
    return in;
  }

  opType = opTypeFromString(opTypeStr);
  return in;
}

} // namespace qc
