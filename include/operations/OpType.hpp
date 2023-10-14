#pragma once

#include <cstdint>
#include <functional>
#include <iostream>
#include <string>
#include <unordered_map>

namespace qc {
// Natively supported operations of the QFR library
enum OpType : std::uint8_t {
  None,
  // Standard Operations
  GPhase,
  I,
  Barrier,
  H,
  X,
  Y,
  Z,
  S,
  Sdg,
  T,
  Tdg,
  V,
  Vdg,
  U,
  U2,
  P,
  SX,
  SXdg,
  RX,
  RY,
  RZ,
  SWAP,
  iSWAP, // NOLINT (readability-identifier-naming)
  Peres,
  Peresdg,
  DCX,
  ECR,
  RXX,
  RYY,
  RZZ,
  RZX,
  XXminusYY,
  XXplusYY,
  // Compound Operation
  Compound,
  // Non Unitary Operations
  Measure,
  Reset,
  Teleportation,
  // Classically-controlled Operation
  ClassicControlled,
  // Noise operations
  ATrue,
  AFalse,
  MultiATrue,
  MultiAFalse,
  // Number of OpTypes
  OpCount
};

inline std::string toString(const OpType& opType) {
  switch (opType) {
  case None:
    return "none";
  case GPhase:
    return "gphase";
  case I:
    return "i";
  case H:
    return "h";
  case X:
    return "x";
  case Y:
    return "y";
  case Z:
    return "z";
  case S:
    return "s";
  case Sdg:
    return "sdg";
  case T:
    return "t";
  case Tdg:
    return "tdg";
  case V:
    return "v";
  case Vdg:
    return "vdg";
  case U:
    return "u";
  case U2:
    return "u2";
  case P:
    return "p";
  case SX:
    return "sx";
  case SXdg:
    return "sxdg";
  case RX:
    return "rx";
  case RY:
    return "ry";
  case RZ:
    return "rz";
  case SWAP:
    return "swap";
  case iSWAP:
    return "iswap";
  case Peres:
    return "peres";
  case Peresdg:
    return "peresdg";
  case DCX:
    return "dcx";
  case ECR:
    return "ecr";
  case RXX:
    return "rxx";
  case RYY:
    return "ryy";
  case RZZ:
    return "rzz";
  case RZX:
    return "rzx";
  case XXminusYY:
    return "xx_minus_yy";
  case XXplusYY:
    return "xx_plus_yy";
  case Compound:
    return "compound";
  case Measure:
    return "measure";
  case Reset:
    return "reset";
  case Barrier:
    return "barrier";
  case Teleportation:
    return "teleportation";
  case ClassicControlled:
    return "classic_controlled";
  // GCOV_EXCL_START
  default:
    throw std::invalid_argument("Invalid OpType!");
    // GCOV_EXCL_STOP
  }
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

inline bool isTwoQubitGate(const OpType& opType) {
  switch (opType) {
  case SWAP:
  case iSWAP:
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

inline std::ostream& operator<<(std::ostream& out, OpType& opType) {
  out << toString(opType);
  return out;
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
