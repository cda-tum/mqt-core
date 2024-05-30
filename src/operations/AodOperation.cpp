#include "operations/AodOperation.hpp"

#include "Definitions.hpp"
#include "cassert"
#include "iomanip"
#include "limits"
#include "operations/OpType.hpp"

#include <cstddef>
#include <cstdint>
#include <ios>
#include <ostream>
#include <tuple>
#include <utility>
#include <vector>

namespace qc {
AodOperation::AodOperation(OpType s, std::vector<Qubit> qubits,
                           const std::vector<uint32_t>& dirs,
                           const std::vector<fp>& start,
                           const std::vector<fp>& end)
    : AodOperation(s, std::move(qubits), convertToDimension(dirs), start, end) {
}

std::vector<Dimension>
AodOperation::convertToDimension(const std::vector<uint32_t>& dirs) {
  std::vector<Dimension> dirsEnum(dirs.size());
  for (size_t i = 0; i < dirs.size(); ++i) {
    dirsEnum[i] = static_cast<Dimension>(dirs[i]);
  }
  return dirsEnum;
}

AodOperation::AodOperation(const OpType s, std::vector<Qubit> qubits,
                           const std::vector<Dimension>& dirs,
                           const std::vector<fp>& start,
                           const std::vector<fp>& end) {
  assert(dirs.size() == start.size() && start.size() == end.size());
  type = s;
  targets = std::move(qubits);
  name = toString(type);

  for (size_t i = 0; i < dirs.size(); ++i) {
    operations.emplace_back(dirs[i], start[i], end[i]);
  }
}

AodOperation::AodOperation(const std::string& type, std::vector<Qubit> targets,
                           const std::vector<uint32_t>& dirs,
                           const std::vector<fp>& start,
                           const std::vector<fp>& end)
    : AodOperation(OP_NAME_TO_TYPE.at(type), std::move(targets),
                   convertToDimension(dirs), start, end) {}

AodOperation::AodOperation(
    OpType s, std::vector<Qubit> targets,
    const std::vector<std::tuple<Dimension, fp, fp>>& operations) {
  type = s;
  this->targets = std::move(targets);
  name = toString(type);

  for (const auto& [dir, index, param] : operations) {
    this->operations.emplace_back(dir, index, param);
  }
}

AodOperation::AodOperation(OpType s, std::vector<Qubit> t,
                           std::vector<SingleOperation> ops)
    : operations(std::move(ops)) {
  type = s;
  this->targets = std::move(t);
  name = toString(type);
}

void AodOperation::dumpOpenQASM(std::ostream& of, const RegisterNames& qreg,
                                [[maybe_unused]] const RegisterNames& creg,
                                size_t indent, bool /*openQASM3*/) const {
  of << std::setprecision(std::numeric_limits<fp>::digits10);
  of << std::string(indent * OUTPUT_INDENT_SIZE, ' ');
  of << name;
  // write AOD operations
  of << " (";
  for (const auto& op : operations) {
    of << op.toQASMString();
  }
  // remove last semicolon
  of.seekp(-1, std::ios_base::end);
  of << ")";
  // write qubit start
  for (const auto& qubit : targets) {
    of << " " << qreg[qubit].second << ",";
  }
  of.seekp(-1, std::ios_base::end);
  of << ";\n";
}

void AodOperation::invert() {
  if (type == OpType::AodMove) {
    for (auto& op : operations) {
      std::swap(op.start, op.end);
    }
  } else if (type == OpType::AodActivate) {
    type = OpType::AodDeactivate;
  } else if (type == OpType::AodDeactivate) {
    type = OpType::AodActivate;
  }
}

} // namespace qc
