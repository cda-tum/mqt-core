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

namespace na {
AodOperation::AodOperation(qc::OpType s, std::vector<qc::Qubit> qubits,
                           const std::vector<uint32_t>& dirs,
                           const std::vector<qc::fp>& start,
                           const std::vector<qc::fp>& end)
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

AodOperation::AodOperation(const qc::OpType s, std::vector<qc::Qubit> qubits,
                           const std::vector<Dimension>& dirs,
                           const std::vector<qc::fp>& start,
                           const std::vector<qc::fp>& end) {
  assert(dirs.size() == start.size() && start.size() == end.size());
  type = s;
  targets = std::move(qubits);
  name = toString(type);

  for (size_t i = 0; i < dirs.size(); ++i) {
    operations.emplace_back(dirs[i], start[i], end[i]);
  }
}

AodOperation::AodOperation(const std::string& typeName,
                           std::vector<qc::Qubit> qubits,
                           const std::vector<uint32_t>& dirs,
                           const std::vector<qc::fp>& start,
                           const std::vector<qc::fp>& end)
    : AodOperation(qc::OP_NAME_TO_TYPE.at(typeName), std::move(qubits),
                   convertToDimension(dirs), start, end) {}

AodOperation::AodOperation(
    const qc::OpType s, std::vector<qc::Qubit> qubits,
    const std::vector<std::tuple<Dimension, qc::fp, qc::fp>>& ops) {
  type = s;
  targets = std::move(qubits);
  name = toString(type);

  for (const auto& [dir, index, param] : ops) {
    operations.emplace_back(dir, index, param);
  }
}

AodOperation::AodOperation(qc::OpType s, std::vector<qc::Qubit> t,
                           std::vector<SingleOperation> ops)
    : operations(std::move(ops)) {
  type = s;
  targets = std::move(t);
  name = toString(type);
}

void AodOperation::dumpOpenQASM(std::ostream& of, const qc::RegisterNames& qreg,
                                [[maybe_unused]] const qc::RegisterNames& creg,
                                size_t indent, bool /*openQASM3*/) const {
  of << std::setprecision(std::numeric_limits<qc::fp>::digits10);
  of << std::string(indent * qc::OUTPUT_INDENT_SIZE, ' ');
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
  if (type == qc::OpType::AodMove) {
    for (auto& op : operations) {
      std::swap(op.start, op.end);
    }
  } else if (type == qc::OpType::AodActivate) {
    type = qc::OpType::AodDeactivate;
  } else if (type == qc::OpType::AodDeactivate) {
    type = qc::OpType::AodActivate;
  }
}

} // namespace na
