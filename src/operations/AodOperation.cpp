#include "operations/AodOperation.hpp"
#include "Definitions.hpp"

#include "cassert"
#include "iomanip"
#include "limits"

#include <utility>

namespace qc {
AodOperation::AodOperation(OpType s, std::vector<Qubit> qubits,
                           const std::vector<uint32_t>& dirs, std::vector<fp> start,
                           std::vector<fp> end)
    : AodOperation(s, std::move(qubits), convertToDimension(dirs), std::move(start), std::move(end)) {}

std::vector<Dimension> AodOperation::convertToDimension(const std::vector<uint32_t>& dirs) {
  std::vector<Dimension> dirsEnum(dirs.size());
  for (size_t i = 0; i < dirs.size(); ++i) {
    dirsEnum[i] = static_cast<Dimension>(dirs[i]);
  }
  return dirsEnum;
}


AodOperation::AodOperation(OpType s, std::vector<Qubit> qubits,
                           std::vector<Dimension> dirs, std::vector<fp> start,
                           std::vector<fp> end) {
  assert(dirs.size() == start.size() && start.size() == end.size());
  type = s;
  targets = std::move(qubits);
  name = toString(type);

  for (size_t i = 0; i < dirs.size(); ++i) {
    operations.emplace_back(dirs[i], start[i], end[i]);
  }
}

AodOperation::AodOperation(const std::string& type, std::vector<Qubit> targets,
                           const std::vector<uint32_t>& dirs, std::vector<fp> start,
                           std::vector<fp> end)
    : AodOperation(OP_NAME_TO_TYPE.at(type), std::move(targets),
                   dirs, std::move(start), std::move(end)) {}

AodOperation::AodOperation(
    OpType s, std::vector<Qubit> targets,
    std::vector<std::tuple<Dimension, fp, fp>>& operations) {
  type = s;
  this->targets = std::move(targets);
  name = toString(type);

  for (const auto& [dir, index, param] : operations) {
    operations.emplace_back(dir, index, param);
  }
}

AodOperation::AodOperation(OpType s, std::vector<Qubit> t,
                           std::vector<SingleOperation>& ops)
    : operations(std::move(ops)) {
  type = s;
  this->targets = std::move(t);
  name = toString(type);
}

void AodOperation::dumpOpenQASM(std::ostream& of, const RegisterNames& qreg,
                                [[maybe_unused]] const RegisterNames& creg,
                                size_t /*indent*/, bool openQASM3) const {
  if (openQASM3) {
    throw std::runtime_error("AOD operations are not supported in OpenQASM3, "
                             "please use the OpenQASM2 output format.");
  }
  of << std::setprecision(std::numeric_limits<fp>::digits10);
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
      op.start = op.end;
      op.end = -op.start;
    }
  }
  if (type == OpType::AodActivate) {
    type = OpType::AodDeactivate;
  }
  if (type == OpType::AodDeactivate) {
    type = OpType::AodActivate;
  }
}

} // namespace qc
