#include "operations/CompoundOperation.hpp"

#include <CircuitOptimizer.hpp>
#include <QuantumComputation.hpp>
#include <algorithm>
#include <cassert>
#include <iterator>
#include <numeric>
#include <unordered_set>

namespace qc {
CompoundOperation::CompoundOperation() {
  name = "Compound operation:";
  type = Compound;
}

CompoundOperation::CompoundOperation(
    std::vector<std::unique_ptr<Operation>>&& operations)
    : CompoundOperation() {
  // NOLINTNEXTLINE(cppcoreguidelines-prefer-member-initializer)
  ops = std::move(operations);
}

CompoundOperation::CompoundOperation(const CompoundOperation& co)
    : Operation(co), ops(co.ops.size()) {
  for (std::size_t i = 0; i < co.ops.size(); ++i) {
    ops[i] = co.ops[i]->clone();
  }
}

CompoundOperation& CompoundOperation::operator=(const CompoundOperation& co) {
  if (this != &co) {
    Operation::operator=(co);
    ops.resize(co.ops.size());
    for (std::size_t i = 0; i < co.ops.size(); ++i) {
      ops[i] = co.ops[i]->clone();
    }
  }
  return *this;
}

std::unique_ptr<Operation> CompoundOperation::clone() const {
  return std::make_unique<CompoundOperation>(*this);
}

bool CompoundOperation::isNonUnitaryOperation() const {
  return std::any_of(ops.cbegin(), ops.cend(), [](const auto& op) {
    return op->isNonUnitaryOperation();
  });
}

bool CompoundOperation::isCompoundOperation() const { return true; }

bool CompoundOperation::isSymbolicOperation() const {
  return std::any_of(ops.begin(), ops.end(),
                     [](const auto& op) { return op->isSymbolicOperation(); });
}

void CompoundOperation::addControl(const Control c) {
  controls.insert(c);
  // we can just add the controls to each operation, as the operations will
  // check if they already act on the control qubits.
  for (auto& op : ops) {
    op->addControl(c);
  }
}

void CompoundOperation::clearControls() {
  // we remove just our controls from nested operations
  removeControls(controls);
}

void CompoundOperation::removeControl(const Control c) {
  // first we iterate over our controls and check if we are actually allowed
  // to remove them
  if (controls.erase(c) == 0) {
    throw QFRException("Cannot remove control from compound operation as it "
                       "is not a control.");
  }

  for (auto& op : ops) {
    op->removeControl(c);
  }
}

Controls::iterator
CompoundOperation::removeControl(const Controls::iterator it) {
  for (auto& op : ops) {
    op->removeControl(*it);
  }

  return controls.erase(it);
}
bool CompoundOperation::equals(const Operation& op, const Permutation& perm1,
                               const Permutation& perm2) const {
  if (const auto* comp = dynamic_cast<const CompoundOperation*>(&op)) {
    if (comp->ops.size() != ops.size()) {
      return false;
    }

    auto it = comp->ops.cbegin();
    for (const auto& operation : ops) {
      if (!operation->equals(**it, perm1, perm2)) {
        return false;
      }
      ++it;
    }
    return true;
  }
  return false;
}

bool CompoundOperation::equals(const Operation& operation) const {
  return equals(operation, {}, {});
}

std::ostream& CompoundOperation::print(std::ostream& os,
                                       const Permutation& permutation,
                                       const std::size_t prefixWidth,
                                       const std::size_t nqubits) const {
  const auto prefix = std::string(prefixWidth - 1, ' ');
  os << std::string(4 * nqubits, '-') << "\n";
  for (const auto& op : ops) {
    os << prefix << ":";
    op->print(os, permutation, prefixWidth, nqubits);
    os << "\n";
  }
  os << prefix << std::string(4 * nqubits + 1, '-');
  return os;
}

bool CompoundOperation::actsOn(const Qubit i) const {
  return std::any_of(ops.cbegin(), ops.cend(),
                     [&i](const auto& op) { return op->actsOn(i); });
}

void CompoundOperation::addDepthContribution(
    std::vector<std::size_t>& depths) const {
  for (const auto& op : ops) {
    op->addDepthContribution(depths);
  }
}

void CompoundOperation::dumpOpenQASM(std::ostream& of,
                                     const RegisterNames& qreg,
                                     const RegisterNames& creg, size_t indent,
                                     bool openQASM3) const {
  for (const auto& op : ops) {
    op->dumpOpenQASM(of, qreg, creg, indent, openQASM3);
  }
}

std::set<Qubit> CompoundOperation::getUsedQubits() const {
  std::set<Qubit> usedQubits{};
  for (const auto& op : ops) {
    usedQubits.merge(op->getUsedQubits());
  }
  return usedQubits;
}

auto CompoundOperation::commutesAtQubit(const Operation& other,
                                        const Qubit& qubit) const -> bool {
  return std::all_of(ops.cbegin(), ops.cend(),
                     [&other, &qubit](const auto& op) {
                       return op->commutesAtQubit(other, qubit);
                     });
}

auto CompoundOperation::isInverseOf(const Operation& other) const -> bool {
  if (other.isCompoundOperation()) {
    // cast other to CompoundOperation
    const auto& co = dynamic_cast<const CompoundOperation&>(other);
    if (size() != co.size()) {
      return false;
    }
    // here both compound operations have the same size
    if (empty()) {
      return true;
    }
    // transform compound to a QuantumComputation such that the invert method
    // and the reorderOperations method can be used to get a canonical form of
    // the compound operations
    const auto& thisUsedQubits = getUsedQubits();
    assert(!thisUsedQubits.empty());
    const auto thisMaxQubit =
        *std::max_element(thisUsedQubits.cbegin(), thisUsedQubits.cend());
    QuantumComputation thisQc(thisMaxQubit);
    std::for_each(cbegin(), cend(),
                  [&](const auto& op) { thisQc.emplace_back(op->clone()); });
    const auto& otherUsedQubits = co.getUsedQubits();
    assert(!otherUsedQubits.empty());
    const auto otherMaxQubit =
        *std::max_element(otherUsedQubits.cbegin(), otherUsedQubits.cend());
    QuantumComputation otherQc(otherMaxQubit);
    std::for_each(co.cbegin(), co.cend(),
                  [&](const auto& op) { otherQc.emplace_back(op->clone()); });
    CircuitOptimizer::reorderOperations(thisQc);
    otherQc.invert();
    CircuitOptimizer::reorderOperations(otherQc);
    return std::equal(
        thisQc.cbegin(), thisQc.cend(), otherQc.cbegin(),
        [](const auto& op1, const auto& op2) { return op1 == op2; });
  }
  return false;
}

void CompoundOperation::invert() {
  for (auto& op : ops) {
    op->invert();
  }
  std::reverse(ops.begin(), ops.end());
}

void CompoundOperation::apply(const Permutation& permutation) {
  Operation::apply(permutation);
  for (auto& op : ops) {
    op->apply(permutation);
  }
}

void CompoundOperation::merge(CompoundOperation& op) {
  ops.reserve(ops.size() + op.size());
  ops.insert(ops.end(), std::make_move_iterator(op.begin()),
             std::make_move_iterator(op.end()));
  op.clear();
}

bool CompoundOperation::isConvertibleToSingleOperation() const {
  if (ops.size() != 1) {
    return false;
  }
  if (!ops.front()->isCompoundOperation()) {
    return true;
  }
  return dynamic_cast<CompoundOperation*>(ops.front().get())
      ->isConvertibleToSingleOperation();
}

std::unique_ptr<Operation> CompoundOperation::collapseToSingleOperation() {
  assert(isConvertibleToSingleOperation());
  if (!ops.front()->isCompoundOperation()) {
    return std::move(ops.front());
  }
  return dynamic_cast<CompoundOperation*>(ops.front().get())
      ->collapseToSingleOperation();
}

} // namespace qc

namespace std {
std::size_t std::hash<qc::CompoundOperation>::operator()(
    const qc::CompoundOperation& co) const noexcept {
  std::size_t seed = 0U;
  for (const auto& op : co) {
    qc::hashCombine(seed, std::hash<qc::Operation>{}(*op));
  }
  return seed;
}
} // namespace std
