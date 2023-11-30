#pragma once

#include "Operation.hpp"

#include <algorithm>

namespace qc {

class CompoundOperation final : public Operation {
private:
  std::vector<std::unique_ptr<Operation>> ops{};

public:
  explicit CompoundOperation(const std::size_t nq) {
    name = "Compound operation:";
    nqubits = nq;
    type = Compound;
  }

  explicit CompoundOperation(
      const std::size_t nq,
      std::vector<std::unique_ptr<Operation>>&& operations)
      : CompoundOperation(nq) {
    // NOLINTNEXTLINE(cppcoreguidelines-prefer-member-initializer)
    ops = std::move(operations);
  }

  CompoundOperation(const CompoundOperation& co)
      : Operation(co), ops(co.ops.size()) {
    for (std::size_t i = 0; i < co.ops.size(); ++i) {
      ops[i] = co.ops[i]->clone();
    }
  }

  CompoundOperation& operator=(const CompoundOperation& co) {
    if (this != &co) {
      Operation::operator=(co);
      ops.resize(co.ops.size());
      for (std::size_t i = 0; i < co.ops.size(); ++i) {
        ops[i] = co.ops[i]->clone();
      }
    }
    return *this;
  }

  [[nodiscard]] std::unique_ptr<Operation> clone() const override {
    return std::make_unique<CompoundOperation>(*this);
  }

  void setNqubits(const std::size_t nq) override {
    nqubits = nq;
    for (auto& op : ops) {
      op->setNqubits(nq);
    }
  }

  [[nodiscard]] bool isCompoundOperation() const override { return true; }

  [[nodiscard]] bool isNonUnitaryOperation() const override {
    return std::any_of(ops.cbegin(), ops.cend(), [](const auto& op) {
      return op->isNonUnitaryOperation();
    });
  }

  [[nodiscard]] inline bool isSymbolicOperation() const override {
    return std::any_of(ops.begin(), ops.end(), [](const auto& op) {
      return op->isSymbolicOperation();
    });
  }

  void addControl(const Control c) override {
    controls.insert(c);
    // we can just add the controls to each operation, as the operations will
    // check if they already act on the control qubits.
    for (auto& op : ops) {
      op->addControl(c);
    }
  }

  void clearControls() override {
    // we remove just our controls from nested operations
    removeControls(controls);
  }

  void removeControl(const Control c) override {
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

  Controls::iterator removeControl(const Controls::iterator it) override {
    for (auto& op : ops) {
      op->removeControl(*it);
    }

    return controls.erase(it);
  }

  [[nodiscard]] bool equals(const Operation& op, const Permutation& perm1,
                            const Permutation& perm2) const override {
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
  [[nodiscard]] bool equals(const Operation& operation) const override {
    return equals(operation, {}, {});
  }

  std::ostream& print(std::ostream& os, const Permutation& permutation,
                      const std::size_t prefixWidth) const override {
    const auto prefix = std::string(prefixWidth - 1, ' ');
    os << std::string(4 * nqubits, '-') << "\n";
    for (const auto& op : ops) {
      os << prefix << ":";
      op->print(os, permutation, prefixWidth);
      os << "\n";
    }
    os << prefix << std::string(4 * nqubits + 1, '-');
    return os;
  }

  [[nodiscard]] bool actsOn(const Qubit i) const override {
    return std::any_of(ops.cbegin(), ops.cend(),
                       [&i](const auto& op) { return op->actsOn(i); });
  }

  void addDepthContribution(std::vector<std::size_t>& depths) const override {
    for (const auto& op : ops) {
      op->addDepthContribution(depths);
    }
  }

  void dumpOpenQASM(std::ostream& of, const RegisterNames& qreg,
                    const RegisterNames& creg) const override {
    for (const auto& op : ops) {
      op->dumpOpenQASM(of, qreg, creg);
    }
  }

  /**
   * Pass-Through
   */

  // Iterators (pass-through)
  auto begin() noexcept { return ops.begin(); }
  [[nodiscard]] auto begin() const noexcept { return ops.begin(); }
  [[nodiscard]] auto cbegin() const noexcept { return ops.cbegin(); }
  auto end() noexcept { return ops.end(); }
  [[nodiscard]] auto end() const noexcept { return ops.end(); }
  [[nodiscard]] auto cend() const noexcept { return ops.cend(); }
  auto rbegin() noexcept { return ops.rbegin(); }
  [[nodiscard]] auto rbegin() const noexcept { return ops.rbegin(); }
  [[nodiscard]] auto crbegin() const noexcept { return ops.crbegin(); }
  auto rend() noexcept { return ops.rend(); }
  [[nodiscard]] auto rend() const noexcept { return ops.rend(); }
  [[nodiscard]] auto crend() const noexcept { return ops.crend(); }

  // Capacity (pass-through)
  [[nodiscard]] bool empty() const noexcept { return ops.empty(); }
  [[nodiscard]] std::size_t size() const noexcept { return ops.size(); }
  // NOLINTNEXTLINE(readability-identifier-naming)
  [[nodiscard]] std::size_t max_size() const noexcept { return ops.max_size(); }
  [[nodiscard]] std::size_t capacity() const noexcept { return ops.capacity(); }

  void reserve(std::size_t newCap) { ops.reserve(newCap); }
  // NOLINTNEXTLINE(readability-identifier-naming)
  void shrink_to_fit() { ops.shrink_to_fit(); }

  // Modifiers (pass-through)
  void clear() noexcept { ops.clear(); }
  // NOLINTNEXTLINE(readability-identifier-naming)
  void pop_back() { return ops.pop_back(); }
  void resize(std::size_t count) { ops.resize(count); }
  std::vector<std::unique_ptr<Operation>>::iterator
  erase(std::vector<std::unique_ptr<Operation>>::const_iterator pos) {
    return ops.erase(pos);
  }
  std::vector<std::unique_ptr<Operation>>::iterator
  erase(std::vector<std::unique_ptr<Operation>>::const_iterator first,
        std::vector<std::unique_ptr<Operation>>::const_iterator last) {
    return ops.erase(first, last);
  }

  // NOLINTNEXTLINE(readability-identifier-naming)
  template <class T, class... Args> void emplace_back(Args&&... args) {
    ops.emplace_back(std::make_unique<T>(args...));
  }

  // NOLINTNEXTLINE(readability-identifier-naming)
  template <class T> void emplace_back(std::unique_ptr<T>& op) {
    ops.emplace_back(std::move(op));
  }

  // NOLINTNEXTLINE(readability-identifier-naming)
  template <class T> void emplace_back(std::unique_ptr<T>&& op) {
    ops.emplace_back(std::move(op));
  }

  template <class T, class... Args>
  std::vector<std::unique_ptr<Operation>>::iterator
  insert(std::vector<std::unique_ptr<Operation>>::const_iterator iterator,
         Args&&... args) {
    return ops.insert(iterator, std::make_unique<T>(args...));
  }
  template <class T>
  std::vector<std::unique_ptr<Operation>>::iterator
  insert(std::vector<std::unique_ptr<Operation>>::const_iterator iterator,
         std::unique_ptr<T>& op) {
    return ops.insert(iterator, std::move(op));
  }

  [[nodiscard]] const auto& at(std::size_t i) const { return ops.at(i); }

  std::vector<std::unique_ptr<Operation>>& getOps() { return ops; }

  [[nodiscard]] std::set<Qubit> getUsedQubits() const override {
    std::set<Qubit> usedQubits{};
    for (const auto& op : ops) {
      usedQubits.merge(op->getUsedQubits());
    }
    return usedQubits;
  }

  void invert() override {
    for (auto& op : ops) {
      op->invert();
    }
    std::reverse(ops.begin(), ops.end());
  }
};
} // namespace qc

namespace std {
template <> struct hash<qc::CompoundOperation> {
  std::size_t operator()(const qc::CompoundOperation& co) const noexcept {
    std::size_t seed = 0U;
    for (const auto& op : co) {
      qc::hashCombine(seed, std::hash<qc::Operation>{}(*op));
    }
    return seed;
  }
};
} // namespace std
