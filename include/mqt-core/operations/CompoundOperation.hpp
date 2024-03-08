#pragma once

#include "Operation.hpp"

namespace qc {

class CompoundOperation : public Operation {
protected:
  std::vector<std::unique_ptr<Operation>> ops{};

public:
  explicit CompoundOperation(std::size_t nq);

  CompoundOperation(std::size_t nq,
                    std::vector<std::unique_ptr<Operation>>&& operations);

  CompoundOperation(const CompoundOperation& co);

  CompoundOperation& operator=(const CompoundOperation& co);

  [[nodiscard]] std::unique_ptr<Operation> clone() const override;

  void setNqubits(std::size_t nq) override;

  [[nodiscard]] bool isCompoundOperation() const override;

  [[nodiscard]] bool isNonUnitaryOperation() const override;

  [[nodiscard]] inline bool isSymbolicOperation() const override;

  void addControl(Control c) override;

  void clearControls() override;

  void removeControl(Control c) override;

  Controls::iterator removeControl(Controls::iterator it) override;

  [[nodiscard]] bool equals(const Operation& op, const Permutation& perm1,
                            const Permutation& perm2) const override;
  [[nodiscard]] bool equals(const Operation& operation) const override;

  std::ostream& print(std::ostream& os, const Permutation& permutation,
                      std::size_t prefixWidth) const override;

  [[nodiscard]] bool actsOn(Qubit i) const override;

  void addDepthContribution(std::vector<std::size_t>& depths) const override;

  void dumpOpenQASM(std::ostream& of, const RegisterNames& qreg,
                    const RegisterNames& creg, size_t indent,
                    bool openQASM3) const override;

  std::vector<std::unique_ptr<Operation>>& getOps() { return ops; }

  [[nodiscard]] std::set<Qubit> getUsedQubits() const override;

  void invert() override;

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
};
} // namespace qc

namespace std {
template <> struct hash<qc::CompoundOperation> {
  std::size_t operator()(const qc::CompoundOperation& co) const noexcept;
};
} // namespace std
