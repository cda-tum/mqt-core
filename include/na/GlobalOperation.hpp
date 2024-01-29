//
// This file is part of the MQT QMAP library released under the MIT license.
// See README.md or go to https://github.com/cda-tum/qmap for more information.
//

#pragma once

#include "Definitions.hpp"
#include "Permutation.hpp"
#include "operations/CompoundOperation.hpp"
#include "operations/Control.hpp"

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace na {

class GlobalOperation : public qc::CompoundOperation {
protected:
  qc::OpType opsType;

private:
  /**
   * @brief The function checks whether all operations are of the type opsType
   * and that there is at most one operations operating on one qubit.
   *
   */
  [[nodiscard]] bool checkAllOpTypeAndOneLayer(
      std::vector<std::unique_ptr<qc::Operation>>&& operations) const;

public:
  explicit GlobalOperation(const qc::OpType& ot, const std::size_t nq)
      : qc::CompoundOperation(nq), opsType(ot) {
    name = "Global operation: " + qc::toString(ot);
    type = qc::Global;
  }

  explicit GlobalOperation(
      const qc::OpType& ot, const std::size_t nq,
      std::vector<std::unique_ptr<qc::Operation>>&& operations)
      : GlobalOperation(ot, nq) {
    if (!checkAllOpTypeAndOneLayer(std::move(operations))) {
      throw std::invalid_argument(
          "Not all operations are of the same type as the global operation or "
          "multiple operations act on the same qubit.");
    }
    ops = std::move(operations);
  }

  [[nodiscard]] std::unique_ptr<qc::Operation> clone() const override {
    return std::make_unique<GlobalOperation>(*this);
  }

  [[nodiscard]] bool isGlobalOperation() const override { return true; }

  [[nodiscard]] qc::OpType getOpsType() const { return opsType; }

  [[nodiscard]] bool isGlobalOn(std::set<qc::Qubit>& qubits) const {
    return getUsedQubits() == qubits;
  };

  void addControl([[maybe_unused]] const qc::Control c) override {
    throw std::logic_error("Global operations cannot have controls.");
  }

  void clearControls() override {
    throw std::logic_error("Global operations cannot have controls.");
  }

  void removeControl([[maybe_unused]] const qc::Control c) override {
    throw std::logic_error("Global operations cannot have controls.");
  }

  qc::Controls::iterator
  removeControl([[maybe_unused]] qc::Controls::iterator it) override {
    throw std::logic_error("Global operations cannot have controls.");
  }

  [[nodiscard]] bool equals(const qc::Operation& op) const override {
    if (const auto& gop = dynamic_cast<const GlobalOperation*>(&op)) {
      return opsType == gop->getOpsType() && qc::CompoundOperation::equals(op);
    }
    return false;
  }

  [[nodiscard]] bool equals(const qc::Operation& op,
                            const qc::Permutation& perm1,
                            const qc::Permutation& perm2) const override {
    if (const auto& gop = dynamic_cast<const GlobalOperation*>(&op)) {
      return opsType == gop->getOpsType() &&
             qc::CompoundOperation::equals(op, perm1, perm2);
    }
    return false;
  }

  /**
   * @brief Adds all operations to the global operation and ensures that all
   * operations are of the same type as the global operation and that there is
   * at most one operation acting on one qubit.
   *
   * @param args
   * @return template <class T, class... Args>
   */
  // NOLINTNEXTLINE(readability-identifier-naming)
  template <class T, class... Args> void emplace_back(Args&&... args);

  /**
   * @brief Adds an operation to the global operation and ensures that all
   * operations are of the same type as the global operation and that there is
   * at most one operation acting on one qubit.
   *
   * @param args
   * @return template <class T, class... Args>
   */
  // NOLINTNEXTLINE(readability-identifier-naming)
  template <class T> void emplace_back(std::unique_ptr<T>& op);

  /**
   * @brief Adds an operation to the global operation and ensures that all
   * operations are of the same type as the global operation and that there is
   * at most one operation acting on one qubit.
   *
   * @param args
   * @return template <class T, class... Args>
   */
  // NOLINTNEXTLINE(readability-identifier-naming)
  template <class T> void emplace_back(std::unique_ptr<T>&& op) {
    emplace_back(std::move(op));
  }

  /**
   * @brief Inserts all operations to the global operation and ensures that all
   * operations are of the same type as the global operation and that there is
   * at most one operation acting on one qubit.
   *
   * @param args
   * @return template <class T, class... Args>
   */
  template <class T, class... Args>
  std::vector<std::unique_ptr<qc::Operation>>::iterator
  insert(std::vector<std::unique_ptr<qc::Operation>>::const_iterator iterator,
         Args&&... args);

  /**
   * @brief Inserts an operation to the global operation and ensures that all
   * operations are of the same type as the global operation and that there is
   * at most one operation acting on one qubit.
   *
   * @param args
   * @return template <class T, class... Args>
   */
  template <class T>
  std::vector<std::unique_ptr<qc::Operation>>::iterator
  insert(std::vector<std::unique_ptr<qc::Operation>>::const_iterator iterator,
         std::unique_ptr<T>& op);

  void invert() override;
};
} // namespace na