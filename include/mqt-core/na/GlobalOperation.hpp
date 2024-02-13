//
// This file is part of the MQT QMAP library released under the MIT license.
// See README.md or go to https://github.com/cda-tum/qmap for more information.
//

#pragma once

#include "Definitions.hpp"
#include "Permutation.hpp"
#include "operations/CompoundOperation.hpp"
#include "operations/Control.hpp"
#include "operations/OpType.hpp"

#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace na {

class GlobalOperation : public qc::CompoundOperation {
protected:
  qc::OpType opsType;
  std::size_t nTargets;
  std::size_t nControls;

public:
  GlobalOperation(const qc::OpType& ot, std::size_t nctrl, const std::size_t nq)
      : qc::CompoundOperation(nq), opsType(ot), nControls(nctrl) {
    std::stringstream ss;
    switch (opsType) {
    case qc::X:
    case qc::Y:
    case qc::Z:
    case qc::RX:
    case qc::RY:
    case qc::RZ:
      nTargets = 1;
    default:
      throw std::invalid_argument(
          "The operation type is not supported for global operations.");
    }
    ss << "Global operation: ";
    for (std::size_t i = 0; i < nctrl; ++i) {
      ss << "c";
    }
    ss << qc::toString(ot);
    name = ss.str();
    type = qc::Global;
  }

  GlobalOperation(const qc::OpType& ot, std::size_t nctrl, std::size_t nq,
                  std::vector<std::unique_ptr<qc::Operation>>&& operations);

  [[nodiscard]] std::unique_ptr<qc::Operation> clone() const override {
    return std::make_unique<GlobalOperation>(*this);
  }

  [[nodiscard]] bool isGlobalOperation() const override { return true; }

  [[nodiscard]] qc::OpType getOpsType() const { return opsType; }

  [[nodiscard]] std::size_t getNtargets() const override { return nTargets; }

  [[nodiscard]] std::size_t getNcontrols() const override { return nControls; }

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
      return opsType == gop->getOpsType() && nControls == gop->nControls &&
             nTargets == gop->nTargets && qc::CompoundOperation::equals(op);
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
  template <class T, class... Args> void emplace_back(Args&&... args) {
    emplace_back(std::make_unique<T>(args...));
  }

  /**
   * @brief Adds an operation to the global operation and ensures that all
   * operations are of the same type as the global operation and that there is
   * at most one operation acting on one qubit.
   *
   * @param args
   * @return template <class T>
   */
  // NOLINTNEXTLINE(readability-identifier-naming)
  template <class T> void emplace_back(std::unique_ptr<T>& op) {
    if (op->getType() != opsType || op->getNcontrols() != nControls ||
        op->getNtargets() != nTargets || op->getParameter() != parameter) {
      throw std::invalid_argument(
          "All operations in the global operation must be of the same type.");
    }
    // if the intersection of getUsedQubits() and op->getUsedQubits() is not
    // empty throw an exception
    std::set<qc::Qubit> opQubits = op->getUsedQubits();
    std::set<qc::Qubit> qubits = getUsedQubits();
    if (!getUsedQubits().empty()) {
      std::vector<qc::Qubit> intersection;
      std::set_intersection(qubits.begin(), qubits.end(), opQubits.begin(),
                            opQubits.end(), std::back_inserter(intersection));
      if (!intersection.empty()) {
        throw std::invalid_argument(
            "The operation acts on a qubit that is already acted on by the "
            "global operation.");
      }
    }
    ops.emplace_back(std::move(op));
  }

  /**
   * @brief Adds an operation to the global operation and ensures that all
   * operations are of the same type as the global operation and that there is
   * at most one operation acting on one qubit.
   *
   * @param args
   * @return template <class T>
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
         Args&&... args) {
    return ops.insert(iterator, std::make_unique<T>(args...));
  }

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
         std::unique_ptr<T>& op) {
    if (op->getType() != opsType || op->getNcontrols() != nControls ||
        op->getNtargets() != nTargets || op->getParameter() != parameter) {
      throw std::invalid_argument(
          "All operations in the global operation must be of the same type.");
    }
    // if the intersection of getUsedQubits() and op->getUsedQubits() is not
    // empty throw an exception
    std::set<qc::Qubit> opQubits = op->getUsedQubits();
    std::set<qc::Qubit> qubits = getUsedQubits();
    if (!getUsedQubits().empty()) {
      std::vector<qc::Qubit> intersection;
      std::set_intersection(qubits.begin(), qubits.end(), opQubits.begin(),
                            opQubits.end(), std::back_inserter(intersection));
      if (!intersection.empty()) {
        throw std::invalid_argument(
            "The operation acts on a qubit that is already acted on by the "
            "global operation.");
      }
    }
    return ops.insert(iterator, std::move(op));
  }

  void invert() override;
};
} // namespace na