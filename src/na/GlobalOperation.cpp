//
// This file is part of the MQT QMAP library released under the MIT license.
// See README.md or go to https://github.com/cda-tum/qmap for more information.
//

#include "na/GlobalOperation.hpp"

bool na::GlobalOperation::checkAllOpTypeAndOneLayer(
    std::vector<std::unique_ptr<qc::Operation>>&& operations) const {
  std::set<qc::Qubit> qubits;
  for (auto& op : operations) {
    if (op->getType() != opsType) {
      return false;
    }
    // check whether the intersection of qubits and op->getUsedQubits() is not
    // empty
    if (!qubits.empty()) {
      std::vector<qc::Qubit> intersection;
      std::set_intersection(
          qubits.cbegin(), qubits.cend(), op->getUsedQubits().cbegin(),
          op->getUsedQubits().cend(), std::back_inserter(intersection));
      if (!intersection.empty()) {
        return false;
      }
    }
    // insert used qubits into qubits
    qubits.merge(op->getUsedQubits());
  }
  return true;
}

template <class T, class... Args>
void na::GlobalOperation::emplace_back(Args&&... args) {
  auto newops = std::make_unique<T>(args...);
  // if not all operations in args are of opsType return false
  for (auto& op : newops) {
    if (op->getType() != opsType) {
      throw std::invalid_argument(
          "Not all operations are of the same type as the global operation.");
    }
  }
  // if the intersection of getUsedQubits() and op->getUsedQubits() is not
  // empty return false
  std::set<qc::Qubit> qubits;
  for (auto& op : newops) {
    qubits.merge(op->getUsedQubits());
    if (!getUsedQubits().empty()) {
      std::vector<qc::Qubit> intersection;
      std::set_intersection(getUsedQubits().cbegin(), getUsedQubits().cend(),
                            qubits.cbegin(), qubits.cend(),
                            std::back_inserter(intersection));
      if (!intersection.empty()) {
        throw std::invalid_argument(
            "The operation acts on a qubit that is already acted on by the "
            "global operation.");
      }
    }
  }
  ops.emplace_back(newops);
}

template <class T>
void na::GlobalOperation::emplace_back(std::unique_ptr<T>& op) {
  if (op->getType() != opsType) {
    throw std::invalid_argument(
        "The operation is not of the same type as the global operation.");
  }
  // if the intersection of getUsedQubits() and op->getUsedQubits() is not
  // empty return false
  if (!getUsedQubits().empty()) {
    std::vector<qc::Qubit> intersection;
    std::set_intersection(getUsedQubits().cbegin(), getUsedQubits().cend(),
                          op->getUsedQubits().cbegin(),
                          op->getUsedQubits().cend(),
                          std::back_inserter(intersection));
    if (!intersection.empty()) {
      throw std::invalid_argument(
          "The operation acts on a qubit that is already acted on by the "
          "global operation.");
    }
  }
  ops.emplace_back(std::move(op));
}

template <class T, class... Args>
std::vector<std::unique_ptr<qc::Operation>>::iterator
na::GlobalOperation::insert(
    std::vector<std::unique_ptr<qc::Operation>>::const_iterator iterator,
    Args&&... args) {
  auto newops = std::make_unique<T>(args...);
  // if not all operations in args are of opsType return false
  for (auto& op : newops) {
    if (op->getType() != opsType) {
      throw std::invalid_argument(
          "Not all operations are of the same type as the global operation.");
    }
  }
  // if the intersection of getUsedQubits() and op->getUsedQubits() is not
  // empty return false
  std::set<qc::Qubit> qubits;
  for (auto& op : newops) {
    qubits.merge(op->getUsedQubits());
    if (!getUsedQubits().empty()) {
      std::vector<qc::Qubit> intersection;
      std::set_intersection(getUsedQubits().cbegin(), getUsedQubits().cend(),
                            qubits.cbegin(), qubits.cend(),
                            std::back_inserter(intersection));
      if (!intersection.empty()) {
        throw std::invalid_argument(
            "The operation acts on a qubit that is already acted on by the "
            "global operation.");
      }
    }
  }
  return ops.insert(iterator, newops);
}

template <class T>
std::vector<std::unique_ptr<qc::Operation>>::iterator
na::GlobalOperation::insert(
    std::vector<std::unique_ptr<qc::Operation>>::const_iterator iterator,
    std::unique_ptr<T>& op) {
  if (op->getType() != opsType) {
    throw std::invalid_argument(
        "The operation is not of the same type as the global operation.");
  }
  // if the intersection of getUsedQubits() and op->getUsedQubits() is not
  // empty return false
  if (!getUsedQubits().empty()) {
    std::vector<qc::Qubit> intersection;
    std::set_intersection(getUsedQubits().cbegin(), getUsedQubits().cend(),
                          op->getUsedQubits().cbegin(),
                          op->getUsedQubits().cend(),
                          std::back_inserter(intersection));
    if (!intersection.empty()) {
      throw std::invalid_argument(
          "The operation acts on a qubit that is already acted on by the "
          "global operation.");
    }
  }
  return ops.insert(iterator, std::move(op));
}

void na::GlobalOperation::invert() {
  switch (opsType) {
  // self-inverting gates
  case qc::I:
  case qc::X:
  case qc::Y:
  case qc::Z:
  case qc::H:
  case qc::SWAP:
  case qc::ECR:
  case qc::Barrier:
  // gates where we just update parameters
  case qc::GPhase:
  case qc::P:
  case qc::RX:
  case qc::RY:
  case qc::RZ:
  case qc::RXX:
  case qc::RYY:
  case qc::RZZ:
  case qc::RZX:
  case qc::U2:
  case qc::U:
  case qc::XXminusYY:
  case qc::XXplusYY:
  case qc::DCX:
    break;
  // gates where we have specialized inverted operation types
  case qc::S:
    opsType = qc::Sdg;
    break;
  case qc::Sdg:
    opsType = qc::S;
    break;
  case qc::T:
    type = qc::Tdg;
    break;
  case qc::Tdg:
    opsType = qc::T;
    break;
  case qc::V:
    opsType = qc::Vdg;
    break;
  case qc::Vdg:
    opsType = qc::V;
    break;
  case qc::SX:
    opsType = qc::SXdg;
    break;
  case qc::SXdg:
    opsType = qc::SX;
    break;
  case qc::Peres:
    opsType = qc::Peresdg;
    break;
  case qc::Peresdg:
    opsType = qc::Peres;
    break;
  case qc::iSWAP:
    opsType = qc::iSWAPdg;
    break;
  case qc::iSWAPdg:
    opsType = qc::iSWAP;
    break;
  default:
    throw std::runtime_error("Inverting gate" + toString(type) +
                             " is not supported.");
  }
  qc::CompoundOperation::invert();
}
