#pragma once

#include "dd/GateMatrixDefinitions.hpp"
#include "dd/Package.hpp"
#include "operations/ClassicControlledOperation.hpp"
#include "operations/CompoundOperation.hpp"
#include "operations/NonUnitaryOperation.hpp"
#include "operations/StandardOperation.hpp"

#include <variant>

namespace dd {
// single-target Operations
template <class Config>
qc::MatrixDD getStandardOperationDD(const qc::StandardOperation* op,
                                    std::unique_ptr<Package<Config>>& dd,
                                    const qc::Controls& controls,
                                    const qc::Qubit target,
                                    const bool inverse) {
  GateMatrix gm;

  const auto type = op->getType();
  const auto nqubits = op->getNqubits();
  const auto startQubit = op->getStartingQubit();
  const auto& parameter = op->getParameter();

  switch (type) {
  case qc::I:
    gm = Imat;
    break;
  case qc::H:
    gm = Hmat;
    break;
  case qc::X:
    gm = Xmat;
    break;
  case qc::Y:
    gm = Ymat;
    break;
  case qc::Z:
    gm = Zmat;
    break;
  case qc::S:
    gm = inverse ? Sdagmat : Smat;
    break;
  case qc::Sdg:
    gm = inverse ? Smat : Sdagmat;
    break;
  case qc::T:
    gm = inverse ? Tdagmat : Tmat;
    break;
  case qc::Tdg:
    gm = inverse ? Tmat : Tdagmat;
    break;
  case qc::V:
    gm = inverse ? Vdagmat : Vmat;
    break;
  case qc::Vdg:
    gm = inverse ? Vmat : Vdagmat;
    break;
  case qc::U:
    gm = inverse ? U3mat(-parameter[1U], -parameter[2U], -parameter[0U])
                 : U3mat(parameter[2U], parameter[1U], parameter[0U]);
    break;
  case qc::U2:
    gm = inverse ? U2mat(-parameter[0U] + PI, -parameter[1U] - PI)
                 : U2mat(parameter[1U], parameter[0U]);
    break;
  case qc::P:
    gm = inverse ? Phasemat(-parameter[0U]) : Phasemat(parameter[0U]);
    break;
  case qc::SX:
    gm = inverse ? SXdagmat : SXmat;
    break;
  case qc::SXdg:
    gm = inverse ? SXmat : SXdagmat;
    break;
  case qc::RX:
    gm = inverse ? RXmat(-parameter[0U]) : RXmat(parameter[0U]);
    break;
  case qc::RY:
    gm = inverse ? RYmat(-parameter[0U]) : RYmat(parameter[0U]);
    break;
  case qc::RZ:
    gm = inverse ? RZmat(-parameter[0U]) : RZmat(parameter[0U]);
    break;
  default:
    std::ostringstream oss{};
    oss << "DD for gate" << op->getName() << " not available!";
    throw qc::QFRException(oss.str());
  }
  return dd->makeGateDD(gm, nqubits, controls, target, startQubit);
}

// two-target Operations
template <class Config>
qc::MatrixDD getStandardOperationDD(const qc::StandardOperation* op,
                                    std::unique_ptr<Package<Config>>& dd,
                                    const qc::Controls& controls,
                                    qc::Qubit target0, qc::Qubit target1,
                                    const bool inverse) {
  const auto type = op->getType();
  const auto nqubits = op->getNqubits();
  const auto startQubit = op->getStartingQubit();
  const auto& parameter = op->getParameter();

  if (type == qc::DCX && inverse) {
    // DCX is not self-inverse, but the inverse is just swapping the targets
    std::swap(target0, target1);
  }

  if (controls.empty()) {
    // the DD creation without controls is faster, so we use it if possible
    // and only use the DD creation with controls if necessary
    TwoQubitGateMatrix gm;
    bool definitionFound = true;
    switch (type) {
    case qc::SWAP:
      gm = SWAPmat;
      break;
    case qc::iSWAP:
      gm = inverse ? iSWAPinvmat : iSWAPmat;
      break;
    case qc::DCX:
      gm = DCXmat;
      break;
    case qc::ECR:
      gm = ECRmat;
      break;
    case qc::RXX:
      gm = inverse ? RXXmat(-parameter[0U]) : RXXmat(parameter[0U]);
      break;
    case qc::RYY:
      gm = inverse ? RYYmat(-parameter[0U]) : RYYmat(parameter[0U]);
      break;
    case qc::RZZ:
      gm = inverse ? RZZmat(-parameter[0U]) : RZZmat(parameter[0U]);
      break;
    case qc::RZX:
      gm = inverse ? RZXmat(-parameter[0U]) : RZXmat(parameter[0U]);
      break;
    case qc::XXminusYY:
      gm = inverse ? XXMinusYYmat(-parameter[0U], parameter[1U])
                   : XXMinusYYmat(parameter[0U], parameter[1U]);
      break;
    case qc::XXplusYY:
      gm = inverse ? XXPlusYYmat(-parameter[0U], parameter[1U])
                   : XXPlusYYmat(parameter[0U], parameter[1U]);
      break;
    default:
      definitionFound = false;
    }
    if (definitionFound) {
      return dd->makeTwoQubitGateDD(gm, nqubits, target0, target1, startQubit);
    }
  }

  switch (type) {
  case qc::SWAP:
    // SWAP is self-inverse
    return dd->makeSWAPDD(nqubits, controls, target0, target1, startQubit);
  case qc::iSWAP:
    if (inverse) {
      return dd->makeiSWAPinvDD(nqubits, controls, target0, target1,
                                startQubit);
    }
    return dd->makeiSWAPDD(nqubits, controls, target0, target1, startQubit);
  case qc::Peres:
    if (inverse) {
      return dd->makePeresdagDD(nqubits, controls, target0, target1,
                                startQubit);
    }
    return dd->makePeresDD(nqubits, controls, target0, target1, startQubit);
  case qc::Peresdg:
    if (inverse) {
      return dd->makePeresDD(nqubits, controls, target0, target1, startQubit);
    }
    return dd->makePeresdagDD(nqubits, controls, target0, target1, startQubit);
  case qc::DCX:
    return dd->makeDCXDD(nqubits, controls, target0, target1, startQubit);
  case qc::ECR:
    // ECR is self-inverse
    return dd->makeECRDD(nqubits, controls, target0, target1, startQubit);
  case qc::RXX: {
    if (inverse) {
      return dd->makeRXXDD(nqubits, controls, target0, target1, -parameter[0U],
                           startQubit);
    }
    return dd->makeRXXDD(nqubits, controls, target0, target1, parameter[0U],
                         startQubit);
  }
  case qc::RYY: {
    if (inverse) {
      return dd->makeRYYDD(nqubits, controls, target0, target1, -parameter[0U],
                           startQubit);
    }
    return dd->makeRYYDD(nqubits, controls, target0, target1, parameter[0U],
                         startQubit);
  }
  case qc::RZZ: {
    if (inverse) {
      return dd->makeRZZDD(nqubits, controls, target0, target1, -parameter[0U],
                           startQubit);
    }
    return dd->makeRZZDD(nqubits, controls, target0, target1, parameter[0U],
                         startQubit);
  }
  case qc::RZX: {
    if (inverse) {
      return dd->makeRZXDD(nqubits, controls, target0, target1, -parameter[0U],
                           startQubit);
    }
    return dd->makeRZXDD(nqubits, controls, target0, target1, parameter[0U],
                         startQubit);
  }
  case qc::XXminusYY: {
    if (inverse) {
      return dd->makeXXMinusYYDD(nqubits, controls, target0, target1,
                                 -parameter[0U], parameter[1U], startQubit);
    }
    return dd->makeXXMinusYYDD(nqubits, controls, target0, target1,
                               parameter[0U], parameter[1U], startQubit);
  }
  case qc::XXplusYY: {
    if (inverse) {
      return dd->makeXXPlusYYDD(nqubits, controls, target0, target1,
                                -parameter[0U], parameter[1U], startQubit);
    }
    return dd->makeXXPlusYYDD(nqubits, controls, target0, target1,
                              parameter[0U], parameter[1U], startQubit);
  }
  default:
    std::ostringstream oss{};
    oss << "DD for gate" << op->getName() << " not available!";
    throw qc::QFRException(oss.str());
  }
}

// The methods with a permutation parameter apply these Operations according to
// the mapping specified by the permutation, e.g.
//      if perm[0] = 1 and perm[1] = 0
//      then cx 0 1 will be translated to cx perm[0] perm[1] == cx 1 0

template <class Config>
qc::MatrixDD getDD(const qc::Operation* op,
                   std::unique_ptr<Package<Config>>& dd,
                   qc::Permutation& permutation, const bool inverse = false) {
  const auto type = op->getType();
  const auto nqubits = op->getNqubits();

  // check whether the operation can be handled by the underlying DD package
  if (nqubits > Package<Config>::MAX_POSSIBLE_QUBITS) {
    throw qc::QFRException(
        "Requested too many qubits to be handled by the DD package. Qubit "
        "datatype only allows up to " +
        std::to_string(Package<Config>::MAX_POSSIBLE_QUBITS) +
        " qubits, while " + std::to_string(nqubits) +
        " were requested. If you want to use more than " +
        std::to_string(Package<Config>::MAX_POSSIBLE_QUBITS) +
        " qubits, you have to recompile the package with a wider Qubit type in "
        "`include/dd/DDDefinitions.hpp!`");
  }

  // if a permutation is provided and the current operation is a SWAP, this
  // routine just updates the permutation
  if (!permutation.empty() && type == qc::SWAP && !op->isControlled()) {
    const auto& targets = op->getTargets();

    const auto target0 = targets[0U];
    const auto target1 = targets[1U];
    // update permutation
    std::swap(permutation.at(target0), permutation.at(target1));
    return dd->makeIdent(nqubits);
  }

  if (type == qc::Barrier) {
    return dd->makeIdent(nqubits);
  }

  if (type == qc::GPhase) {
    auto phase = op->getParameter()[0U];
    if (inverse) {
      phase = -phase;
    }
    auto id = dd->makeIdent(nqubits);
    id.w = dd->cn.lookup(std::cos(phase), std::sin(phase));
    return id;
  }

  if (const auto* standardOp = dynamic_cast<const qc::StandardOperation*>(op)) {
    auto targets = op->getTargets();
    auto controls = op->getControls();
    if (!permutation.empty()) {
      targets = permutation.apply(targets);
      controls = permutation.apply(controls);
    }

    if (qc::isTwoQubitGate(type)) {
      assert(targets.size() == 2);
      return getStandardOperationDD(standardOp, dd, controls, targets[0U],
                                    targets[1U], inverse);
    }
    assert(targets.size() == 1);
    return getStandardOperationDD(standardOp, dd, controls, targets[0U],
                                  inverse);
  }

  if (const auto* compoundOp = dynamic_cast<const qc::CompoundOperation*>(op)) {
    auto e = dd->makeIdent(op->getNqubits());
    if (inverse) {
      for (const auto& operation : *compoundOp) {
        e = dd->multiply(e, getInverseDD(operation.get(), dd, permutation));
      }
    } else {
      for (const auto& operation : *compoundOp) {
        e = dd->multiply(getDD(operation.get(), dd, permutation), e);
      }
    }
    return e;
  }

  if (const auto* classicOp =
          dynamic_cast<const qc::ClassicControlledOperation*>(op)) {
    return getDD(classicOp->getOperation(), dd, permutation, inverse);
  }

  assert(op->isNonUnitaryOperation());
  throw qc::QFRException("DD for non-unitary operation not available!");
}

template <class Config>
qc::MatrixDD getDD(const qc::Operation* op,
                   std::unique_ptr<Package<Config>>& dd,
                   const bool inverse = false) {
  qc::Permutation perm{};
  return getDD(op, dd, perm, inverse);
}

template <class Config>
qc::MatrixDD getInverseDD(const qc::Operation* op,
                          std::unique_ptr<Package<Config>>& dd) {
  return getDD(op, dd, true);
}

template <class Config>
qc::MatrixDD getInverseDD(const qc::Operation* op,
                          std::unique_ptr<Package<Config>>& dd,
                          qc::Permutation& permutation) {
  return getDD(op, dd, permutation, true);
}

template <class Config>
void dumpTensor(qc::Operation* op, std::ostream& of,
                std::vector<std::size_t>& inds, std::size_t& gateIdx,
                std::unique_ptr<Package<Config>>& dd);

// apply swaps 'on' DD in order to change 'from' to 'to'
// where |from| >= |to|
template <class DDType, class Config>
void changePermutation(DDType& on, qc::Permutation& from,
                       const qc::Permutation& to,
                       std::unique_ptr<Package<Config>>& dd,
                       const bool regular = true) {
  assert(from.size() >= to.size());

  // iterate over (k,v) pairs of second permutation
  for (const auto& [i, goal] : to) {
    // search for key in the first map
    auto it = from.find(i);
    if (it == from.end()) {
      throw qc::QFRException(
          "[changePermutation] Key " + std::to_string(it->first) +
          " was not found in first permutation. This should never happen.");
    }
    auto current = it->second;

    // permutations agree for this key value
    if (current == goal) {
      continue;
    }

    // search for goal value in first permutation
    qc::Qubit j = 0;
    for (const auto& [key, value] : from) {
      if (value == goal) {
        j = key;
        break;
      }
    }

    // swap i and j
    auto saved = on;
    const auto swapDD = dd->makeSWAPDD(on.p->v + 1U, from.at(i), from.at(j));
    if constexpr (std::is_same_v<DDType, qc::VectorDD>) {
      on = dd->multiply(swapDD, on);
    } else {
      // the regular flag only has an effect on matrix DDs
      if (regular) {
        on = dd->multiply(swapDD, on);
      } else {
        on = dd->multiply(on, swapDD);
      }
    }

    dd->incRef(on);
    dd->decRef(saved);
    dd->garbageCollect();

    // update permutation
    from.at(i) = goal;
    from.at(j) = current;
  }
}

} // namespace dd
