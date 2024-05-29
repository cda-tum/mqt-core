#include "dd/NoiseFunctionality.hpp"

#include "Definitions.hpp"
#include "dd/ComplexNumbers.hpp"
#include "dd/DDDefinitions.hpp"
#include "dd/DDpackageConfig.hpp"
#include "dd/GateMatrixDefinitions.hpp"
#include "dd/Node.hpp"
#include "dd/Package.hpp"
#include "operations/OpType.hpp"
#include "operations/Operation.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<dd::NoiseOperations>
initializeNoiseEffects(const std::string& cNoiseEffects) {
  std::vector<dd::NoiseOperations> noiseOperationVector{};
  for (const auto noise : cNoiseEffects) {
    switch (noise) {
    case 'A':
      noiseOperationVector.push_back(dd::AmplitudeDamping);
      break;
    case 'P':
      noiseOperationVector.push_back(dd::PhaseFlip);
      break;
    case 'D':
      noiseOperationVector.push_back(dd::Depolarization);
      break;
    case 'I':
      noiseOperationVector.push_back(dd::Identity);
      break;
    default:
      throw std::runtime_error("Unknown noise operation '" + cNoiseEffects +
                               "'\n");
    }
  }
  return noiseOperationVector;
}
} // namespace

namespace dd {
StochasticNoiseFunctionality::StochasticNoiseFunctionality(
    const std::unique_ptr<Package<StochasticNoiseSimulatorDDPackageConfig>>& dd,
    const std::size_t nq, double gateNoiseProbability,
    double amplitudeDampingProb, double multiQubitGateFactor,
    const std::string& cNoiseEffects)
    : package(dd.get()), nQubits(nq), dist(0.0, 1.0L),
      noiseProbability(gateNoiseProbability),
      noiseProbabilityMulti(gateNoiseProbability * multiQubitGateFactor),
      sqrtAmplitudeDampingProbability(std::sqrt(amplitudeDampingProb)),
      oneMinusSqrtAmplitudeDampingProbability(
          std::sqrt(1 - amplitudeDampingProb)),
      sqrtAmplitudeDampingProbabilityMulti(std::sqrt(gateNoiseProbability) *
                                           multiQubitGateFactor),
      oneMinusSqrtAmplitudeDampingProbabilityMulti(
          std::sqrt(1 - multiQubitGateFactor * amplitudeDampingProb)),
      ampDampingTrue({0, sqrtAmplitudeDampingProbability, 0, 0}),
      ampDampingTrueMulti({0, sqrtAmplitudeDampingProbabilityMulti, 0, 0}),
      ampDampingFalse({1, 0, 0, oneMinusSqrtAmplitudeDampingProbability}),
      ampDampingFalseMulti(
          {1, 0, 0, oneMinusSqrtAmplitudeDampingProbabilityMulti}),
      noiseEffects(initializeNoiseEffects(cNoiseEffects)),
      identityDD(package->makeIdent()) {
  sanityCheckOfNoiseProbabilities(gateNoiseProbability, amplitudeDampingProb,
                                  multiQubitGateFactor);
  package->incRef(identityDD);
}

double StochasticNoiseFunctionality::getNoiseProbability(
    bool multiQubitNoiseFlag) const {
  return multiQubitNoiseFlag ? noiseProbabilityMulti : noiseProbability;
}

qc::OpType StochasticNoiseFunctionality::getAmplitudeDampingOperationType(
    const bool multiQubitNoiseFlag, const bool amplitudeDampingFlag) {
  if (amplitudeDampingFlag) {
    return multiQubitNoiseFlag ? qc::MultiATrue : qc::ATrue;
  }
  return multiQubitNoiseFlag ? qc::MultiAFalse : qc::AFalse;
}

GateMatrix StochasticNoiseFunctionality::getAmplitudeDampingOperationMatrix(
    const bool multiQubitNoiseFlag, const bool amplitudeDampingFlag) const {
  if (amplitudeDampingFlag) {
    return multiQubitNoiseFlag ? ampDampingTrueMulti : ampDampingTrue;
  }
  return multiQubitNoiseFlag ? ampDampingFalseMulti : ampDampingFalse;
}

void StochasticNoiseFunctionality::applyNoiseOperation(
    const std::set<qc::Qubit>& targets, mEdge operation, vEdge& state,
    std::mt19937_64& generator) {
  const bool multiQubitOperation = targets.size() > 1;

  for (const auto& target : targets) {
    auto stackedOperation = generateNoiseOperation(operation, target, generator,
                                                   false, multiQubitOperation);
    auto tmp = package->multiply(stackedOperation, state);

    if (ComplexNumbers::mag2(tmp.w) < dist(generator)) {
      // The probability of amplitude damping does not only depend on the
      // noise probability, but also the quantum state. Due to the
      // normalization constraint of decision diagrams the probability for
      // applying amplitude damping stands in the root edge weight, of the dd
      // after the noise has been applied
      stackedOperation = generateNoiseOperation(operation, target, generator,
                                                true, multiQubitOperation);
      tmp = package->multiply(stackedOperation, state);
    }
    tmp.w = Complex::one();

    package->incRef(tmp);
    package->decRef(state);
    state = tmp;

    // I only need to apply the operations once
    operation = identityDD;
  }
}

mEdge StochasticNoiseFunctionality::stackOperation(
    mEdge operation, const qc::Qubit target, const qc::OpType noiseOperation,
    const GateMatrix matrix) {
  if (const auto* op =
          package->stochasticNoiseOperationCache.lookup(noiseOperation, target);
      op != nullptr) {
    return package->multiply(*op, operation);
  }
  const auto gateDD = package->makeGateDD(matrix, target);
  package->stochasticNoiseOperationCache.insert(noiseOperation, target, gateDD);
  return package->multiply(gateDD, operation);
}

mEdge StochasticNoiseFunctionality::generateNoiseOperation(
    mEdge operation, qc::Qubit target, std::mt19937_64& generator,
    const bool amplitudeDamping, const bool multiQubitOperation) {
  for (const auto& noiseType : noiseEffects) {
    const auto effect = noiseType == AmplitudeDamping
                            ? getAmplitudeDampingOperationType(
                                  multiQubitOperation, amplitudeDamping)
                            : returnNoiseOperation(noiseType, dist(generator),
                                                   multiQubitOperation);
    switch (effect) {
    case (qc::I): {
      continue;
    }
    case (qc::MultiATrue):
    case (qc::ATrue): {
      const GateMatrix amplitudeDampingMatrix =
          getAmplitudeDampingOperationMatrix(multiQubitOperation, true);
      operation =
          stackOperation(operation, target, effect, amplitudeDampingMatrix);
      break;
    }
    case (qc::MultiAFalse):
    case (qc::AFalse): {
      const GateMatrix amplitudeDampingMatrix =
          getAmplitudeDampingOperationMatrix(multiQubitOperation, false);
      operation =
          stackOperation(operation, target, effect, amplitudeDampingMatrix);
      break;
    }
    case (qc::X): {
      operation = stackOperation(operation, target, effect, X_MAT);
      break;
    }
    case (qc::Y): {
      operation = stackOperation(operation, target, effect, Y_MAT);
      break;
    }
    case (qc::Z): {
      operation = stackOperation(operation, target, effect, Z_MAT);
      break;
    }
    default: {
      throw std::runtime_error("Unknown noise operation '" +
                               std::to_string(effect) + "'\n");
    }
    }
  }
  return operation;
}

qc::OpType StochasticNoiseFunctionality::returnNoiseOperation(
    NoiseOperations noiseOperation, const double prob,
    const bool multiQubitNoiseFlag) const {
  switch (noiseOperation) {
  case NoiseOperations::Depolarization: {
    if (prob >= (getNoiseProbability(multiQubitNoiseFlag) * 0.75)) {
      // prob > prob apply qc::I, also 25 % of the time when depolarization is
      // applied nothing happens
      return qc::I;
    }
    if (prob < (getNoiseProbability(multiQubitNoiseFlag) * 0.25)) {
      // if 0 < prob < 0.25 (25 % of the time when applying depolarization)
      // apply qc::X
      return qc::X;
    }
    if (prob < (getNoiseProbability(multiQubitNoiseFlag) * 0.5)) {
      // if 0.25 < prob < 0.5 (25 % of the time when applying depolarization)
      // apply qc::Y
      return qc::Y;
    }
    // if 0.5 < prob < 0.75 (25 % of the time when applying depolarization)
    // apply qc::Z
    return qc::Z;
  }
  case NoiseOperations::PhaseFlip: {
    if (prob > getNoiseProbability(multiQubitNoiseFlag)) {
      return qc::I;
    }
    return qc::Z;
  }
  case NoiseOperations::Identity: {
    return qc::I;
  }
  default:
    throw std::runtime_error(std::string{"Unknown noise effect '"} +
                             std::to_string(noiseOperation) + "'");
  }
}

DeterministicNoiseFunctionality::DeterministicNoiseFunctionality(
    const std::unique_ptr<Package<DensityMatrixSimulatorDDPackageConfig>>& dd,
    const std::size_t nq, double noiseProbabilitySingleQubit,
    double noiseProbabilityMultiQubit, double ampDampProbSingleQubit,
    double ampDampProbMultiQubit, const std::string& cNoiseEffects)
    : package(dd.get()), nQubits(nq),
      noiseProbSingleQubit(noiseProbabilitySingleQubit),
      noiseProbMultiQubit(noiseProbabilityMultiQubit),
      ampDampingProbSingleQubit(ampDampProbSingleQubit),
      ampDampingProbMultiQubit(ampDampProbMultiQubit),
      noiseEffects(initializeNoiseEffects(cNoiseEffects)) {
  sanityCheckOfNoiseProbabilities(noiseProbabilitySingleQubit,
                                  ampDampProbSingleQubit, 1);
  sanityCheckOfNoiseProbabilities(noiseProbabilityMultiQubit,
                                  ampDampProbMultiQubit, 1);
}

void DeterministicNoiseFunctionality::applyNoiseEffects(
    dEdge& originalEdge, const std::unique_ptr<qc::Operation>& qcOperation) {
  auto usedQubits = qcOperation->getUsedQubits();
  dCachedEdge nodeAfterNoise = {};
  dEdge::applyDmChangesToEdge(originalEdge);
  nodeAfterNoise = applyNoiseEffects(originalEdge, usedQubits, false,
                                     static_cast<Qubit>(nQubits));
  dEdge::revertDmChangesToEdge(originalEdge);
  auto r = dEdge{nodeAfterNoise.p, package->cn.lookup(nodeAfterNoise.w)};
  package->incRef(r);
  dEdge::alignDensityEdge(originalEdge);
  package->decRef(originalEdge);
  originalEdge = r;
  dEdge::setDensityMatrixTrue(originalEdge);
}

dCachedEdge DeterministicNoiseFunctionality::applyNoiseEffects(
    dEdge& originalEdge, const std::set<qc::Qubit>& usedQubits,
    const bool firstPathEdge, const Qubit level) {

  const auto originalWeight = static_cast<ComplexValue>(originalEdge.w);
  if (originalEdge.isZeroTerminal() || level <= *usedQubits.begin()) {
    return {originalEdge.p, originalWeight};
  }

  auto originalCopy = dEdge{originalEdge.p, Complex::one()};
  ArrayOfEdges newEdges{};
  const auto nextLevel = static_cast<dd::Qubit>(level - 1U);
  if (originalEdge.isIdentity()) {
    newEdges[0] =
        applyNoiseEffects(originalCopy, usedQubits, firstPathEdge, nextLevel);
    newEdges[3] =
        applyNoiseEffects(originalCopy, usedQubits, firstPathEdge, nextLevel);
  } else {
    for (std::size_t i = 0; i < newEdges.size(); i++) {
      auto& successor = originalCopy.p->e[i];
      if (firstPathEdge || i == 1) {
        // If I am to the firstPathEdge I cannot minimize the necessary
        // operations anymore
        dEdge::applyDmChangesToEdge(successor);
        newEdges[i] = applyNoiseEffects(successor, usedQubits, true, nextLevel);
        dEdge::revertDmChangesToEdge(successor);
      } else if (i == 2) {
        // Since e[1] == e[2] (due to density matrix representation), I can skip
        // calculating e[2]
        newEdges[2] = newEdges[1];
      } else {
        dEdge::applyDmChangesToEdge(successor);
        newEdges[i] =
            applyNoiseEffects(successor, usedQubits, false, nextLevel);
        dEdge::revertDmChangesToEdge(successor);
      }
    }
  }
  if (std::any_of(
          usedQubits.begin(), usedQubits.end(),
          [&nextLevel](const qc::Qubit qubit) { return nextLevel == qubit; })) {
    for (auto const& type : noiseEffects) {
      switch (type) {
      case AmplitudeDamping:
        applyAmplitudeDampingToEdges(newEdges, (usedQubits.size() == 1)
                                                   ? ampDampingProbSingleQubit
                                                   : ampDampingProbMultiQubit);
        break;
      case PhaseFlip:
        applyPhaseFlipToEdges(newEdges, (usedQubits.size() == 1)
                                            ? noiseProbSingleQubit
                                            : noiseProbMultiQubit);
        break;
      case Depolarization:
        applyDepolarisationToEdges(newEdges, (usedQubits.size() == 1)
                                                 ? noiseProbSingleQubit
                                                 : noiseProbMultiQubit);
        break;
      case Identity:
        continue;
      }
    }
  }

  auto e = package->makeDDNode(nextLevel, newEdges, firstPathEdge);
  if (e.w.exactlyZero()) {
    return e;
  }
  e.w = e.w * originalWeight;
  return e;
}

void DeterministicNoiseFunctionality::applyPhaseFlipToEdges(
    ArrayOfEdges& e, const double probability) {
  const auto complexProb = 1. - 2. * probability;

  // e[0] = e[0]
  // e[1] = (1-2p)*e[1]
  if (!e[1].w.exactlyZero()) {
    e[1].w *= complexProb;
  }
  // e[2] = (1-2p)*e[2]
  if (!e[2].w.exactlyZero()) {
    e[2].w *= complexProb;
  }
  // e[3] = e[3]
}

void DeterministicNoiseFunctionality::applyAmplitudeDampingToEdges(
    ArrayOfEdges& e, const double probability) {
  // e[0] = e[0] + p*e[3]
  if (!e[3].w.exactlyZero()) {
    if (!e[0].w.exactlyZero()) {
      const auto var = static_cast<Qubit>(std::max(
          {e[0].p != nullptr ? e[0].p->v : 0, e[1].p != nullptr ? e[1].p->v : 0,
           e[2].p != nullptr ? e[2].p->v : 0,
           e[3].p != nullptr ? e[3].p->v : 0}));
      e[0] = package->add2(e[0], {e[3].p, e[3].w * probability}, var);
    } else {
      e[0] = {e[3].p, e[3].w * probability};
    }
  }

  // e[1] = sqrt(1-p)*e[1]
  if (!e[1].w.exactlyZero()) {
    e[1].w *= std::sqrt(1 - probability);
  }

  // e[2] = sqrt(1-p)*e[2]
  if (!e[2].w.exactlyZero()) {
    e[2].w *= std::sqrt(1 - probability);
  }

  // e[3] = (1-p)*e[3]
  if (!e[3].w.exactlyZero()) {
    e[3].w *= (1 - probability);
  }
}

void DeterministicNoiseFunctionality::applyDepolarisationToEdges(
    ArrayOfEdges& e, const double probability) {
  std::array<dCachedEdge, 2> helperEdge{};

  const auto var = static_cast<Qubit>(std::max(
      {e[0].p != nullptr ? e[0].p->v : 0, e[1].p != nullptr ? e[1].p->v : 0,
       e[2].p != nullptr ? e[2].p->v : 0, e[3].p != nullptr ? e[3].p->v : 0}));

  auto oldE0Edge = e[0];

  // e[0] = 0.5*((2-p)*e[0] + p*e[3])
  {
    // helperEdge[0] = 0.5*((2-p)*e[0]
    helperEdge[0].p = e[0].p;
    if (!e[0].w.exactlyZero()) {
      helperEdge[0].w = e[0].w * (2 - probability) * 0.5;
    } else {
      helperEdge[0].w = 0;
    }

    // helperEdge[1] = 0.5*p*e[3]
    helperEdge[1].p = e[3].p;
    if (!e[3].w.exactlyZero()) {
      helperEdge[1].w = e[3].w * probability * 0.5;
    } else {
      helperEdge[1].w = 0;
    }

    // e[0] = helperEdge[0] + helperEdge[1]
    e[0] = package->add2(helperEdge[0], helperEdge[1], var);
  }

  // e[1]=(1-p)*e[1]
  if (!e[1].w.exactlyZero()) {
    e[1].w *= (1 - probability);
  }
  // e[2]=(1-p)*e[2]
  if (!e[2].w.exactlyZero()) {
    e[2].w *= (1 - probability);
  }

  // e[3] = 0.5*((2-p)*e[3]) + 0.5*(p*e[0])
  {
    // helperEdge[0] = 0.5*((2-p)*e[3])
    helperEdge[0].p = e[3].p;
    if (!e[3].w.exactlyZero()) {
      helperEdge[0].w = e[3].w * (2 - probability) * 0.5;
    } else {
      helperEdge[0].w = 0;
    }

    // helperEdge[1] = 0.5*p*e[0]
    helperEdge[1].p = oldE0Edge.p;
    if (!oldE0Edge.w.exactlyZero()) {
      helperEdge[1].w = oldE0Edge.w * probability * 0.5;
    } else {
      helperEdge[1].w = 0;
    }
    e[3] = package->add2(helperEdge[0], helperEdge[1], var);
  }
}

void sanityCheckOfNoiseProbabilities(const double noiseProbability,
                                     const double amplitudeDampingProb,
                                     const double multiQubitGateFactor) {
  if (noiseProbability < 0 || amplitudeDampingProb < 0 ||
      noiseProbability * multiQubitGateFactor > 1 ||
      amplitudeDampingProb * multiQubitGateFactor > 1) {
    throw std::runtime_error(
        "Error probabilities are faulty!"
        "\n single qubit error probability: " +
        std::to_string(noiseProbability) + " multi qubit error probability: " +
        std::to_string(noiseProbability * multiQubitGateFactor) +
        "\n single qubit amplitude damping  probability: " +
        std::to_string(amplitudeDampingProb) +
        " multi qubit amplitude damping  probability: " +
        std::to_string(amplitudeDampingProb * multiQubitGateFactor));
  }
}
} // namespace dd
