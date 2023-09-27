#pragma once

#include "dd/Operations.hpp"

#include <random>
#include <utility>

namespace dd {

using CN = ComplexNumbers;
using NrEdges = std::tuple_size<decltype(dNode::e)>;
using ArrayOfEdges = std::array<qc::DensityMatrixDD, NrEdges::value>;

// noise operations available for deterministic noise aware quantum circuit
// simulation
enum NoiseOperations : std::uint8_t {
  AmplitudeDamping,
  PhaseFlip,
  Depolarization,
  Identity
};

template <class Config> class StochasticNoiseFunctionality {
public:
  StochasticNoiseFunctionality(const std::unique_ptr<Package<Config>>& dd,
                               const std::size_t nq,
                               double gateNoiseProbability,
                               double amplitudeDampingProb,
                               double multiQubitGateFactor,
                               std::vector<NoiseOperations> effects)
      : package(dd), nQubits(nq), dist(0.0, 1.0L),
        noiseProbability(gateNoiseProbability),
        noiseProbabilityMulti(gateNoiseProbability * multiQubitGateFactor),
        sqrtAmplitudeDampingProbability({std::sqrt(amplitudeDampingProb), 0}),
        oneMinusSqrtAmplitudeDampingProbability(
            {std::sqrt(1 - amplitudeDampingProb), 0}),
        sqrtAmplitudeDampingProbabilityMulti(
            {std::sqrt(gateNoiseProbability) * multiQubitGateFactor, 0}),
        oneMinusSqrtAmplitudeDampingProbabilityMulti(
            {std::sqrt(1 - multiQubitGateFactor * amplitudeDampingProb), 0}),
        ampDampingTrue(
            GateMatrix({complex_zero, sqrtAmplitudeDampingProbability,
                        complex_zero, complex_zero})),
        ampDampingTrueMulti(
            GateMatrix({complex_zero, sqrtAmplitudeDampingProbabilityMulti,
                        complex_zero, complex_zero})),
        ampDampingFalse(GateMatrix({complex_one, complex_zero, complex_zero,
                                    oneMinusSqrtAmplitudeDampingProbability})),
        ampDampingFalseMulti(
            GateMatrix({complex_one, complex_zero, complex_zero,
                        oneMinusSqrtAmplitudeDampingProbabilityMulti})),
        noiseEffects(std::move(effects)),
        identityDD(package->makeIdent(nQubits)) {
    package->incRef(identityDD);
  }

  ~StochasticNoiseFunctionality() { package->decRef(identityDD); }

protected:
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  const std::unique_ptr<Package<Config>>& package;
  std::size_t nQubits;
  std::uniform_real_distribution<fp> dist;

  double noiseProbability;
  double noiseProbabilityMulti;
  ComplexValue sqrtAmplitudeDampingProbability;
  ComplexValue oneMinusSqrtAmplitudeDampingProbability;
  ComplexValue sqrtAmplitudeDampingProbabilityMulti;
  ComplexValue oneMinusSqrtAmplitudeDampingProbabilityMulti;
  GateMatrix ampDampingTrue{};
  GateMatrix ampDampingTrueMulti{};
  GateMatrix ampDampingFalse{};
  GateMatrix ampDampingFalseMulti{};
  std::vector<NoiseOperations> noiseEffects;
  mEdge identityDD;

  [[nodiscard]] std::size_t getNumberOfQubits() const { return nQubits; }
  [[nodiscard]] double getNoiseProbability(bool multiQubitNoiseFlag) const {
    return multiQubitNoiseFlag ? noiseProbabilityMulti : noiseProbability;
  }

  [[nodiscard]] qc::OpType
  getAmplitudeDampingOperationType(bool multiQubitNoiseFlag,
                                   bool amplitudeDampingFlag) const {
    if (amplitudeDampingFlag) {
      return multiQubitNoiseFlag ? qc::MultiATrue : qc::ATrue;
    }
    return multiQubitNoiseFlag ? qc::MultiAFalse : qc::AFalse;
  }

  [[nodiscard]] GateMatrix
  getAmplitudeDampingOperationMatrix(bool multiQubitNoiseFlag,
                                     bool amplitudeDampingFlag) const {
    if (amplitudeDampingFlag) {
      return multiQubitNoiseFlag ? ampDampingTrueMulti : ampDampingTrue;
    }
    return multiQubitNoiseFlag ? ampDampingFalseMulti : ampDampingFalse;
  }

public:
  [[nodiscard]] mEdge getIdentityDD() const { return identityDD; }
  void setNoiseEffects(std::vector<NoiseOperations> newNoiseEffects) {
    noiseEffects = std::move(newNoiseEffects);
  }

  void applyNoiseOperation(const std::set<qc::Qubit>& targets, mEdge operation,
                           vEdge& state, std::mt19937_64& generator) {
    const bool multiQubitOperation = targets.size() > 1;

    for (const auto& target : targets) {
      auto stackedOperation = generateNoiseOperation(
          operation, target, generator, false, multiQubitOperation);
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
      tmp.w = Complex::one;

      package->incRef(tmp);
      package->decRef(state);
      state = tmp;

      // I only need to apply the operations once
      operation = identityDD;
    }
  }

protected:
  [[nodiscard]] mEdge stackOperation(mEdge operation, const qc::Qubit target,
                                     const qc::OpType noiseOperation,
                                     const GateMatrix matrix) {
    if (const auto* op = package->stochasticNoiseOperationCache.lookup(
            noiseOperation, target);
        op != nullptr) {
      return package->multiply(*op, operation);
    }
    const auto gateDD =
        package->makeGateDD(matrix, getNumberOfQubits(), target);
    package->stochasticNoiseOperationCache.insert(noiseOperation, target,
                                                  gateDD);
    return package->multiply(gateDD, operation);
  }

  mEdge generateNoiseOperation(mEdge operation, qc::Qubit target,
                               std::mt19937_64& generator,
                               bool amplitudeDamping,
                               bool multiQubitOperation) {
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
        operation = stackOperation(operation, target, effect, Xmat);
        break;
      }
      case (qc::Y): {
        operation = stackOperation(operation, target, effect, Ymat);
        break;
      }
      case (qc::Z): {
        operation = stackOperation(operation, target, effect, Zmat);
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

  [[nodiscard]] qc::OpType
  returnNoiseOperation(NoiseOperations noiseOperation, double prob,
                       bool multiQubitNoiseFlag) const {
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
};

template <class Config> class DeterministicNoiseFunctionality {
public:
  DeterministicNoiseFunctionality(const std::unique_ptr<Package<Config>>& dd,
                                  const std::size_t nq,
                                  double noiseProbabilitySingleQubit,
                                  double noiseProbabilityMultiQubit,
                                  double ampDampProbSingleQubit,
                                  double ampDampProbMultiQubit,
                                  std::vector<NoiseOperations> effects,
                                  bool useDensityMatType, bool seqApplyNoise)
      : package(dd), nQubits(nq),
        noiseProbSingleQubit(noiseProbabilitySingleQubit),
        noiseProbMultiQubit(noiseProbabilityMultiQubit),
        ampDampingProbSingleQubit(ampDampProbSingleQubit),
        ampDampingProbMultiQubit(ampDampProbMultiQubit),
        noiseEffects(std::move(effects)),
        useDensityMatrixType(useDensityMatType),
        sequentiallyApplyNoise(seqApplyNoise) {}

protected:
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  const std::unique_ptr<Package<Config>>& package;
  std::size_t nQubits;

  double noiseProbSingleQubit;
  double noiseProbMultiQubit;
  double ampDampingProbSingleQubit;
  double ampDampingProbMultiQubit;

  std::vector<NoiseOperations> noiseEffects;
  bool useDensityMatrixType;
  bool sequentiallyApplyNoise;

  inline static const std::map<NoiseOperations, std::size_t>
      SEQUENTIAL_NOISE_MAP = {
          {Identity, 1},         // Identity Noise
          {PhaseFlip, 2},        // Phase-flip
          {AmplitudeDamping, 2}, // Amplitude Damping
          {Depolarization, 4},   // Depolarisation
  };

  [[nodiscard]] std::size_t getNumberOfQubits() const { return nQubits; }

public:
  void applyNoiseEffects(qc::DensityMatrixDD& originalEdge,
                         const std::unique_ptr<qc::Operation>& qcOperation) {
    auto usedQubits = qcOperation->getUsedQubits();

    [[maybe_unused]] const auto cacheSizeBefore = package->cn.cacheCount();

    if (sequentiallyApplyNoise) {
      applyDetNoiseSequential(originalEdge, usedQubits);
    } else {
      qc::DensityMatrixDD nodeAfterNoise = {};
      if (useDensityMatrixType) {
        qc::DensityMatrixDD::applyDmChangesToEdge(originalEdge);
        nodeAfterNoise = applyNoiseEffects(originalEdge, usedQubits, false);
        qc::DensityMatrixDD::revertDmChangesToEdge(originalEdge);
      } else {
        nodeAfterNoise = applyNoiseEffects(originalEdge, usedQubits, true);
      }
      nodeAfterNoise.w = package->cn.lookup(nodeAfterNoise.w, true);

      package->incRef(nodeAfterNoise);
      qc::DensityMatrixDD::alignDensityEdge(originalEdge);
      package->decRef(originalEdge);
      originalEdge = nodeAfterNoise;
      if (useDensityMatrixType) {
        qc::DensityMatrixDD::setDensityMatrixTrue(originalEdge);
      }
    }
    [[maybe_unused]] const auto cacheSizeAfter = package->cn.cacheCount();
    assert(cacheSizeAfter == cacheSizeBefore);
  }

private:
  qc::DensityMatrixDD applyNoiseEffects(qc::DensityMatrixDD& originalEdge,
                                        const std::set<qc::Qubit>& usedQubits,
                                        bool firstPathEdge) {
    if (originalEdge.isTerminal() || originalEdge.p->v < *usedQubits.begin()) {
      if (ComplexNumbers::isStaticComplex(originalEdge.w)) {
        return originalEdge;
      }
      return {originalEdge.p, package->cn.getCached(originalEdge.w)};
    }

    auto originalCopy = qc::DensityMatrixDD{originalEdge.p, Complex::one};
    ArrayOfEdges newEdges{};
    for (size_t i = 0; i < newEdges.size(); i++) {
      auto& successor = originalCopy.p->e[i];
      if (firstPathEdge || i == 1) {
        // If I am to the firstPathEdge I cannot minimize the necessary
        // operations anymore
        qc::DensityMatrixDD::applyDmChangesToEdge(successor);
        newEdges[i] = applyNoiseEffects(successor, usedQubits, true);
        qc::DensityMatrixDD::revertDmChangesToEdge(successor);
      } else if (i == 2) {
        // Since e[1] == e[2] (due to density matrix representation), I can skip
        // calculating e[2]
        newEdges[2].p = newEdges[1].p;
        newEdges[2].w = package->cn.getCached(newEdges[1].w);
      } else {
        qc::DensityMatrixDD::applyDmChangesToEdge(successor);
        newEdges[i] = applyNoiseEffects(successor, usedQubits, false);
        qc::DensityMatrixDD::revertDmChangesToEdge(successor);
      }
    }
    qc::DensityMatrixDD e = {};
    if (std::any_of(usedQubits.begin(), usedQubits.end(),
                    [originalEdge](const qc::Qubit qubit) {
                      return originalEdge.p->v == qubit;
                    })) {
      for (auto const& type : noiseEffects) {
        switch (type) {
        case AmplitudeDamping:
          applyAmplitudeDampingToEdges(
              newEdges, (usedQubits.size() == 1) ? ampDampingProbSingleQubit
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

    e = package->makeDDNode(originalCopy.p->v, newEdges, true, firstPathEdge);

    // Multiplying the old edge weight with the new one
    if (!e.w.exactlyZero()) {
      if (e.w.exactlyOne()) {
        e.w = package->cn.getCached(originalEdge.w);
      } else {
        CN::mul(e.w, e.w, originalEdge.w);
      }
    }
    return e;
  }

  void applyPhaseFlipToEdges(ArrayOfEdges& e, double probability) {
    auto complexProb = package->cn.getCached(1. - 2. * probability, 0.);

    // e[0] = e[0]
    // e[1] = (1-2p)*e[1]
    if (!e[1].w.approximatelyZero()) {
      CN::mul(e[1].w, complexProb, e[1].w);
    }
    // e[2] = (1-2p)*e[2]
    if (!e[2].w.approximatelyZero()) {
      CN::mul(e[2].w, complexProb, e[2].w);
    }
    // e[3] = e[3]

    package->cn.returnToCache(complexProb);
  }

  void applyAmplitudeDampingToEdges(ArrayOfEdges& e, double probability) {
    Complex complexProb = package->cn.getCached(0., 0.);

    // e[0] = e[0] + p*e[3]
    if (!e[3].w.exactlyZero()) {
      complexProb.r->value = probability;
      if (!e[0].w.exactlyZero()) {
        const auto w = package->cn.mulCached(complexProb, e[3].w);
        const auto var =
            static_cast<Qubit>(std::max({e[0].p != nullptr ? e[0].p->v : 0,
                                         e[1].p != nullptr ? e[1].p->v : 0,
                                         e[2].p != nullptr ? e[2].p->v : 0,
                                         e[3].p != nullptr ? e[3].p->v : 0}));
        const auto tmp = package->add2(e[0], {e[3].p, w}, var);
        package->cn.returnToCache(w);
        package->cn.returnToCache(e[0].w);
        e[0] = tmp;
      } else {
        // e[0].w is exactly zero therefore I need to get a new cached value
        e[0].w = package->cn.mulCached(complexProb, e[3].w);
        e[0].p = e[3].p;
      }
    }

    // e[1] = sqrt(1-p)*e[1]
    if (!e[1].w.exactlyZero()) {
      complexProb.r->value = std::sqrt(1 - probability);
      if (e[1].w.exactlyOne()) {
        e[1].w = package->cn.getCached(complexProb);
      } else {
        CN::mul(e[1].w, complexProb, e[1].w);
      }
    }

    // e[2] = sqrt(1-p)*e[2]
    if (!e[2].w.exactlyZero()) {
      if (e[1].w.exactlyZero()) {
        complexProb.r->value = std::sqrt(1 - probability);
      }
      if (e[2].w.exactlyOne()) {
        e[2].w = package->cn.getCached(complexProb);
      } else {
        CN::mul(e[2].w, complexProb, e[2].w);
      }
    }

    // e[3] = (1-p)*e[3]
    if (!e[3].w.exactlyZero()) {
      complexProb.r->value = 1 - probability;
      if (e[3].w.exactlyOne()) {
        e[3].w = package->cn.getCached(complexProb);
      } else {
        CN::mul(e[3].w, complexProb, e[3].w);
      }
    }
    package->cn.returnToCache(complexProb);
  }

  void applyDepolarisationToEdges(ArrayOfEdges& e, double probability) {
    std::array<qc::DensityMatrixDD, 2> helperEdge{};
    Complex complexProb = package->cn.getCached();
    complexProb.i->value = 0;

    const auto var = static_cast<Qubit>(std::max(
        {e[0].p != nullptr ? e[0].p->v : 0, e[1].p != nullptr ? e[1].p->v : 0,
         e[2].p != nullptr ? e[2].p->v : 0,
         e[3].p != nullptr ? e[3].p->v : 0}));

    qc::DensityMatrixDD oldE0Edge{e[0].p, package->cn.getCached(e[0].w)};

    // e[0] = 0.5*((2-p)*e[0] + p*e[3])
    {
      // helperEdge[0] = 0.5*((2-p)*e[0]
      helperEdge[0].p = e[0].p;
      if (!e[0].w.exactlyZero()) {
        complexProb.r->value = (2 - probability) * 0.5;
        helperEdge[0].w = package->cn.mulCached(e[0].w, complexProb);
      } else {
        helperEdge[0].w = Complex::zero;
      }

      // helperEdge[1] = 0.5*p*e[3]
      helperEdge[1].p = e[3].p;
      if (!e[3].w.exactlyZero()) {
        complexProb.r->value = probability * 0.5;
        helperEdge[1].w = package->cn.mulCached(e[3].w, complexProb);
      } else {
        helperEdge[1].w = Complex::zero;
      }

      // e[0] = helperEdge[0] + helperEdge[1]
      package->cn.returnToCache(e[0].w);
      e[0] = package->add2(helperEdge[0], helperEdge[1], var);
      package->cn.returnToCache(helperEdge[0].w);
      package->cn.returnToCache(helperEdge[1].w);
    }

    // e[1]=(1-p)*e[1]
    if (!e[1].w.exactlyZero()) {
      complexProb.r->value = 1 - probability;
      if (e[1].w.exactlyOne()) {
        e[1].w = package->cn.getCached(complexProb);
      } else {
        CN::mul(e[1].w, e[1].w, complexProb);
      }
    }
    // e[2]=(1-p)*e[2]
    if (!e[2].w.exactlyZero()) {
      if (e[1].w.exactlyZero()) {
        complexProb.r->value = 1 - probability;
      }
      if (e[2].w.exactlyOne()) {
        e[2].w = package->cn.getCached(complexProb);
      } else {
        CN::mul(e[2].w, e[2].w, complexProb);
      }
    }

    // e[3] = 0.5*((2-p)*e[3]) + 0.5*(p*e[0])
    {
      // helperEdge[0] = 0.5*((2-p)*e[3])
      helperEdge[0].p = e[3].p;
      if (!e[3].w.exactlyZero()) {
        complexProb.r->value = (2 - probability) * 0.5;
        helperEdge[0].w = package->cn.mulCached(e[3].w, complexProb);
      } else {
        helperEdge[0].w = Complex::zero;
      }

      // helperEdge[1] = 0.5*p*e[0]
      helperEdge[1].p = oldE0Edge.p;
      if (!oldE0Edge.w.exactlyZero()) {
        complexProb.r->value = probability * 0.5;
        helperEdge[1].w = package->cn.mulCached(oldE0Edge.w, complexProb);
      } else {
        helperEdge[1].w = Complex::zero;
      }

      package->cn.returnToCache(e[3].w);
      e[3] = package->add2(helperEdge[0], helperEdge[1], var);
      package->cn.returnToCache(helperEdge[0].w);
      package->cn.returnToCache(helperEdge[1].w);
    }
    package->cn.returnToCache(oldE0Edge.w);
    package->cn.returnToCache(complexProb);
  }

  void applyDetNoiseSequential(qc::DensityMatrixDD& originalEdge,
                               const std::set<qc::Qubit>& targets) {
    qc::DensityMatrixDD tmp = {};

    std::array<mEdge, NrEdges::value> idleOperation{};

    // Iterate over qubits and check if the qubit had been used
    for (const auto targetQubit : targets) {
      for (auto const& type : noiseEffects) {
        generateGate(idleOperation, type, targetQubit,
                     getNoiseProbability(type, targets));
        tmp.p = nullptr;
        // Apply all noise matrices of the current noise effect
        for (std::size_t m = 0; m < SEQUENTIAL_NOISE_MAP.find(type)->second;
             m++) {
          auto tmp0 = package->conjugateTranspose(idleOperation.at(m));
          auto tmp1 = package->multiply(originalEdge,
                                        densityFromMatrixEdge(tmp0), 0, false);
          auto tmp2 =
              package->multiply(densityFromMatrixEdge(idleOperation.at(m)),
                                tmp1, 0, useDensityMatrixType);
          if (tmp.p == nullptr) {
            tmp = tmp2;
          } else {
            tmp = package->add(tmp2, tmp);
          }
        }
        package->incRef(tmp);
        qc::DensityMatrixDD::alignDensityEdge(originalEdge);
        package->decRef(originalEdge);
        originalEdge = tmp;
        if (useDensityMatrixType) {
          qc::DensityMatrixDD::setDensityMatrixTrue(originalEdge);
        }
      }
    }
  }

  void generateDepolarizationGate(
      std::array<mEdge, NrEdges::value>& pointerForMatrices,
      const qc::Qubit target, const double probability) {
    std::array<GateMatrix, NrEdges::value> idleNoiseGate{};
    ComplexValue tmp = {};

    tmp.r = std::sqrt(1 - ((3 * probability) / 4)) * complex_one.r;
    //                   (1 0)
    // sqrt(1- ((3p)/4))*(0 1)
    idleNoiseGate[0][0] = idleNoiseGate[0][3] = tmp;
    idleNoiseGate[0][1] = idleNoiseGate[0][2] = complex_zero;

    pointerForMatrices[0] =
        package->makeGateDD(idleNoiseGate[0], getNumberOfQubits(), target);

    //                      (0 1)
    // sqrt(probability/4))*(1 0)
    tmp.r = std::sqrt(probability / 4) * complex_one.r;
    idleNoiseGate[1][1] = idleNoiseGate[1][2] = tmp;
    idleNoiseGate[1][0] = idleNoiseGate[1][3] = complex_zero;

    pointerForMatrices[1] =
        package->makeGateDD(idleNoiseGate[1], getNumberOfQubits(), target);

    //                      (1 0)
    // sqrt(probability/4))*(0 -1)
    tmp.r = std::sqrt(probability / 4) * complex_one.r;
    idleNoiseGate[2][0] = tmp;
    tmp.r = tmp.r * -1;
    idleNoiseGate[2][3] = tmp;
    idleNoiseGate[2][1] = idleNoiseGate[2][2] = complex_zero;

    pointerForMatrices[3] =
        package->makeGateDD(idleNoiseGate[2], getNumberOfQubits(), target);

    //                      (0 -i)
    // sqrt(probability/4))*(i 0)
    tmp.r = complex_zero.r;
    tmp.i = std::sqrt(probability / 4) * 1;
    idleNoiseGate[3][2] = tmp;
    tmp.i = tmp.i * -1;
    idleNoiseGate[3][1] = tmp;
    idleNoiseGate[3][0] = idleNoiseGate[3][3] = complex_zero;

    pointerForMatrices[2] =
        package->makeGateDD(idleNoiseGate[3], getNumberOfQubits(), target);
  }

  void generateGate(std::array<mEdge, NrEdges::value>& pointerForMatrices,
                    const NoiseOperations noiseType, const qc::Qubit target,
                    const double probability) {
    std::array<GateMatrix, NrEdges::value> idleNoiseGate{};
    ComplexValue tmp = {};

    switch (noiseType) {
      // identity noise (for testing)
      //                  (1  0)
      //                  (0  1),
    case Identity: {
      idleNoiseGate[0][0] = idleNoiseGate[0][3] = complex_one;
      idleNoiseGate[0][1] = idleNoiseGate[0][2] = complex_zero;

      pointerForMatrices[0] =
          package->makeGateDD(idleNoiseGate[0], getNumberOfQubits(), target);

      break;
    }
      // phase flip
      //                          (1  0)                         (1  0)
      //  e0= sqrt(1-probability)*(0  1), e1=  sqrt(probability)*(0 -1)
    case PhaseFlip: {
      tmp.r = std::sqrt(1 - probability) * complex_one.r;
      idleNoiseGate[0][0] = idleNoiseGate[0][3] = tmp;
      idleNoiseGate[0][1] = idleNoiseGate[0][2] = complex_zero;
      tmp.r = std::sqrt(probability) * complex_one.r;
      idleNoiseGate[1][0] = tmp;
      tmp.r *= -1;
      idleNoiseGate[1][3] = tmp;
      idleNoiseGate[1][1] = idleNoiseGate[1][2] = complex_zero;

      pointerForMatrices[0] =
          package->makeGateDD(idleNoiseGate[0], getNumberOfQubits(), target);
      pointerForMatrices[1] =
          package->makeGateDD(idleNoiseGate[1], getNumberOfQubits(), target);

      break;
    }
      // amplitude damping
      //      (1                  0)       (0      sqrt(probability))
      //  e0= (0 sqrt(1-probability), e1=  (0                      0)
    case AmplitudeDamping: {
      tmp.r = std::sqrt(1 - probability) * complex_one.r;
      idleNoiseGate[0][0] = complex_one;
      idleNoiseGate[0][1] = idleNoiseGate[0][2] = complex_zero;
      idleNoiseGate[0][3] = tmp;

      tmp.r = std::sqrt(probability) * complex_one.r;
      idleNoiseGate[1][0] = idleNoiseGate[1][3] = idleNoiseGate[1][2] =
          complex_zero;
      idleNoiseGate[1][1] = tmp;

      pointerForMatrices[0] =
          package->makeGateDD(idleNoiseGate[0], getNumberOfQubits(), target);
      pointerForMatrices[1] =
          package->makeGateDD(idleNoiseGate[1], getNumberOfQubits(), target);
      break;
    }
      // depolarization
    case Depolarization:
      generateDepolarizationGate(pointerForMatrices, target, probability);
      break;
    default:
      throw std::runtime_error("Unknown noise effect received.");
    }
  }

  double getNoiseProbability(const NoiseOperations type,
                             const std::set<qc::Qubit>& targets) {
    if (type == AmplitudeDamping) {
      return (targets.size() == 1) ? ampDampingProbSingleQubit
                                   : ampDampingProbMultiQubit;
    }
    return (targets.size() == 1) ? noiseProbSingleQubit : noiseProbMultiQubit;
  }
};

} // namespace dd
