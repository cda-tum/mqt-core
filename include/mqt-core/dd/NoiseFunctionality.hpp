#pragma once

#include "dd/ComplexNumbers.hpp"
#include "dd/GateMatrixDefinitions.hpp"
#include "dd/Node.hpp"
#include "dd/Package.hpp"
#include "operations/OpType.hpp"
#include "operations/Operation.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <complex>
#include <map>
#include <optional>
#include <random>
#include <utility>
#include <vector>

namespace dd {

using NrEdges = std::tuple_size<decltype(dNode::e)>;
using ArrayOfEdges = std::array<dCachedEdge, NrEdges::value>;

// noise operations available for deterministic noise aware quantum circuit
// simulation
enum NoiseOperations : std::uint8_t {
  AmplitudeDamping,
  PhaseFlip,
  Depolarization,
  Identity
};

void sanityCheckOfNoiseProbabilities(double noiseProbability,
                                     double amplitudeDampingProb,
                                     double multiQubitGateFactor);

class StochasticNoiseFunctionality {
public:
  StochasticNoiseFunctionality(
      const std::unique_ptr<Package<StochasticNoiseSimulatorDDPackageConfig>>&
          dd,
      std::size_t nq, double gateNoiseProbability, double amplitudeDampingProb,
      double multiQubitGateFactor, const std::string& cNoiseEffects);

  ~StochasticNoiseFunctionality() { package->decRef(identityDD); }

protected:
  Package<StochasticNoiseSimulatorDDPackageConfig>* package;
  std::size_t nQubits;
  std::uniform_real_distribution<fp> dist;

  double noiseProbability;
  double noiseProbabilityMulti;
  fp sqrtAmplitudeDampingProbability;
  fp oneMinusSqrtAmplitudeDampingProbability;
  fp sqrtAmplitudeDampingProbabilityMulti;
  fp oneMinusSqrtAmplitudeDampingProbabilityMulti;
  GateMatrix ampDampingTrue{};
  GateMatrix ampDampingTrueMulti{};
  GateMatrix ampDampingFalse{};
  GateMatrix ampDampingFalseMulti{};
  std::vector<NoiseOperations> noiseEffects;
  mEdge identityDD;

  [[nodiscard]] std::size_t getNumberOfQubits() const { return nQubits; }
  [[nodiscard]] double getNoiseProbability(bool multiQubitNoiseFlag) const;

  [[nodiscard]] static qc::OpType
  getAmplitudeDampingOperationType(bool multiQubitNoiseFlag,
                                   bool amplitudeDampingFlag);

  [[nodiscard]] GateMatrix
  getAmplitudeDampingOperationMatrix(bool multiQubitNoiseFlag,
                                     bool amplitudeDampingFlag) const;

public:
  void applyNoiseOperation(const std::set<qc::Qubit>& targets, mEdge operation,
                           vEdge& state, std::mt19937_64& generator);

protected:
  [[nodiscard]] mEdge stackOperation(mEdge operation, qc::Qubit target,
                                     qc::OpType noiseOperation,
                                     GateMatrix matrix);

  mEdge generateNoiseOperation(mEdge operation, qc::Qubit target,
                               std::mt19937_64& generator,
                               bool amplitudeDamping, bool multiQubitOperation);

  [[nodiscard]] qc::OpType returnNoiseOperation(NoiseOperations noiseOperation,
                                                double prob,
                                                bool multiQubitNoiseFlag) const;
};

class DeterministicNoiseFunctionality {
public:
  DeterministicNoiseFunctionality(
      const std::unique_ptr<Package<DensityMatrixSimulatorDDPackageConfig>>& dd,
      std::size_t nq, double noiseProbabilitySingleQubit,
      double noiseProbabilityMultiQubit, double ampDampProbSingleQubit,
      double ampDampProbMultiQubit, const std::string& cNoiseEffects);

protected:
  Package<DensityMatrixSimulatorDDPackageConfig>* package;
  std::size_t nQubits;

  double noiseProbSingleQubit;
  double noiseProbMultiQubit;
  double ampDampingProbSingleQubit;
  double ampDampingProbMultiQubit;

  std::vector<NoiseOperations> noiseEffects;

  [[nodiscard]] std::size_t getNumberOfQubits() const { return nQubits; }

public:
  void applyNoiseEffects(dEdge& originalEdge,
                         const std::unique_ptr<qc::Operation>& qcOperation);

private:
  dCachedEdge applyNoiseEffects(dEdge& originalEdge,
                                const std::set<qc::Qubit>& usedQubits,
                                bool firstPathEdge, Qubit level);

  static void applyPhaseFlipToEdges(ArrayOfEdges& e, double probability);

  void applyAmplitudeDampingToEdges(ArrayOfEdges& e, double probability);

  void applyDepolarisationToEdges(ArrayOfEdges& e, double probability);
};

} // namespace dd
