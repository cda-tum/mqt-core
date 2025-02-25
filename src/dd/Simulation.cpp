/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "dd/Simulation.hpp"

#include "Definitions.hpp"
#include "dd/DDDefinitions.hpp"
#include "dd/GateMatrixDefinitions.hpp"
#include "dd/Package.hpp"
#include "dd/RealNumber.hpp"
#include "ir/QuantumComputation.hpp"
#include "ir/operations/ClassicControlledOperation.hpp"
#include "ir/operations/NonUnitaryOperation.hpp"
#include "ir/operations/OpType.hpp"

#include <array>
#include <cstddef>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace dd {
template <class Config>
std::map<std::string, std::size_t>
sample(const QuantumComputation& qc, const VectorDD& in, Package<Config>& dd,
       const std::size_t shots, const std::size_t seed) {
  auto isDynamicCircuit = false;
  auto hasMeasurements = false;
  auto measurementsLast = true;

  std::mt19937_64 mt{};
  if (seed != 0U) {
    mt.seed(seed);
  } else {
    // create and properly seed rng
    std::array<std::mt19937_64::result_type, std::mt19937_64::state_size>
        randomData{};
    std::random_device rd;
    std::generate(std::begin(randomData), std::end(randomData),
                  [&rd]() { return rd(); });
    std::seed_seq seeds(std::begin(randomData), std::end(randomData));
    mt.seed(seeds);
  }

  std::map<qc::Qubit, std::size_t> measurementMap{};

  // rudimentary check whether circuit is dynamic
  for (const auto& op : qc) {
    // if it contains any dynamic circuit primitives, it certainly is dynamic
    if (op->isClassicControlledOperation() || op->getType() == qc::Reset) {
      isDynamicCircuit = true;
      break;
    }

    // once a measurement is encountered we store the corresponding mapping
    // (qubit -> bit)
    if (const auto* measure = dynamic_cast<qc::NonUnitaryOperation*>(op.get());
        measure != nullptr && measure->getType() == qc::Measure) {
      hasMeasurements = true;

      const auto& quantum = measure->getTargets();
      const auto& classic = measure->getClassics();

      for (std::size_t i = 0; i < quantum.size(); ++i) {
        measurementMap[quantum.at(i)] = classic.at(i);
      }
    }

    // if an operation happens after a measurement, the resulting circuit can
    // only be simulated in single shots
    if (hasMeasurements &&
        (op->isUnitary() || op->isClassicControlledOperation())) {
      measurementsLast = false;
    }
  }

  if (!measurementsLast) {
    isDynamicCircuit = true;
  }

  if (!isDynamicCircuit) {
    // if all gates are unitary (besides measurements at the end), we just
    // simulate once and measure all qubits repeatedly
    auto permutation = qc.initialLayout;
    auto e = in;

    for (const auto& op : qc) {
      // simply skip any non-unitary
      if (!op->isUnitary()) {
        continue;
      }

      // SWAP gates can be executed virtually by changing the permutation
      if (op->getType() == OpType::SWAP && !op->isControlled()) {
        const auto& targets = op->getTargets();
        std::swap(permutation.at(targets[0U]), permutation.at(targets[1U]));
        continue;
      }

      e = applyUnitaryOperation(*op, e, dd, permutation);
    }

    // correct permutation if necessary
    changePermutation(e, permutation, qc.outputPermutation, dd);
    e = dd.reduceGarbage(e, qc.getGarbage());

    // measure all qubits
    std::map<std::string, std::size_t> counts{};
    for (std::size_t i = 0U; i < shots; ++i) {
      // measure all returns a string of the form "q(n-1) ... q(0)"
      auto measurement = dd.measureAll(e, false, mt);
      counts.operator[](measurement) += 1U;
    }
    // reduce reference count of measured state
    dd.decRef(e);

    std::map<std::string, std::size_t> actualCounts{};
    const auto numBits =
        qc.getClassicalRegisters().empty() ? qc.getNqubits() : qc.getNcbits();
    for (const auto& [bitstring, count] : counts) {
      std::string measurement(numBits, '0');
      if (hasMeasurements) {
        // if the circuit contains measurements, we only want to return the
        // measured bits
        for (const auto& [qubit, bit] : measurementMap) {
          // measurement map specifies that the circuit `qubit` is measured into
          // a certain `bit`
          measurement[numBits - 1U - bit] =
              bitstring[bitstring.size() - 1U - qc.outputPermutation.at(qubit)];
        }
      } else {
        // otherwise, we consider the output permutation for determining where
        // to measure the qubits to
        for (const auto& [qubit, bit] : qc.outputPermutation) {
          measurement[numBits - 1U - bit] =
              bitstring[bitstring.size() - 1U - qubit];
        }
      }
      actualCounts[measurement] += count;
    }
    return actualCounts;
  }

  std::map<std::string, std::size_t> counts{};

  for (std::size_t i = 0U; i < shots; i++) {
    std::vector<bool> measurements(qc.getNcbits(), false);

    auto permutation = qc.initialLayout;
    auto e = in;
    dd.incRef(e);
    for (const auto& op : qc) {
      if (op->isUnitary()) {
        // SWAP gates can be executed virtually by changing the permutation
        if (op->getType() == OpType::SWAP && !op->isControlled()) {
          const auto& targets = op->getTargets();
          std::swap(permutation.at(targets[0U]), permutation.at(targets[1U]));
          continue;
        }

        e = applyUnitaryOperation(*op, e, dd, permutation);
        continue;
      }

      if (op->getType() == Measure) {
        const auto& measure = dynamic_cast<const NonUnitaryOperation&>(*op);
        e = applyMeasurement(measure, e, dd, mt, measurements, permutation);
        continue;
      }

      if (op->getType() == Reset) {
        const auto& reset = dynamic_cast<const NonUnitaryOperation&>(*op);
        e = applyReset(reset, e, dd, mt, permutation);
        continue;
      }

      if (op->isClassicControlledOperation()) {
        const auto& classic =
            dynamic_cast<const ClassicControlledOperation&>(*op);
        e = applyClassicControlledOperation(classic, e, dd, measurements,
                                            permutation);
        continue;
      }

      qc::unreachable();
    }

    // reduce reference count of measured state
    dd.decRef(e);

    std::string shot(qc.getNcbits(), '0');
    for (size_t bit = 0U; bit < qc.getNcbits(); ++bit) {
      if (measurements[bit]) {
        shot[qc.getNcbits() - bit - 1U] = '1';
      }
    }
    counts[shot]++;
  }
  return counts;
}

std::map<std::string, std::size_t> sample(const QuantumComputation& qc,
                                          const std::size_t shots,
                                          const std::size_t seed) {
  const auto nqubits = qc.getNqubits();
  auto dd = std::make_unique<dd::Package<>>(nqubits);
  return sample(qc, dd->makeZeroState(nqubits), *dd, shots, seed);
}

namespace {
template <class Config>
void extractProbabilityVectorRecursive(const QuantumComputation& qc,
                                       const VectorDD& currentState,
                                       decltype(qc.begin()) currentIt,
                                       Permutation& permutation,
                                       std::map<std::size_t, char> measurements,
                                       fp commonFactor, SparsePVec& probVector,
                                       Package<Config>& dd) {
  auto state = currentState;
  for (auto it = currentIt; it != qc.end(); ++it) {
    const auto& op = (*it);

    // any standard operation is applied here
    if (op->isUnitary()) {
      // SWAP gates can be executed virtually by changing the permutation
      if (op->getType() == OpType::SWAP && !op->isControlled()) {
        const auto& targets = op->getTargets();
        std::swap(permutation.at(targets[0U]), permutation.at(targets[1U]));
        continue;
      }
      state = applyUnitaryOperation(*op, state, dd, permutation);
      continue;
    }

    // check whether a classic controlled operations can be applied
    if (op->isClassicControlledOperation()) {
      const auto& classicControlled =
          dynamic_cast<const ClassicControlledOperation&>(*op);
      const auto& expectedValue = classicControlled.getExpectedValue();
      // determine the actual value from measurements
      qc::Bit actualValue = 0U;
      if (const auto& controlRegister = classicControlled.getControlRegister();
          controlRegister.has_value()) {
        const auto regStart = controlRegister->getStartIndex();
        const auto regSize = controlRegister->getSize();
        for (std::size_t j = 0; j < regSize; ++j) {
          if (measurements[regStart + j] == '1') {
            actualValue |= 1ULL << j;
          }
        }
      }
      if (const auto& controlBit = classicControlled.getControlBit();
          controlBit.has_value()) {
        actualValue = measurements[*controlBit] == '1' ? 1U : 0U;
      }

      // do not apply an operation if the value is not the expected one
      if (actualValue != expectedValue) {
        continue;
      }

      state = applyUnitaryOperation(*classicControlled.getOperation(), state,
                                    dd, permutation);
      continue;
    }

    if (op->getType() == Reset) {
      // a reset operation should only happen once a qubit has been measured,
      // i.e., the qubit is in a basis state thus the probabilities for 0 and 1
      // need to be determined if p(1) ~= 1, an X operation has to be applied to
      // the qubit if p(0) ~= 1, nothing has to be done if 0 < p(0), p(1) < 1,
      // an error should be raised

      const auto& targets = op->getTargets();
      if (targets.size() != 1) {
        throw qc::QFRException(
            "Resets on multiple qubits are currently not supported. Please "
            "split them into multiple single resets.");
      }
      const auto target = targets[0];
      auto [pzero, pone] = dd.determineMeasurementProbabilities(
          state, static_cast<Qubit>(permutation.at(target)));

      // normalize probabilities
      const auto norm = pzero + pone;
      pzero /= norm;
      pone /= norm;

      if (RealNumber::approximatelyEquals(pone, 1.)) {
        const qc::MatrixDD xGate =
            dd.makeGateDD(opToSingleQubitGateMatrix(qc::X),
                          static_cast<Qubit>(permutation.at(target)));
        state = dd.applyOperation(xGate, state);
        continue;
      }

      if (!RealNumber::approximatelyEquals(pzero, 1.)) {
        throw qc::QFRException("Reset on non basis state encountered. This is "
                               "not supported in this method.");
      }

      continue;
    }

    // measurements form splitting points in this extraction scheme
    if (op->getType() == qc::Measure) {
      const auto& measurement = dynamic_cast<const NonUnitaryOperation&>(*op);
      const auto& targets = measurement.getTargets();
      const auto& classics = measurement.getClassics();
      if (targets.size() != 1U || classics.size() != 1U) {
        throw qc::QFRException(
            "Measurements on multiple qubits are not supported right now. "
            "Split your measurements into individual operations.");
      }

      const auto target = targets[0];
      const auto bit = classics[0];

      // determine probabilities for this measurement
      auto [pzero, pone] = dd.determineMeasurementProbabilities(
          state, static_cast<Qubit>(permutation.at(target)));

      // normalize probabilities
      const auto norm = pzero + pone;
      pzero /= norm;
      pone /= norm;

      // base case -> determine the basis state from the measurement and safe
      // the probability
      if (measurements.size() == qc.getNcbits() - 1) {
        std::size_t idx0 = 0U;
        std::size_t idx1 = 0U;
        for (std::size_t i = 0U; i < qc.getNcbits(); ++i) {
          // if this is the qubit being measured and the result is one
          if (i == static_cast<std::size_t>(bit)) {
            idx1 |= (1ULL << i);
          } else {
            // sanity check
            auto findIt = measurements.find(i);
            if (findIt == measurements.end()) {
              throw qc::QFRException("No information on classical bit " +
                                     std::to_string(i));
            }
            // if i-th bit is set increase the index appropriately
            if (findIt->second == '1') {
              idx0 |= (1ULL << i);
              idx1 |= (1ULL << i);
            }
          }
        }
        const auto prob0 = commonFactor * pzero;
        if (!RealNumber::approximatelyZero(prob0)) {
          probVector[idx0] = prob0;
        }
        const auto prob1 = commonFactor * pone;
        if (!RealNumber::approximatelyZero(prob1)) {
          probVector[idx1] = prob1;
        }

        // probabilities have been written -> this path is done
        dd.decRef(state);
        return;
      }

      const bool nonZeroP0 = !RealNumber::approximatelyZero(pzero);
      const bool nonZeroP1 = !RealNumber::approximatelyZero(pone);

      // in case both outcomes are non-zero the reference count of the state has
      // to be increased once more in order to avoid reference counting errors
      if (nonZeroP0 && nonZeroP1) {
        dd.incRef(state);
      }

      // recursive case -- outcome 0
      if (nonZeroP0) {
        // save measurement result
        measurements[bit] = '0';
        // determine accumulated probability
        auto probability = commonFactor * pzero;
        // determine the next iteration point
        auto nextIt = it + 1;
        // actually collapse the state
        auto measuredState = state;
        dd.performCollapsingMeasurement(
            measuredState, static_cast<Qubit>(permutation.at(target)), pzero,
            true);
        // recursive call from here
        extractProbabilityVectorRecursive(qc, measuredState, nextIt,
                                          permutation, measurements,
                                          probability, probVector, dd);
      }

      // recursive case -- outcome 1
      if (nonZeroP1) {
        // save measurement result
        measurements[bit] = '1';
        // determine accumulated probability
        auto probability = commonFactor * pone;
        // determine the next iteration point
        auto nextIt = it + 1;
        // actually collapse the state
        auto measuredState = state;
        dd.performCollapsingMeasurement(
            measuredState, static_cast<Qubit>(permutation.at(target)), pone,
            false);
        // recursive call from here
        extractProbabilityVectorRecursive(qc, measuredState, nextIt,
                                          permutation, measurements,
                                          probability, probVector, dd);
      }

      // everything is said and done
      return;
    }
  }
}
} // namespace

template <class Config>
void extractProbabilityVector(const QuantumComputation& qc, const VectorDD& in,
                              SparsePVec& probVector, Package<Config>& dd) {
  auto permutation = qc.initialLayout;
  dd.incRef(in);
  extractProbabilityVectorRecursive(qc, in, qc.begin(), permutation,
                                    std::map<std::size_t, char>{}, 1.,
                                    probVector, dd);
}

template std::map<std::string, std::size_t>
sample<DDPackageConfig>(const QuantumComputation& qc, const VectorDD& in,
                        Package<>& dd, std::size_t shots, std::size_t seed);
template void extractProbabilityVector<DDPackageConfig>(
    const QuantumComputation& qc, const VectorDD& in, SparsePVec& probVector,
    Package<>& dd);
} // namespace dd
