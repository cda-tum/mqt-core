#pragma once

#include "CircuitOptimizer.hpp"
#include "QuantumComputation.hpp"

#include <cmath>
#include <complex>
#include <utility>

/**
 * Class to prepare a generic Quantum State from a list of normalized complex
 amplitudes
 * Adapted implementation of Qiskit State Preparation:
 *
 https://github.com/Qiskit/qiskit/blob/main/qiskit/circuit/library/data_preparation/state_preparation.py#
 * based on paper:
 *      Shende, Bullock, Markov. Synthesis of Quantum Logic Circuits (2004)
        [`https://ieeexplore.ieee.org/document/1629135`]
 * */
static const double EPS = 1e-10;

class StatePreparation {
  using Matrix = std::vector<std::vector<double>>;

public:
  /**
   * @throws invalid_argument when amplitudes are not normalized or length not
   * power of 2
   * @param list of complex amplitudes to initialize to
   * @return MQT Circuit that initializes a state
   * */
  static qc::QuantumComputation
  prepareState(const std::vector<std::complex<double>>& amplitudes) {

    if (!isNormalized(amplitudes)) {
      throw std::invalid_argument{
          "Using State Preparation with Amplitudes that are not normalized"};
    }

    // get number of qubits needed
    double const numQubits = std::log2(amplitudes.size());

    if (numQubits == 0 || std::floor(numQubits) != numQubits) {
      throw std::invalid_argument{
          "Using State Preparation with vector size that is not a power of 2"};
    }

    qc::QuantumComputation toZeroCircuit =
        gatesToUncompute(amplitudes, static_cast<size_t>(numQubits));

    // invert circuit
    qc::CircuitOptimizer::flattenOperations(toZeroCircuit);
    toZeroCircuit.invert();

    return toZeroCircuit;
  }

private:
  // check whether vector is normalized
  template <typename T> static bool isNormalized(std::vector<T> vec) {
    return std::abs(1 - twoNorm(vec)) < EPS;
  }

  template <typename T> static double twoNorm(std::vector<T> vec) {
    double norm = 0;
    for (auto elem : vec) {
      norm += std::norm(elem);
    }
    return sqrt(norm);
  }

  // kronecker product of two matrices with form vector<vector> where inner
  // vectors represent rows
  static Matrix kroneckerProduct(Matrix matrixA, Matrix matrixB) {
    size_t const rowA = matrixA.size();
    size_t const rowB = matrixB.size();
    size_t const colA = matrixA[0].size();
    size_t const colB = matrixB[0].size();
    // initialize size
    Matrix newMatrix{(rowA * rowB), std::vector<double>(colA * colB, 0)};
    // code taken from RosettaCode slightly adapted
    for (size_t i = 0; i < rowA; ++i) {
      // k loops till rowB
      for (size_t j = 0; j < colA; ++j) {
        // j loops till colA
        for (size_t k = 0; k < rowB; ++k) {
          // l loops till colB
          for (size_t l = 0; l < colB; ++l) {
            // Each element of matrix A is
            // multiplied by whole Matrix B
            // resp and stored as Matrix C
            newMatrix[i * rowB + k][j * colB + l] =
                matrixA[i][j] * matrixB[k][l];
          }
        }
      }
    }
    return newMatrix;
  }

  static Matrix createIdentity(size_t size) {
    Matrix identity{
        std::vector<std::vector<double>>(size, std::vector<double>(size, 0))};
    for (size_t i = 0; i < size; ++i) {
      identity[i][i] = 1;
    }
    return identity;
  }

  static std::vector<double> matrixVectorProd(const Matrix& matrix,
                                              std::vector<double> vector) {
    std::vector<double> result;
    for (const auto& matrixVec : matrix) {
      double sum{0};
      for (size_t i = 0; i < matrixVec.size(); ++i) {
        sum += matrixVec[i] * vector[i];
      }
      result.push_back(sum);
    }
    return result;
  }

  // creates circuit that takes desired vector to zero
  static qc::QuantumComputation
  gatesToUncompute(std::vector<std::complex<double>> amplitudes,
                   size_t numQubits) {
    qc::QuantumComputation disentangler{numQubits};
    for (size_t i = 0; i < numQubits; ++i) {
      // rotations to disentangle LSB
      auto [remainingParams, thetas, phis] = rotationsToDisentangle(amplitudes);
      amplitudes = remainingParams;
      // perform required rotations
      bool addLastCnot = true;
      double const phisNorm = twoNorm(phis);
      double const thetasNorm = twoNorm(thetas);
      if (phisNorm != 0 && thetasNorm != 0) {
        addLastCnot = false;
      }
      if (phisNorm != 0) {
        // call multiplex with RZGate
        qc::QuantumComputation rzMultiplexer =
            multiplex(qc::OpType{qc::RZ}, phis, addLastCnot);
        // append rzMultiplexer to disentangler, but it should only attach on
        // qubits i-numQubits, thus "i" is added to the local qubit indices
        for (auto& op : rzMultiplexer) {
          for (auto& target : op->getTargets()) {
            target += static_cast<unsigned int>(i);
          }
          for (auto control : op->getControls()) {
            // there were some errors when accessing the qubit directly and
            // adding to it
            op->setControls(qc::Controls{
                qc::Control{control.qubit + static_cast<unsigned int>(i)}});
          }
        }
        disentangler.emplace_back<qc::Operation>(rzMultiplexer.asOperation());
      }
      if (thetasNorm != 0) {
        // call multiplex with RYGate
        qc::QuantumComputation ryMultiplexer =
            multiplex(qc::OpType{qc::RY}, thetas, addLastCnot);
        // append reversed ry_multiplexer to disentangler, but it should only
        // attach on qubits i-numQubits, thus "i" is added to the local qubit
        // indices
        std::reverse(ryMultiplexer.begin(), ryMultiplexer.end());
        for (auto& op : ryMultiplexer) {
          for (auto& target : op->getTargets()) {
            target += static_cast<unsigned int>(i);
          }
          for (auto control : op->getControls()) {
            // there were some errors when accessing the qubit directly and
            // adding to it
            op->setControls(qc::Controls{
                qc::Control{control.qubit + static_cast<unsigned int>(i)}});
          }
        }
        disentangler.emplace_back<qc::Operation>(ryMultiplexer.asOperation());
      }
    }
    // adjust global phase according to the last e^(it)
    double const arg = -std::arg(std::accumulate(
        amplitudes.begin(), amplitudes.end(), std::complex<double>(0, 0)));
    if (arg != 0) {
      disentangler.gphase(arg);
    }
    return disentangler;
  }

  // works out Ry and Rz rotation angles used to disentangle LSB qubit
  // rotations make up block diagonal matrix U
  static std::tuple<std::vector<std::complex<double>>, std::vector<double>,
                    std::vector<double>>
  rotationsToDisentangle(std::vector<std::complex<double>> amplitudes) {
    std::vector<std::complex<double>> remainingVector;
    std::vector<double> thetas;
    std::vector<double> phis;
    for (size_t i = 0; i < (amplitudes.size() / 2); ++i) {
      auto [remains, theta, phi] =
          blochAngles(amplitudes[2 * i], amplitudes[2 * i + 1]);
      remainingVector.push_back(remains);
      // minus sign because we move it to zero
      thetas.push_back(-theta);
      phis.push_back(-phi);
    }
    return {remainingVector, thetas, phis};
  }

  // work out rotations to create passed-in qubit from zero vector
  static std::tuple<std::complex<double>, double, double>
  blochAngles(std::complex<double> const complexA,
              std::complex<double> const complexB) {
    double theta{0};
    double phi{0};
    double finalT{0};
    double const magA = std::abs(complexA);
    double const magB = std::abs(complexB);
    double const finalR = sqrt(pow(magA, 2) + pow(magB, 2));
    if (finalR > EPS) {
      theta = 2 * acos(magA / finalR);
      double const aAngle = std::arg(complexA);
      double const bAngle = std::arg(complexB);
      finalT = aAngle + bAngle;
      phi = bAngle - aAngle;
    }
    return {finalR * exp(std::complex<double>{0, 1} * finalT / 2.), theta, phi};
  }

  // recursive implementation that returns multiplexer circuit
  /**
   * @param target_gate : Ry or Rz gate to apply to target qubit, multiplexed
   * over all other "select" qubits
   * @param angles : list of rotation angles to apply Ry and Rz
   * @param lastCnot : add last cnot if true
   * @return multiplexer circuit as QuantumComputation
   */
  static qc::QuantumComputation
  multiplex(qc::OpType targetGate, std::vector<double> angles, bool lastCnot) {
    size_t const listLen = angles.size();
    double const localNumQubits =
        std::floor(std::log2(static_cast<double>(listLen))) + 1;
    qc::QuantumComputation multiplexer{static_cast<size_t>(localNumQubits)};
    // recursion base case
    if (localNumQubits == 1) {
      multiplexer.emplace_back<qc::StandardOperation>(
          multiplexer.getNqubits(), qc::Controls{}, 0, targetGate,
          std::vector{angles[0]});
      return multiplexer;
    }

    Matrix const matrix{std::vector<double>{0.5, 0.5},
                        std::vector<double>{0.5, -0.5}};
    Matrix const identity =
        createIdentity(static_cast<size_t>(pow(2., localNumQubits - 2.)));
    Matrix const angleWeights = kroneckerProduct(matrix, identity);

    angles = matrixVectorProd(angleWeights, angles);

    std::vector<double> const angles1{
        std::make_move_iterator(angles.begin()),
        std::make_move_iterator(angles.begin() +
                                static_cast<int64_t>(listLen) / 2)};
    qc::QuantumComputation multiplex1 = multiplex(targetGate, angles1, false);

    // append multiplex1 to multiplexer
    multiplexer.emplace_back<qc::Operation>(multiplex1.asOperation());
    // flips the LSB qubit, control on MSB
    multiplexer.cx(0, static_cast<qc::Qubit>(localNumQubits - 1));

    std::vector<double> const angles2{std::make_move_iterator(angles.begin()) +
                                          static_cast<int64_t>(listLen) / 2,
                                      std::make_move_iterator(angles.end())};
    qc::QuantumComputation multiplex2 = multiplex(targetGate, angles2, false);

    // extra efficiency by reversing (!= inverting) second multiplex
    if (listLen > 1) {
      multiplex2.reverse();
      multiplexer.emplace_back<qc::Operation>(multiplex2.asOperation());
    } else {
      multiplexer.emplace_back<qc::Operation>(multiplex2.asOperation());
    }

    if (lastCnot) {
      multiplexer.cx(0, static_cast<qc::Qubit>(localNumQubits - 1));
    }

    qc::CircuitOptimizer::flattenOperations(multiplexer);
    return multiplexer;
  }
};
