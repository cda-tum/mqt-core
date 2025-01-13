#pragma once

#include "CircuitOptimizer.hpp"
#include "QuantumComputation.hpp"

#include <cmath>
#include <complex>
#include <utility>

namespace qc {
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
class StatePreparation : public QuantumComputation {

public:
    explicit StatePreparation(const std::vector<std::complex<double>>& amplitudes);

private:
  template <typename T> static bool isNormalized(std::vector<T> vec);
  template <typename T> static double twoNorm(std::vector<T> vec);
  static std::vector<std::vector<double>> kroneckerProduct(std::vector<std::vector<double>> matrixA, std::vector<std::vector<double>> matrixB);
  static std::vector<std::vector<double>> createIdentity(size_t size);
  static std::vector<double> matrixVectorProd(const std::vector<std::vector<double>>& matrix,
                                              std::vector<double> vector);
  static qc::QuantumComputation
  gatesToUncompute(std::vector<std::complex<double>> amplitudes,
                   size_t numQubits);
  static std::tuple<std::vector<std::complex<double>>, std::vector<double>,
                    std::vector<double>>
  rotationsToDisentangle(std::vector<std::complex<double>> amplitudes);
  static std::tuple<std::complex<double>, double, double>
  blochAngles(std::complex<double> const complexA,
              std::complex<double> const complexB);
  static qc::QuantumComputation
  multiplex(qc::OpType targetGate, std::vector<double> angles, bool lastCnot);
};
} // namespace qc
