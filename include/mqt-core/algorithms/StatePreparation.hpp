/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include "ir/QuantumComputation.hpp"

#include <complex>
#include <vector>

namespace qc {
/**
 * Prepares a generic Quantum State from a list of normalized complex
 amplitudes
 * Adapted implementation of Qiskit State Preparation:
 *
 https://github.com/Qiskit/qiskit/blob/main/qiskit/circuit/library/data_preparation/state_preparation.py#
 * based on paper:
 *      Shende, Bullock, Markov. Synthesis of Quantum Logic Circuits (2004)
        [`https://ieeexplore.ieee.org/document/1629135`]
 * */

/**
 * @throws invalid_argument when amplitudes are not normalized or length not
 * power of 2
 * @param list of complex amplitudes to initialize to
 * @return MQT Circuit that initializes a state
 * */
[[nodiscard]] auto
createStatePreparationCircuit(std::vector<std::complex<double>>& amplitudes)
    -> QuantumComputation;
} // namespace qc
