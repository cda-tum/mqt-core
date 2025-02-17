/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include "dd/Package_fwd.hpp"
#include "ir/QuantumComputation.hpp"

namespace dd {
using namespace qc;

/**
 * @brief Sequentially build a decision diagram representation for the
 * functionality of a purely-quantum @ref QuantumComputation.
 *
 * @details For a circuit $G$ with $|G|$ gates $g_0, g_1, \ldots, g_{|G|-1}$,
 * the functionality of $G$ is defined as the unitary matrix $U$ such that
 * $$ U = U_{|G|-1}) \cdot U_{|G|-2} \cdot \ldots \cdot U_1 \cdot U_0, $$
 * where $U_i$ is the unitary matrix corresponding to gate $g_i$. For an
 * $n$-qubit quantum computation, $U$ is a $2^n \times 2^n$ matrix.
 *
 * By representing every single operation in the circuit as a decision diagram
 * instead of a unitary matrix and performing the matrix multiplication directly
 * using decision diagrams, a representation of the functionality of a quantum
 * computation can oftentimes be computed more efficiently in terms of memory
 * and runtime.
 *
 * This function effectively computes
 * $$ DD(U) = DD(g_{|G|-1}) \otimes DD(g_{|G|-2}) \otimes \ldots \otimes DD(g_0)
 * $$ by sequentially applying the decision diagrams of the gates in the circuit
 * to the current decision diagram representing the functionality of the quantum
 * computation.
 *
 * @param qc The quantum computation to construct the functionality for
 * @param dd The DD package to use for the construction
 * @tparam Config The configuration of the DD package
 * @return The matrix diagram representing the functionality of the quantum
 * computation
 */
template <class Config>
MatrixDD buildFunctionality(const QuantumComputation& qc, Package<Config>& dd);

/**
 * @brief Recursively build a decision diagram representation for the
 * functionality of a purely-quantum @ref QuantumComputation.
 *
 * @see buildFunctionality
 * @details Instead of sequentially applying the decision diagrams of the gates
 * in the circuit, this function builds a binary computation tree out of the
 * decision diagrams of the gates in the circuit.
 * This results in a recursive pairwise grouping that can be more memory and
 * runtime efficient compared to the sequential approach.
 * @see https://arxiv.org/abs/2103.08281
 *
 * @param qc The quantum computation to construct the functionality for
 * @param dd The DD package to use for the construction
 * @tparam Config The configuration of the DD package
 * @return The matrix diagram representing the functionality of the quantum
 * computation
 */
template <class Config>
MatrixDD buildFunctionalityRecursive(const QuantumComputation& qc,
                                     Package<Config>& dd);

} // namespace dd
