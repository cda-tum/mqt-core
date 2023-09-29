#pragma once

#include "dd/Edge.hpp"

namespace dd {

// forward declaration
struct dNode;

/**
 * @brief Recursively transform a given DD into a density matrix
 * @param e the root node of the DD
 * @param amp the accumulated amplitude from previous traversals
 * @param i the current row index into the matrix
 * @param j the current column index into the matrix
 * @param mat the matrix to be filled
 */
void getDensityMatrixFromDD(Edge<dNode>& e, const std::complex<fp>& amp,
                            std::size_t i, std::size_t j, CMat& mat);

/**
 * @brief Transform a density matrix DD into a density matrix
 * @param e the root of the density matrix DD
 * @return the density matrix
 */
CMat getDensityMatrixFromDD(Edge<dNode>& e);

} // namespace dd
