#pragma once

#include "dd/Edge.hpp"

namespace dd {

// forward declaration
struct mNode;

/**
 * @brief Recursively transform a given DD into a matrix
 * @param e the root node of the DD
 * @param amp the accumulated amplitude from previous traversals
 * @param i the current row index into the matrix
 * @param j the current column index into the matrix
 * @param mat the matrix to be filled
 */
void getMatrixFromDD(const Edge<mNode>& e, const std::complex<fp>& amp,
                     std::size_t i, std::size_t j, CMat& mat);

/**
 * @brief Transform a matrix DD into a matrix
 * @param e the root of the matrix DD
 * @return the matrix
 */
CMat getMatrixFromDD(const Edge<mNode>& e);

/**
 * @brief Print a matrix representation of a matrix DD
 * @param e the root of the matrix DD
 */
void printMatrix(const Edge<mNode>& e);

/**
 * @brief Get the value of a certain entry in a matrix DD
 * @param e the root of the matrix DD
 * @param i the row index of the amplitude
 * @param j the column index of the amplitude
 * @return the entry U_{i,j} of the matrix DD
 */
std::complex<fp> getValueByIndex(const Edge<mNode>& e, std::size_t i,
                                 std::size_t j);

/**
 * @brief Recursively get the value of a certain entry in a matrix DD
 * @param e the current node
 * @param amp the accumulated amplitude from previous traversals
 * @param i the row index of the amplitude
 * @param j the column index of the amplitude
 * @return the entry U_{i,j} of the matrix DD
 */
std::complex<fp> getValueByIndex(const Edge<mNode>& e,
                                 const std::complex<fp>& amp, std::size_t i,
                                 std::size_t j);

} // namespace dd
