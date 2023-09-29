#pragma once

#include "dd/Edge.hpp"

namespace dd {

// forward declaration
struct vNode;

/**
 * @brief Recursively transform a given DD into a state vector
 * @param root the root node of the DD
 * @param amp the accumulated amplitude from previous traversals
 * @param i the current index into the vector
 * @param vec pointer to the state vector array
 * @details This method intentionally takes a raw pointer for the data array so
 * that the implementation can be reused in the Python bindings
 */
void getVectorFromDD(const Edge<vNode>& root, const std::complex<fp>& amp,
                     std::size_t i, std::complex<dd::fp>* vec);

/**
 * @brief Transform a vector DD into a state vector
 * @param root the root of the vector DD
 * @return the state vector
 */
CVec getVectorFromDD(const Edge<vNode>& root);

/**
 * @brief Print a vector representation of a vector DD
 * @param e the root of the vector DD
 */
void printVector(const Edge<vNode>& e);

/**
 * @brief Get the value of a certain amplitude in a vector DD
 * @param e the root of the vector DD
 * @param i the index of the amplitude
 * @return the amplitude a_i of the vector DD
 */
std::complex<fp> getValueByIndex(const Edge<vNode>& e, std::size_t i);

/**
 * @brief Recursively get the value of a certain amplitude in a vector DD
 * @param e the current node
 * @param amp the accumulated amplitude from previous traversals
 * @param i the index of the amplitude
 * @return the amplitude a_i of the vector DD
 */
std::complex<fp> getValueByIndex(const Edge<vNode>& e,
                                 const std::complex<fp>& amp, std::size_t i);

/**
 * @brief Get the norm of a complex-valued vector
 * @param vec the vector
 * @return the norm of the vector
 */
fp getVectorNorm(const CVec& vec);
} // namespace dd
