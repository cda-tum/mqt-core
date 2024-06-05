#pragma once

#include <cstddef>
#include <vector>
/**
 * @brief Symmetric matrix class with same number of rows and columns that
 * allows access by row and column but uses less memory than a full matrix
 */
template <typename T> class SymmetricMatrix {
private:
  std::vector<std::vector<T>> data;

public:
  // Constructors
  SymmetricMatrix() = default;
  explicit SymmetricMatrix(const size_t size) {
    data.resize(size);
    for (size_t i = 0; i < size; ++i) {
      data[i].resize(i + 1);
    }
  }

  SymmetricMatrix(const size_t size, const T& value) {
    data.resize(size);
    for (size_t i = 0; i < size; ++i) {
      data[i].resize(i + 1, value);
    }
  }

  [[nodiscard]] const T& operator()(const size_t row, const size_t col) const {
    if (row < col) {
      return data[col][row];
    }
    return data[row][col];
  }

  [[nodiscard]] T& operator()(const size_t row, const size_t col) {
    if (row < col) {
      return data[col][row];
    }
    return data[row][col];
  }

  [[nodiscard]] size_t size() const { return data.size(); }
};
