#pragma once

#include <cstdint>
#include <vector>
/**
 * @brief Symmetric matrix class with same number of rows and columns that
 * allows access by row and column but uses less memory than a full matrix
 */
template <typename dataType> class SymmetricMatrix {
private:
  std::vector<std::vector<dataType>> data;
  uint32_t size = 0;

public:
  // Constructors
  SymmetricMatrix() = default;
  explicit SymmetricMatrix(const uint32_t size) : size(size) {
    data.resize(size);
    for (uint32_t i = 0; i < size; ++i) {
      data[i].resize(i + 1);
    }
  }

  SymmetricMatrix(const uint32_t size, const dataType value) : size(size) {
    data.resize(size);
    for (uint32_t i = 0; i < size; ++i) {
      data[i].resize(i + 1, value);
    }
  }

  [[nodiscard]] dataType& operator()(const uint32_t row, const uint32_t col) {
    if (row < col) {
      return data[col][row];
    }
    return data[row][col];
  }

  [[nodiscard]] dataType operator()(const uint32_t row,
                                    const uint32_t col) const {
    if (row < col) {
      return data[col][row];
    }
    return data[row][col];
  }

  [[nodiscard]] uint32_t getSize() const { return size; }
};
