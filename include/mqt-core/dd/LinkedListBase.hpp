/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include "dd/DDDefinitions.hpp"

#include <array>
#include <cstddef>
namespace dd {

/**
 * @brief Members required UniqueTable and MemoryManager
 */
struct LLBase {
  /**
   * @brief The pointer tho the next object
   * @details The next pointer is used to from linked lists. Classes that should
   * be used in a linked list must solely inherit from this class. Other code in
   * mqt-core relies on this assumption that all objects in a linked list are of
   * the exact same type.
   */

  LLBase* next_ = nullptr; // used to link nodes in unique table

  /**
   * @brief default method to get the next object
   * @details Classes that inherit from LLBase should implement their own next()
   * method to return the next object in the list with a specialized return
   * type.
   * @return LLBase*
   */
  LLBase* next() const noexcept { return next_; }

  // set the pointer to the next object
  void setNext(LLBase* n) noexcept { next_ = n; }
};

} // namespace dd
