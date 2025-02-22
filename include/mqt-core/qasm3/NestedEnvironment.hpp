/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include <map>
#include <optional>
#include <string>
#include <vector>

template <typename T> class NestedEnvironment {
  std::vector<std::map<std::string, T>> env{};

public:
  NestedEnvironment() { env.emplace_back(); };

  void push() { env.emplace_back(); }

  void pop() { env.pop_back(); }

  std::optional<T> find(std::string key) {
    for (auto it = env.rbegin(); it != env.rend(); ++it) {
      auto found = it->find(key);
      if (found != it->end()) {
        return found->second;
      }
    }
    return std::nullopt;
  }

  void emplace(std::string key, T value) { env.back().emplace(key, value); }
};
