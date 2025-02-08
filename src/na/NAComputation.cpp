/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "na/NAComputation.hpp"

#include "na/operations/LoadOp.hpp"
#include "na/operations/LocalOp.hpp"
#include "na/operations/ShuttlingOp.hpp"
#include "na/operations/StoreOp.hpp"

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace na {
auto NAComputation::toString() const -> std::string {
  std::stringstream ss;
  for (const auto& [atom, loc] : initialLocations) {
    ss << "atom " << loc << " " << *atom << "\n";
  }
  for (const auto& op : *this) {
    ss << op << "\n";
  }
  return ss.str();
}
auto NAComputation::validate() const -> bool {
  std::size_t counter = 1; // the first operation is `init at ...;`
  std::unordered_map<const Atom*, Location> currentLocations = initialLocations;
  std::unordered_set<const Atom*> currentlyShuttling{};
  for (const auto& op : *this) {
    ++counter;
    if (op.is<ShuttlingOp>()) {
      const auto& shuttlingOp = op.as<ShuttlingOp>();
      const auto& atoms = shuttlingOp.getAtoms();
      if (shuttlingOp.is<LoadOp>()) {
        if (std::any_of(atoms.begin(), atoms.end(),
                        [&currentLocations](const auto* atom) {
                          return currentLocations.find(atom) !=
                                 currentLocations.end();
                        })) {
          std::cout << "Error in op number " << counter
                    << " (atom already loaded)\n";
          return false;
        }
        std::for_each(atoms.begin(), atoms.end(),
                      [&currentlyShuttling](const auto* atom) {
                        currentlyShuttling.insert(atom);
                      });
      } else {
        if (std::any_of(atoms.begin(), atoms.end(),
                        [&currentlyShuttling](const auto* atom) {
                          return currentlyShuttling.find(atom) ==
                                 currentlyShuttling.end();
                        })) {
          std::cout << "Error in op number " << counter
                    << " (atom not loaded)\n";
          return false;
        }
      }
      if ((op.is<LoadOp>() && op.as<LoadOp>().hasTargetLocations()) ||
          (op.is<LoadOp>() && op.as<StoreOp>().hasTargetLocations())) {
        const auto& targetLocations = shuttlingOp.getTargetLocations();
        for (std::size_t i = 0; i < atoms.size(); ++i) {
          const auto* a = atoms[i];
          for (std::size_t j = i + 1; j < atoms.size(); ++j) {
            const auto* b = atoms[j];
            const auto& s1 = currentLocations[a];
            const auto& s2 = currentLocations[b];
            const auto& e1 = targetLocations[i];
            const auto& e2 = targetLocations[j];
            if (e1 == e2) {
              std::cout << "Error in op number " << counter
                        << " (two end points identical)\n";
              return false;
            }
            if (s1.x == s2.x && e1.x != e2.x) {
              std::cout << "Error in op number " << counter
                        << " (columns not preserved)\n";
              return false;
            }
            if (s1.y == s2.y && e1.y != e2.y) {
              std::cout << "Error in op number " << counter
                        << " (rows not preserved)\n";
              return false;
            }
            if (s1.x < s2.x && e1.x >= e2.x) {
              std::cout << "Error in op number " << counter
                        << " (column order not preserved)\n";
              return false;
            }
            if (s1.y < s2.y && e1.y >= e2.y) {
              std::cout << "Error in op number " << counter
                        << " (row order not preserved)\n";
              return false;
            }
            if (s1.x > s2.x && e1.x <= e2.x) {
              std::cout << "Error in op number " << counter
                        << " (column order not preserved)\n";
              return false;
            }
            if (s1.y > s2.y && e1.y <= e2.y) {
              std::cout << "Error in op number " << counter
                        << " (row order not preserved)\n";
              return false;
            }
          }
        }
        for (std::size_t i = 0; i < atoms.size(); ++i) {
          currentLocations[atoms[i]] = targetLocations[i];
        }
      }
      if (shuttlingOp.is<StoreOp>()) {
        std::for_each(atoms.begin(), atoms.end(), [&](const auto* atom) {
          currentlyShuttling.erase(atom);
        });
      }
    }
  }
  return true;
}
} // namespace na
