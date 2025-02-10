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
  std::vector<std::pair<const Atom*, Location>> initialLocationsAsc(
      initialLocations.begin(), initialLocations.end());
  std::sort(initialLocationsAsc.begin(), initialLocationsAsc.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; });
  for (const auto& [atom, loc] : initialLocationsAsc) {
    ss << "atom " << loc << " " << *atom << "\n";
  }
  for (const auto& op : *this) {
    ss << *op << "\n";
  }
  return ss.str();
}
auto NAComputation::validate() const -> bool {
  std::size_t counter = 1; // the first operation is `init at ...;`
  if (atoms.size() != initialLocations.size()) {
    std::cout << "Number of atoms and initial locations must be equal\n";
  }
  std::unordered_map<const Atom*, Location> currentLocations = initialLocations;
  std::unordered_set<const Atom*> currentlyShuttling{};
  for (const auto& op : *this) {
    ++counter;
    if (op->is<ShuttlingOp>()) {
      const auto& shuttlingOp = op->as<ShuttlingOp>();
      const auto& opAtoms = shuttlingOp.getAtoms();
      if (shuttlingOp.is<LoadOp>()) {
        if (std::any_of(opAtoms.begin(), opAtoms.end(),
                        [&currentlyShuttling](const auto* atom) {
                          return currentlyShuttling.find(atom) !=
                                 currentlyShuttling.end();
                        })) {
          std::cout << "Error in op number " << counter
                    << " (atom already loaded)\n";
          return false;
        }
        std::for_each(opAtoms.begin(), opAtoms.end(),
                      [&currentlyShuttling](const auto* atom) {
                        currentlyShuttling.insert(atom);
                      });
      } else {
        if (std::any_of(opAtoms.begin(), opAtoms.end(),
                        [&currentlyShuttling](const auto* atom) {
                          return currentlyShuttling.find(atom) ==
                                 currentlyShuttling.end();
                        })) {
          std::cout << "Error in op number " << counter
                    << " (atom not loaded)\n";
          return false;
        }
      }
      if ((op->is<LoadOp>() && op->as<LoadOp>().hasTargetLocations()) ||
          (op->is<StoreOp>() && op->as<StoreOp>().hasTargetLocations())) {
        const auto& targetLocations = shuttlingOp.getTargetLocations();
        for (std::size_t i = 0; i < opAtoms.size(); ++i) {
          const auto* a = opAtoms[i];
          for (std::size_t j = i + 1; j < opAtoms.size(); ++j) {
            const auto* b = opAtoms[j];
            if (a == b) {
              std::cout << "Error in op number " << counter
                        << " (two atoms identical)\n";
              return false;
            }
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
        for (std::size_t i = 0; i < opAtoms.size(); ++i) {
          currentLocations[opAtoms[i]] = targetLocations[i];
        }
      }
      if (shuttlingOp.is<StoreOp>()) {
        std::for_each(opAtoms.begin(), opAtoms.end(), [&](const auto* atom) {
          currentlyShuttling.erase(atom);
        });
      }
    } else if (op->is<LocalOp>()) {
      const auto& opAtoms = op->as<LocalOp>().getAtoms();
      for (std::size_t i = 0; i < opAtoms.size(); ++i) {
        const auto* a = opAtoms[i];
        for (std::size_t j = i + 1; j < opAtoms.size(); ++j) {
          if (const auto* b = opAtoms[j]; a == b) {
            std::cout << "Error in op number " << counter
                      << " (two atoms identical)\n";
            return false;
          }
        }
      }
    }
  }
  return true;
}
} // namespace na
