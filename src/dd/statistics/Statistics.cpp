/*
 * Copyright (c) 2025 Chair for Design Automation, TUM
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "dd/statistics/Statistics.hpp"

#include <nlohmann/json.hpp>
#include <string>

namespace dd {

nlohmann::basic_json<> Statistics::json() const { return nlohmann::json{}; }

std::string Statistics::toString() const { return json().dump(2U); }

} // namespace dd
