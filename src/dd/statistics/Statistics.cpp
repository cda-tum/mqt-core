#include "dd/statistics/Statistics.hpp"

#include <nlohmann/json.hpp>

namespace dd {

nlohmann::json Statistics::json() const { return nlohmann::json{}; }

std::string Statistics::toString() const { return json().dump(2U); }

} // namespace dd
